"""
src/live/tick_provider.py -- Real-Time Tick Feed via Upstox WebSocket
======================================================================
Connects to Upstox Market Data WebSocket for live tick prices.
Falls back to simulated random ticks if UPSTOX_ACCESS_TOKEN is not set.

Required:  pip install upstox-python-sdk
Or:        pip install websockets protobuf requests

The provider maps trading symbols (e.g. "SBIN") to Upstox instrument keys
(e.g. "NSE_EQ|INE062A01020") using the sector_universe.csv file.

Auto-Reconnect: On any close/error the provider automatically retries
with exponential backoff (5 -> 10 -> 20 -> 40 -> 60 s cap).
"""

import time
import logging
import random
import threading
from datetime import datetime
from typing import Optional, List, Dict
from collections import deque
import pandas as pd

from trading_core.core.config import base_config as config

logger = logging.getLogger(__name__)

# --- Ultra-Low Latency Background Tick Logger ---
import csv

class AsyncTickLogger:
    def __init__(self, directory, base_filename="raw_ticks_dump", flush_interval=config.TICK_FLUSH_INTERVAL):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.base_filename = base_filename
        self.flush_interval = flush_interval
        
        # Buffer now holds tuples of (date_str, [row_data])
        # So we can group writes by date if a tick crosses midnight
        self._buffer = []
        self._lock = threading.Lock()
        
        self._thread = threading.Thread(target=self._flusher, daemon=True, name="TickLogFlusher")
        self._thread.start()

    def _get_filepath(self, date_str):
        return self.directory / f"{self.base_filename}_{date_str}.csv"

    def log_tick(self, timestamp, symbol, ltp, volume=0):
        # We assume timestamp is an ISO string like "2026-03-02T15:00:00"
        # Extract just the "YYYY-MM-DD" part for the daily file
        date_str = timestamp[:10]  
        
        with self._lock:
            self._buffer.append((date_str, [timestamp, symbol, ltp, volume]))

    def _flusher(self):
        while True:
            time.sleep(self.flush_interval)
            
            with self._lock:
                if not self._buffer:
                    continue
                to_write = self._buffer
                self._buffer = []

            # Group the rows by date_str so we only open each file once per flush
            writes_by_date = {}
            for date_str, row in to_write:
                if date_str not in writes_by_date:
                    writes_by_date[date_str] = []
                writes_by_date[date_str].append(row)
                
            for date_str, rows in writes_by_date.items():
                filepath = self._get_filepath(date_str)
                is_new = not filepath.exists()
                
                try:
                    with open(filepath, "a", newline="") as f:
                        writer = csv.writer(f)
                        if is_new:
                            writer.writerow(["timestamp", "symbol", "ltp", "volume"])
                        writer.writerows(rows)
                except Exception as e:
                    logger.error(f"Failed to flush tick logs to {filepath}: {e}")

from pathlib import Path
RAW_TICK_LOGGER = AsyncTickLogger(directory=config.DATA_DIR / "raw_ticks", flush_interval=1.0)



# =============================================================================
# UPSTOX WEBSOCKET TICK PROVIDER
# =============================================================================
class TickProvider:
    """
    Real-time tick provider for the trading engine.

    Behavior:
      - If UPSTOX_ACCESS_TOKEN is set -> connects to Upstox WebSocket
      - Otherwise -> falls back to simulated random ticks (paper testing)
      - Auto-reconnects on disconnect with exponential backoff

    Output from get_latest_ticks():
      { "SBIN": {"ltp": 625.5, "high": 626.0, "low": 624.8, "timestamp": ...}, ... }
    """

    # Exponential backoff delays in seconds (last value is the cap)
    _RECONNECT_DELAYS = config.TICK_RECONNECT_DELAYS

    def __init__(self, symbols: list[str], spoof_file: Optional[str] = None):
        self.symbols = symbols
        self._connected = False
        self._ticks: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._ws_thread: Optional[threading.Thread] = None
        self._running = False
        self._msg_count = 0

        # Reconnect state
        self._reconnect_attempt = 0
        self._reconnect_lock = threading.Lock()
        self._reconnecting = False

        # Upstox streamer (set in _connect_upstox)
        self._streamer = None

        # Build symbol <-> instrument_key mapping from universe CSV
        self._sym_to_ikey: dict[str, str] = {}
        self._ikey_to_sym: dict[str, str] = {}
        self._load_instrument_map()

        # Spoofing State (Fix 1)
        self.spoof_file = spoof_file
        self._is_spoofing = bool(spoof_file)
        self._current_spoof_time = None
        self._spoof_buffer = deque()

        if self._is_spoofing:
            self._use_live = False
            self._load_spoof_data()
        else:
            # Check if we have a real access token
            self._access_token = config.UPSTOX_ACCESS_TOKEN
            self._use_live = bool(self._access_token and self._access_token.strip())

        # Performance Fix: Pre-calculate index set for O(1) inside hot loop
        self._index_symbols = set()
        try:
            udf = pd.read_csv(config.UNIVERSE_CSV)
            self._index_symbols = set(udf[udf["is_index"].astype(bool)]["symbol"].tolist())
        except: pass

    def _load_instrument_map(self):
        """Load symbol to instrument_key mapping from sector_universe.csv."""
        import pandas as pd
        try:
            df = pd.read_csv(config.UNIVERSE_CSV)
            for _, row in df.iterrows():
                sym = row["symbol"]
                ikey = row.get("instrument_token", row.get("instrument_key", ""))
                if not ikey:
                    continue
                if sym in self.symbols:
                    self._sym_to_ikey[sym] = ikey
                    self._ikey_to_sym[ikey] = sym
            logger.info(f"Mapped {len(self._sym_to_ikey)}/{len(self.symbols)} "
                        f"symbols to instrument keys")
        except Exception as e:
            logger.warning(f"Could not load instrument map: {e}")

    def _load_spoof_data(self):
        """Build the Historical Data Loader (Fix 2)"""
        if not self.spoof_file:
            return
        logger.info(f"SPOOF: Loading historical ticks from {self.spoof_file}...")
        try:
            df = pd.read_csv(self.spoof_file)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
            
            # Filter for requested symbols
            if self.symbols:
                df = df[df["symbol"].isin(self.symbols)]
                
            for _, row in df.iterrows():
                self._spoof_buffer.append({
                    "timestamp": row["timestamp"],
                    "symbol": row["symbol"],
                    "ltp": float(row["ltp"]),
                    "volume": float(row.get("volume", 0))
                })
            
            if self._spoof_buffer:
                self._current_spoof_time = self._spoof_buffer[0]["timestamp"]
            
            logger.info(f"SPOOF: Loaded {len(self._spoof_buffer):,} ticks for playback.")
        except Exception as e:
            logger.error(f"SPOOF: Failed to load {self.spoof_file}: {e}")
            self._is_spoofing = False

    # -- Connection ----------------------------------------------------------
    def connect(self):
        """Connect to data source."""
        self._running = True
        if self._use_live:
            self._connect_upstox()
        else:
            logger.info("No UPSTOX_ACCESS_TOKEN found -- using SIMULATED ticks")
            logger.info("Set env var UPSTOX_ACCESS_TOKEN for live market data")
            self._connected = True

    def _connect_upstox(self):
        """Connect to Upstox WebSocket using the official SDK."""
        try:
            import upstox_client
            from upstox_client.feeder import MarketDataStreamerV3

            # Configure API client
            configuration = upstox_client.Configuration()
            configuration.access_token = self._access_token
            api_client = upstox_client.ApiClient(configuration)

            # Get instrument keys for our symbols
            instrument_keys = [
                self._sym_to_ikey[sym] for sym in self.symbols
                if sym in self._sym_to_ikey
            ]

            if not instrument_keys:
                logger.error("No instrument keys mapped. Check sector_universe.csv")
                self._use_live = False
                self._connected = True
                return

            logger.info(f"Connecting to Upstox WebSocket with "
                        f"{len(instrument_keys)} instruments... "
                        f"(attempt #{self._reconnect_attempt + 1})")

            # Create streamer with LTPC mode (lightest: last price + close)
            self._streamer = MarketDataStreamerV3(
                api_client,
                instrumentKeys=instrument_keys,
                mode="ltpc"
            )

            # Register event handlers
            self._streamer.on("message", self._on_message)
            self._streamer.on("open",    self._on_open)
            self._streamer.on("error",   self._on_error)
            self._streamer.on("close",   self._on_close)

            # Connect in background thread
            self._ws_thread = threading.Thread(
                target=self._streamer.connect,
                daemon=True,
                name="UpstoxWebSocket"
            )
            self._ws_thread.start()

            # Wait for connection (up to 15 seconds)
            for _ in range(150):
                if self._connected:
                    break
                time.sleep(0.1)

            if not self._connected:
                logger.warning("WebSocket connection timed out. Will retry.")
                self._schedule_reconnect()

        except ImportError:
            logger.warning("upstox-python-sdk not installed. "
                           "Run: pip install upstox-python-sdk")
            logger.info("Falling back to SIMULATED ticks")
            self._use_live = False
            self._connected = True

        except Exception as e:
            logger.error(f"Upstox WebSocket connection failed: {e}")
            if self._running:
                self._schedule_reconnect()
            else:
                logger.info("Engine stopped -- not reconnecting.")
                self._use_live = False
                self._connected = True

    # -- Auto-Reconnect ------------------------------------------------------
    def _schedule_reconnect(self):
        """
        Schedule a reconnect attempt in a background thread with
        exponential backoff (5 -> 10 -> 20 -> 40 -> 60 s cap).
        Thread-safe: at most one reconnect thread runs at a time.
        """
        with self._reconnect_lock:
            if self._reconnecting:
                return          # already a reconnect pending
            if not self._running:
                return          # engine is shutting down
            self._reconnecting = True

        delay = self._RECONNECT_DELAYS[
            min(self._reconnect_attempt, len(self._RECONNECT_DELAYS) - 1)
        ]
        self._reconnect_attempt += 1
        logger.warning(f"WebSocket closed/failed -- reconnecting in {delay}s "
                       f"(attempt #{self._reconnect_attempt})")

        def _do_reconnect():
            time.sleep(delay)
            with self._reconnect_lock:
                self._reconnecting = False
            if not self._running:
                return
            self._connected = False
            self._connect_upstox()

        t = threading.Thread(target=_do_reconnect, daemon=True,
                             name="WS-Reconnect")
        t.start()

    def _reset_reconnect_counter(self):
        """Call after a successful open to reset the backoff."""
        self._reconnect_attempt = 0

    # -- WebSocket Event Handlers --------------------------------------------
    def _on_open(self, *args, **kwargs):
        logger.info("Upstox WebSocket CONNECTED -- receiving live ticks")
        self._connected = True
        self._reset_reconnect_counter()
        # Clear stale tick timestamps from before the reconnect.
        # Old timestamps would falsely trip the circuit breaker until fresh ticks arrive.
        with self._lock:
            self._ticks.clear()
        logger.info("Tick cache cleared after reconnect -- waiting for fresh ticks")

    def _on_error(self, *args, **kwargs):
        logger.error(f"Upstox WebSocket error: {args}")
        # Don't schedule reconnect here - _on_close always fires after error

    def _on_close(self, *args, **kwargs):
        logger.warning("Upstox WebSocket connection CLOSED")
        self._connected = False
        self._schedule_reconnect()

    def _on_message(self, message):
        """Process incoming market data message from Upstox.

        Upstox SDK v3 sends messages as plain dicts:
        {
            'type': 'live_feed',
            'feeds': {
                'NSE_EQ|INE062A01020': {
                    'ltpc': {'ltp': 625.5, 'cp': 620.0, 'ltt': '...', 'ltq': '7'}
                }
            }
        }
        """
        try:
            feeds = None
            if isinstance(message, dict):
                feeds = message.get("feeds")
            elif hasattr(message, "feeds"):
                feeds = message.feeds

            if not feeds:
                return

            now = datetime.now()
            count = 0
            with self._lock:
                for ikey, feed in feeds.items():
                    sym = self._ikey_to_sym.get(ikey)
                    if not sym:
                        continue

                    # Dict-based feed (SDK v3)
                    if isinstance(feed, dict):
                        # LTPC mode
                        ltpc = feed.get("ltpc")
                        if ltpc:
                            ltp = float(ltpc.get("ltp", 0))
                            volume = float(ltpc.get("ltq", 0))
                            # TRUE TRADE FILTER: Only accept ticks where a real execution occurred
                            # OR if sym is an index (which has 0 volume)
                            is_index = sym in self._index_symbols
                            if ltp > 0 and (volume > 0 or is_index):
                                self._ticks[sym] = {
                                    "ltp": ltp,
                                    "high": ltp,
                                    "low": ltp,
                                    "close": float(ltpc.get("cp", ltp)),
                                    "volume": volume,
                                    "timestamp": now,
                                }
                                RAW_TICK_LOGGER.log_tick(now.isoformat(), sym, ltp, volume)
                                count += 1

                        # Full-feed mode (if subscribed to full)
                        ff = feed.get("ff")
                        if ff:
                            mff = ff.get("marketFF", {})
                            ltpc_data = mff.get("ltpc", {})
                            ohlc_list = mff.get("marketOHLC", {}).get("ohlc", [])
                            ohlc = ohlc_list[0] if ohlc_list else {}

                            ltp = float(ltpc_data.get("ltp", 0))
                            if ltp > 0 and ohlc:
                                self._ticks[sym] = {
                                    "ltp": ltp,
                                    "high": float(ohlc.get("high", ltp)),
                                    "low": float(ohlc.get("low", ltp)),
                                    "close": float(ohlc.get("close", ltp)),
                                    "timestamp": now,
                                }
                                RAW_TICK_LOGGER.log_tick(now.isoformat(), sym, ltp, 0)
                                count += 1

                    # Protobuf-based feed (legacy SDK)
                    else:
                        ltpc = getattr(feed, "ltpc", None)
                        if ltpc:
                            ltp = float(ltpc.ltp) if ltpc.ltp else 0.0
                            self._ticks[sym] = {
                                "ltp": ltp,
                                "high": ltp,
                                "low": ltp,
                                "close": float(ltpc.cp) if ltpc.cp else ltp,
                                "timestamp": now,
                            }
                            count += 1

            self._msg_count += 1
            if self._msg_count <= 3 or self._msg_count % 500 == 0:
                logger.info(f"WS ticks #{self._msg_count}: {count} symbols updated, "
                            f"total tracked: {len(self._ticks)}")
        except Exception as e:
            logger.warning(f"Tick parse error: {e}")

    # -- Public Interface ----------------------------------------------------
    def disconnect(self):
        """Disconnect from data source (suppresses auto-reconnect)."""
        self._running = False      # Must be first -- stops _schedule_reconnect
        if self._streamer is not None:
            try:
                self._streamer.disconnect()
            except Exception:
                pass
        self._connected = False
        logger.info("Tick provider disconnected")

    def get_latest_ticks(self) -> dict:
        """
        Returns latest tick data for all symbols.

        Returns:
            { "SBIN": {"ltp": 625.5, "high": 626.0, "low": 624.8,
                        "timestamp": datetime}, ... }
        """
        if self._is_spoofing:
            return self._get_spoofed_ticks()
        elif self._use_live:
            return self._get_live_ticks()
        else:
            return self._get_simulated_ticks()

    def get_current_time(self) -> datetime:
        """Expose the Unified Clock (Fix 3)"""
        if self._is_spoofing:
            return self._current_spoof_time or datetime.now()
        return datetime.now()

    def _get_live_ticks(self) -> dict:
        """Return latest ticks from WebSocket buffer."""
        with self._lock:
            return dict(self._ticks)  # Return copy

    def _get_spoofed_ticks(self) -> dict:
        """Update the Tick Generator playback (Fix 4)"""
        if not self._spoof_buffer:
            return self._ticks

        # Pop the next tick
        first = self._spoof_buffer.popleft()
        self._current_spoof_time = first["timestamp"]
        
        # Update cache
        self._ticks[first["symbol"]] = {
            "ltp": first["ltp"],
            "high": first["ltp"],
            "low": first["ltp"],
            "volume": first["volume"],
            "timestamp": first["timestamp"]
        }
        
        # Peeking: If multiple ticks share the exact same timestamp, pop them all
        while self._spoof_buffer and self._spoof_buffer[0]["timestamp"] == self._current_spoof_time:
            nxt = self._spoof_buffer.popleft()
            self._ticks[nxt["symbol"]] = {
                "ltp": nxt["ltp"],
                "high": nxt["ltp"],
                "low": nxt["ltp"],
                "volume": nxt["volume"],
                "timestamp": nxt["timestamp"]
            }
            
        return dict(self._ticks)

    def _get_simulated_ticks(self) -> dict:
        """Generate simulated random ticks (placeholder mode)."""
        now = datetime.now()
        ticks = {}

        for sym in self.symbols:
            # Use persistent base price per symbol for realistic simulation
            if sym not in self._ticks:
                self._ticks[sym] = {
                    "_base": random.uniform(100, 5000),
                    "ltp": 0, "high": 0, "low": 0, "timestamp": now,
                }

            base = self._ticks[sym].get("_base", random.uniform(100, 5000))
            # Random walk: +-0.3% per tick
            change = random.gauss(0, base * 0.003)
            base += change
            self._ticks[sym]["_base"] = base

            ticks[sym] = {
                "ltp": base,
                "high": base + random.uniform(0, base * 0.002),
                "low": base - random.uniform(0, base * 0.002),
                "timestamp": now,
            }
        return ticks

    @property
    def is_live(self) -> bool:
        """Whether we're connected to real market data."""
        return self._use_live and self._connected

