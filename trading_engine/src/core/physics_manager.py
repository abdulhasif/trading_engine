"""
src/live/physics_manager.py - Renko & Splicer Management
=========================================================
Handles the conversion of raw ticks to bricks and manages historical state.
STRICT LOGIC PRESERVATION from engine_main.py.
"""

import logging
import pandas as pd
from datetime import datetime
from trading_core.core.config import base_config as config
from trading_core.core.physics.renko import LiveRenkoState

logger = logging.getLogger(__name__)

class PhysicsManager:
    def __init__(self, universe_df, exec_guard):
        self.universe = universe_df
        self.exec_guard = exec_guard
        self.renko_states = {}
        self.sector_renko = {}
        self.brick_sizes = {}

    def warmup_brick_sizes(self):
        """STRICT: Verbatim from engine_main.py."""
        logger.info("WARMUP - Calculating brick sizes")
        for _, row in self.universe.iterrows():
            sym, sec = row["symbol"], row["sector"]
            stock_dir = config.DATA_DIR / sec / sym
            if stock_dir.exists():
                pqs = sorted(stock_dir.glob("*.parquet"))
                if pqs:
                    try:
                        df = pd.read_parquet(pqs[-1])
                        if not df.empty:
                            self.brick_sizes[sym] = df["brick_close"].iloc[-1] * config.NATR_BRICK_PERCENT
                            continue
                    except Exception:
                        pass
            self.brick_sizes[sym] = 500 * config.NATR_BRICK_PERCENT  # fallback
        logger.info(f"Brick sizes ready for {len(self.brick_sizes)} symbols")
        return self.brick_sizes

    def initialize_states(self, stocks, indices):
        """STRICT: Initialize LiveRenkoState and connect to ExecGuard."""
        for _, r in stocks.iterrows():
            sym = r["symbol"]
            st = LiveRenkoState(sym, r["sector"], self.brick_sizes.get(sym, 1.0))
            st.load_history(config.RENKO_HISTORY_LIMIT)
            self.renko_states[sym] = st
            # Sync with ExecGuard buffer
            if sym in self.exec_guard.buffers:
                st.bricks = list(self.exec_guard.buffers[sym]._buffer)

        for _, r in indices.iterrows():
            sym = r["symbol"]
            st = LiveRenkoState(sym, r["sector"], self.brick_sizes.get(sym, 1.0))
            st.load_history(config.RENKO_HISTORY_LIMIT)
            self.sector_renko[sym] = st
            if sym in self.exec_guard.buffers:
                st.bricks = list(self.exec_guard.buffers[sym]._buffer)

    def process_sector_tick(self, sym, tick, now):
        """STRICT: Logic for processing sector index ticks."""
        st = self.sector_renko[sym]
        # Heartbeat check
        self.exec_guard.heartbeat.check_and_inject(sym, st, now)
        
        # Register tick and process
        self.exec_guard.heartbeat.register_tick(sym, tick["ltp"])
        prev_cnt = len(st.bricks)
        st.process_tick(tick["ltp"], tick["high"], tick["low"], tick["timestamp"], volume=tick.get("volume", 0))

        # Update buffers if new brick formed
        if len(st.bricks) > prev_cnt:
            new_brick = st.bricks[-1]
            self.exec_guard.buffers[sym].append(new_brick)
            self.exec_guard.splicers[sym].append_live_brick(new_brick)
            return True
        return False

    def process_stock_tick(self, sym, tick, now):
        """STRICT: Logic for processing stock ticks."""
        st = self.renko_states[sym]
        # Heartbeat check
        self.exec_guard.heartbeat.check_and_inject(sym, st, now)
        
        # Register tick and process
        self.exec_guard.heartbeat.register_tick(sym, tick["ltp"])
        prev_cnt = len(st.bricks)
        st.process_tick(tick["ltp"], tick["high"], tick["low"], tick["timestamp"], volume=tick.get("volume", 0))

        # Update buffers if new brick formed
        if len(st.bricks) > prev_cnt:
            new_brick = st.bricks[-1]
            self.exec_guard.buffers[sym].append(new_brick)
            self.exec_guard.splicers[sym].append_live_brick(new_brick)
            return True
        return False

    def get_sector_directions(self):
        """STRICT: Verbatim from engine_main.py."""
        return {
            st.sector: (st.bricks[-1]["direction"] if st.bricks else 0)
            for st in self.sector_renko.values()
        }
