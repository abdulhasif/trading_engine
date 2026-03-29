import csv
import logging
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Optional

from trading_core.core.config import base_config as config

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# SCHEMA (one column per gate for easy filtering in Excel/Pandas)
# -----------------------------------------------------------------------------
_HEADERS = [
    "timestamp", "symbol", "sector", "price", "brick_dir", "sec_dir", "new_bricks",
    "velocity", "wick_pressure", "relative_strength", "brick_size", "duration_seconds",
    "consecutive_same", "oscillation_rate", "brain1_prob", "brain2_conv", "signal",
    "score", "structural_score", "global_kill", "global_pause", "ticker_paused", "bias", "eff_prob_thresh",
    "gate_prob", "gate_conv", "gate_rs", "gate_wick", "gate_whipsaw", "gate_losses",
    "gate_positions", "gate_time", "gate_vwap", "action", "reason", "open_positions", "live_pnl",
]

# -----------------------------------------------------------------------------
# ASYNC WORKER
# -----------------------------------------------------------------------------
class _AuditWorker(threading.Thread):
    def __init__(self, log_dir: Path):
        super().__init__(daemon=True, name="AuditLogWorker")
        self.log_dir = log_dir
        self.queue = queue.Queue()
        self._current_date = ""
        self._current_file = None

    def _get_log_file(self, date_str: str) -> Path:
        if date_str != self._current_date:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._current_date = date_str
            self._current_file = self.log_dir / f"{date_str}.csv"
            if not self._current_file.exists():
                with open(self._current_file, "w", newline="") as f:
                    csv.writer(f).writerow(_HEADERS)
        return self._current_file

    def run(self):
        while True:
            try:
                event = self.queue.get()
                if event is None: break
                
                ts_dt = event[0]
                is_sim = event[-1] # New field
                date_str = ts_dt.strftime("%Y-%m-%d")
                
                if is_sim:
                    date_str = f"SIM_{date_str}"
                    
                path = self._get_log_file(date_str)
                
                # Convert first element to string for CSV
                # Exclude is_sim from the raw CSV row
                row = [ts_dt.strftime("%Y-%m-%d %H:%M:%S")] + list(event[1:-1])
                
                with open(path, "a", newline="") as f:
                    csv.writer(f).writerow(row)
                
                self.queue.task_done()
            except Exception as e:
                logger.error(f"AuditWorker write error: {e}")

_WORKER = _AuditWorker(config.LOGS_DIR / "paper_debug")
_WORKER.start()

def log_brick_event(
    *,
    ts: datetime,
    symbol: str,
    sector: str,
    price: float,
    brick_dir: int,
    sec_dir: int,
    new_bricks: int,
    velocity: float = 0.0,
    wick_pressure: float = 0.0,
    relative_strength: float = 0.0,
    brick_size: float = 0.0,
    duration_seconds: float = 0.0,
    consecutive_same: int = 0,
    oscillation_rate: float = 0.0,
    brain1_prob: float = 0.0,
    brain2_conv: float = 0.0,
    signal: str = "",
    score: float = 0.0,
    structural_score: float = 0.0,
    global_kill: bool = False,
    global_pause: bool = False,
    ticker_paused: bool = False,
    bias: str = "",
    eff_prob_thresh: float = 0.55, # Default fallback
    gate_prob: str = "SKIP",
    gate_conv: str = "SKIP",
    gate_rs: str = "SKIP",
    gate_wick: str = "SKIP",
    gate_whipsaw: str = "SKIP",
    gate_losses: str = "SKIP",
    gate_positions: str = "SKIP",
    action: str = "",
    reason: str = "",
    open_positions: int = 0,
    live_pnl: float = 0.0,
    is_sim: bool = False,
    **kwargs # Capture any extra gate audits
) -> None:
    """Non-blocking: Puts audit data into a background queue."""
    # Build list of 36 fields to match _HEADERS
    event = (
        ts, symbol, sector, round(price, 2), brick_dir, sec_dir, new_bricks,
        round(velocity, 6), round(wick_pressure, 4), round(relative_strength, 4),
        round(brick_size, 2), round(duration_seconds, 1), consecutive_same,
        round(oscillation_rate, 4), round(brain1_prob, 6), round(brain2_conv, 2),
        signal, round(score, 2), round(structural_score, 4), int(global_kill), int(global_pause),
        int(ticker_paused), bias, round(eff_prob_thresh, 4),
        kwargs.get("gate_prob", gate_prob),
        kwargs.get("gate_conv", gate_conv),
        kwargs.get("gate_rs", gate_rs),
        kwargs.get("gate_wick", gate_wick),
        kwargs.get("gate_whipsaw", gate_whipsaw),
        kwargs.get("gate_losses", gate_losses),
        kwargs.get("gate_positions", gate_positions),
        kwargs.get("gate_time", "SKIP"),
        kwargs.get("gate_vwap", "SKIP"),
        action, reason, open_positions, round(live_pnl, 2),
        is_sim
    )
    _WORKER.queue.put(event)


