"""
src/live/control_state.py - Thread-Safe Global Control State
=============================================================
Single source of truth shared by:
  - The FastAPI server     (reads via asyncio coroutines)
  - The trading loop      (reads via threading in a threadpool executor)

Design:
  _thread_lock  - standard threading.Lock for the blocking trading loop.
  _async_lock   - asyncio.Lock for FastAPI route handlers.

Both locks wrap writes to CONTROL_STATE so mutations are always atomic.

Structure:
  CONTROL_STATE = {
      "GLOBAL_KILL":    bool           # True  -> square_off_all() + break loop immediately
      "GLOBAL_PAUSE":   bool           # True  -> suppress all entries across every ticker
      "PAUSED_TICKERS": set[str]       # {"RELIANCE"} -> per-ticker entry suppression
      "BIAS":           dict[str, str] # {"LT": "LONG" | "SHORT"} -> Hunter Mode
  }
"""

import asyncio
import threading

# -----------------------------------------------------------------------------
# THE CONTROL DICTIONARY
# -----------------------------------------------------------------------------
CONTROL_STATE: dict = {
    "GLOBAL_KILL":    False,  # Biometric kill -> square_off_all() + break
    "GLOBAL_PAUSE":   False,  # Engine-wide entry suppression
    "PAUSED_TICKERS": set(),  # {"RELIANCE", "LT"} -> per-ticker suppression
    "BIAS":           {},     # {"LT": "LONG"} -> Hunter Mode per ticker
}

# -----------------------------------------------------------------------------
# DUAL-MODE LOCKS
# -----------------------------------------------------------------------------

# For the blocking while-True trading loop (runs in a threadpool executor)
_thread_lock: threading.Lock = threading.Lock()

# For FastAPI async route handlers (runs on the asyncio event loop)
_async_lock: asyncio.Lock = asyncio.Lock()

