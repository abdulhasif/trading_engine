"""
src/live/engine.py - Phase 4: Real-Time Trading Engine
========================================================
Daily lifecycle: 08:50 wake -> 09:08 warmup -> 09:15 15:30 trade -> 15:35 shutdown.
Writes live_state.json every 1s for the dashboard.

Run:  python -m src.live.engine
"""

import sys
import os
os.environ["KERAS_BACKEND"] = "torch"
import json
import time
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from collections import deque

import xgboost as xgb

from trading_engine import config
from trading_core.core.physics.renko import LiveRenkoState
from trading_core.core.features import compute_features_live
from trading_core.core.risk.risk_fortress import RiskFortress
from trading_engine.src.tick_provider import TickProvider
from trading_engine.src.execution_guard import LiveExecutionGuard, SyncPendingOrderGuard
from trading_engine.src.upstox_simulator import UpstoxSimulator
from trading_api.src.api.server import compute_market_regime, _get_sentiment_feed
from trading_core.core.physics.quant_fixes import IsotonicCalibrationWrapper
from trading_core.core.risk.strategy import check_entry_gates, check_exit_conditions
from trading_engine.src.daily_logger import log_brick_event

# =============================================================================
# ENGINE CONSTANTS (Synchronized with config.py)
# =============================================================================
LONG_ENTRY_PROB_THRESH  = config.LONG_ENTRY_PROB_THRESH
SHORT_ENTRY_PROB_THRESH = config.SHORT_ENTRY_PROB_THRESH

RAW_LONG_ENTRY_PROB_THRESH  = config.RAW_LONG_ENTRY_PROB_THRESH
RAW_SHORT_ENTRY_PROB_THRESH = config.RAW_SHORT_ENTRY_PROB_THRESH

USE_CALIBRATED_MODELS    = config.USE_CALIBRATED_MODELS

ENTRY_CONV_THRESH       = config.ENTRY_CONV_THRESH
ENTRY_RS_THRESHOLD      = config.ENTRY_RS_THRESHOLD
MAX_ENTRY_WICK          = config.MAX_ENTRY_WICK
MIN_PRICE_FILTER        = config.MIN_PRICE_FILTER

# =============================================================================
# FEATURE ORDER SHIELD
# =============================================================================
EXPECTED_FEATURES = config.FEATURE_COLS

# -- Whipsaw Protection ------------------------------------------------------
MIN_CONSECUTIVE_BRICKS = config.MIN_CONSECUTIVE_BRICKS
MIN_BRICKS_TODAY       = config.MIN_BRICKS_TODAY

# -- Trading Control --------------------------------------------------------

def is_trading_active() -> bool:
    """Check if trading is paused by the user via the control file."""
    if not config.TRADE_CONTROL_FILE.exists():
        return True
    try:
        with open(config.TRADE_CONTROL_FILE, "r") as f:
            data = json.load(f)
            return data.get("active", True)
    except Exception:
        return True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.LIVE_LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# -- Model Loader ------------------------------------------------------------

def load_models():
    """Load Brain 1 (CNN), Scaler, and Brain 2 (Booster)."""
    import keras
    import joblib
    b1_long  = keras.models.load_model(str(config.BRAIN1_CNN_LONG_PATH))
    b1_short = keras.models.load_model(str(config.BRAIN1_CNN_SHORT_PATH))
    scaler   = joblib.load(str(config.BRAIN1_SCALER_PATH))
    
    # Use Booster for class-agnostic JSON loading
    b2 = xgb.Booster(); b2.load_model(str(config.BRAIN2_MODEL_PATH))
    logger.info(f"Models loaded (Brain1: CNN, Brain2: JSON, Scaler: Robust)")
    return b1_long, b1_short, b2, scaler


# These are set once at startup so execute_trade() can access them without
# passing them through every function call in the hot loop.
_simulator:       UpstoxSimulator | None = None
_order_guard:     SyncPendingOrderGuard  = SyncPendingOrderGuard(lock_timeout_seconds=config.ORDER_LOCK_TIMEOUT_SEC)


# -- Execute Trade ----------------------------------------------------------

def execute_trade(signal: dict) -> bool:
    """
    FIX 5 (COMPLETE): Pending-Order-Guarded paper order placement.

    Flow:
      1. try_acquire() - non-blocking mutex check. If symbol already has a
         pending order in this 0.1s loop window, drop the signal immediately.
      2. open_position() - delegate to PaperPortfolio. Returns False if already
         in the stock or max positions reached (double-entry prevention).
      3. release() - ALWAYS in finally to guarantee the mutex is freed even if
         open_position() raises an unexpected exception.

    Returns:
        True  if order was accepted and position opened.
        False if blocked by mutex or portfolio rules.
    """
    global _simulator, _order_guard

    if _simulator is None:
        logger.error("execute_trade called before _simulator initialised!")
        return False

    symbol    = signal["symbol"]
    sector    = signal["sector"]
    side      = signal["direction"]   # "BUY" or "SELL"
    price     = signal["price"]
    ts        = datetime.now()

    # Fix 4: Real-World Execution Latency (Slippage Mirage)
    # Apply entry slippage to simulate market order friction
    slippage   = config.T1_SLIPPAGE_PCT
    fill_price = price * (1.0 + slippage) if side == "BUY" else price * (1.0 - slippage)

    # -- Fix 5: Non-blocking mutex - drop signal if already pending --------
    if not _order_guard.try_acquire(symbol, side):
        return False

    # Calculate SL level: Entry +/- (STRUCTURAL_REVERSAL_BRICKS * brick_size)
    brick_size = signal["brick_size"]
    sl_dist    = config.STRUCTURAL_REVERSAL_BRICKS * brick_size
    sl_price   = fill_price - sl_dist if side == "BUY" else fill_price + sl_dist

    try:
        # Delegate to UpstoxSimulator
        req = _simulator.place_order(
            symbol   = symbol,
            side     = side,
            qty      = signal.get("qty", 1),
            price    = fill_price,
            sl_price = sl_price,
            ts       = ts
        )
        if req and req.state != "REJECTED":
            _simulator.fill_pending_order(symbol, ts)
            logger.info(f"[Engine->Sim] ORDER FILLED: {side} {symbol} @ Rs {fill_price:.2f} "
                        f"(Orig: {price:.2f}) | prob={signal.get('brain1_prob',0):.3f} "
                        f"| conv={signal.get('brain2_conviction',0):.1f}")
            return True
        return False

    except Exception as e:
        logger.error(f"[Engine->Sim] execute_trade EXCEPTION {symbol}: {e}")
        return False

    finally:
        # ALWAYS release - even on exception - to prevent permanent lockout
        _order_guard.release(symbol)


# -- Soft Veto --------------------------------------------------------------

def passes_soft_veto(signal: str, rel_strength: float) -> bool:
    if signal == "LONG" and rel_strength < -config.SOFT_VETO_THRESHOLD:
        return False
    if signal == "SHORT" and rel_strength > config.SOFT_VETO_THRESHOLD:
        return False
    return True


# -- Warmup ------------------------------------------------------------------

def warmup_brick_sizes(universe: pd.DataFrame) -> dict[str, float]:
    """Compute today's brick size from previous-day close for each symbol."""
    logger.info("WARMUP - Calculating brick sizes")
    sizes = {}
    for _, row in universe.iterrows():
        sym, sec = row["symbol"], row["sector"]
        stock_dir = config.DATA_DIR / sec / sym
        if stock_dir.exists():
            pqs = sorted(stock_dir.glob("*.parquet"))
            if pqs:
                try:
                    df = pd.read_parquet(pqs[-1])
                    if not df.empty:
                        sizes[sym] = df["brick_close"].iloc[-1] * config.NATR_BRICK_PERCENT
                        continue
                except Exception:
                    pass
        sizes[sym] = 500 * config.NATR_BRICK_PERCENT  # fallback
    logger.info(f"Brick sizes ready for {len(sizes)} symbols")
    return sizes


# -- State Writer ------------------------------------------------------------

def _serialize_active_trades(simulator: UpstoxSimulator) -> list:
    """Serialize UpstoxSimulator active trades for live_state.json."""
    if simulator is None:
        return []
    trades = []
    try:
        for sym, order in simulator.active_trades.items():
            trades.append({
                "symbol":         sym,
                "side":           order.side,
                "qty":            order.qty,
                "entry_price":    round(order.entry_price, 2),
                "sl_price":       round(getattr(order, 'sl_price', 0.0), 2),
                "last_price":     round(order.last_price, 2),
                "unrealized_pnl": round(order.unrealized_pnl, 2),
                "locked_margin":  round(order.locked_margin, 2),
                "entry_time":     order.filled_at.isoformat() if order.filled_at else None,
            })
    except Exception as e:
        logger.warning(f"Could not serialize active trades: {e}")
    return trades


def _serialize_margin(simulator: UpstoxSimulator) -> dict:
    """Serialize margin usage for live_state.json."""
    if simulator is None:
        return {"total_capital": 0.0, "available_margin": 0.0,
                "locked_margin": 0.0, "margin_usage_pct": 0.0}
    try:
        return simulator.get_margin_usage()
    except Exception:
        return {"total_capital": 0.0, "available_margin": 0.0,
                "locked_margin": 0.0, "margin_usage_pct": 0.0}


def write_live_state(top_signals, renko_states, risk: RiskFortress, latency_ms: float):
    chart_bricks = []
    if top_signals:
        sym = top_signals[0]["symbol"]
        if sym in renko_states:
            bdf = renko_states[sym].to_dataframe()
            if not bdf.empty:
                chart_bricks = bdf.tail(config.REGIME_WINDOW * 5).to_dict(orient="records")
                for b in chart_bricks:
                    for k in ["brick_timestamp", "brick_start_time", "brick_end_time"]:
                        if k in b and hasattr(b[k], "isoformat"):
                            b[k] = b[k].isoformat()

    # Include active trades + PnL so mobile app can read them from this file
    active_trades = _serialize_active_trades(_simulator)
    margin_usage  = _serialize_margin(_simulator)
    live_pnl = 0.0
    try:
        if _simulator is not None:
            live_pnl = _simulator.get_live_pnl()
    except Exception:
        pass

    state = {
        "timestamp": datetime.now().isoformat(),
        "top_signals": top_signals,
        "chart_symbol": top_signals[0]["symbol"] if top_signals else None,
        "chart_bricks": chart_bricks,
        "active_trades": active_trades,
        "live_pnl":      live_pnl,
        "margin_usage":  margin_usage,
        "market_regime": compute_market_regime(),
        "sentiment_feed": _get_sentiment_feed(),
        "health": {
            "loop_latency_ms": round(latency_ms, 1),
            "drift_accuracy": risk.drift_accuracy,
            "yellow_alert": risk.yellow_alert,
            "active_symbols": len(renko_states),
        },
    }
    try:
        with open(config.LIVE_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"State write failed: {e}")


# -- Sector Direction Helper ------------------------------------------------

def get_sector_directions(sector_renko: dict[str, LiveRenkoState]) -> dict[str, int]:
    return {
        st.sector: (st.bricks[-1]["direction"] if st.bricks else 0)
        for st in sector_renko.values()
    }


# -- Main Live Loop ----------------------------------------------------------

def run_live_engine():
    logger.info("=" * 70)
    logger.info("LIVE ENGINE - Starting")
    logger.info("=" * 70)

    # -- Sleep until 09:00 --------------------------------------------------
    now = datetime.now()
    target = now.replace(hour=9, minute=0, second=0, microsecond=0)
    if now < target:
        logger.info(f"Sleeping {(target-now).total_seconds():.0f}s until 09:00")
        time.sleep((target - now).total_seconds())

    # -- Load universe & models ---------------------------------------------
    universe = pd.read_csv(config.UNIVERSE_CSV)
    universe["is_index"] = universe["is_index"].astype(str).str.lower().isin(["true","1","yes"])
    stocks  = universe[~universe["is_index"]].reset_index(drop=True)
    indices = universe[ universe["is_index"]].reset_index(drop=True)

    brain1_long, brain1_short, brain2, scaler = load_models()
    sector_index_map = {r["sector"]: r["symbol"] for _, r in indices.iterrows()}

    # Initialise the module-level UpstoxSimulator singleton used by execute_trade()
    global _simulator
    _simulator = UpstoxSimulator(starting_capital=config.STARTING_CAPITAL)
    logger.info("UpstoxSimulator initialised.")

    # -- Warmup at specific config time (Default 09:05) ----------------------
    wt = datetime.now().replace(hour=config.WARMUP_HOUR, minute=config.WARMUP_MINUTE, second=0, microsecond=0)
    if datetime.now() < wt:
        sleep_sec = (wt - datetime.now()).total_seconds()
        logger.info(f"Sleeping {sleep_sec:.0f}s until {config.WARMUP_HOUR:02d}:{config.WARMUP_MINUTE:02d} AM Warmup...")
        time.sleep(sleep_sec)
    brick_sizes = warmup_brick_sizes(universe)

    renko_states = {}
    for _, r in stocks.iterrows():
        st = LiveRenkoState(r["symbol"], r["sector"], brick_sizes.get(r["symbol"], 1.0))
        st.load_history(config.RENKO_HISTORY_LIMIT)
        renko_states[r["symbol"]] = st

    sector_renko = {}
    for _, r in indices.iterrows():
        st = LiveRenkoState(r["symbol"], r["sector"], brick_sizes.get(r["symbol"], 1.0))
        st.load_history(config.RENKO_HISTORY_LIMIT)
        sector_renko[r["symbol"]] = st

    risk = RiskFortress()

    # FIX 1 + 3 + 4 + 5: Build the execution guard at warm-up time
    # Include BOTH stocks and indices in the guard universe for warmup and heartbeats
    all_sym_sector_map = {r["symbol"]: r["sector"] for _, r in universe.iterrows()}
    exec_guard = LiveExecutionGuard(
        symbols   = list(renko_states.keys()) + list(sector_renko.keys()),
        sectors   = all_sym_sector_map,
        silence_threshold  = config.HEARTBEAT_INJECT_SEC,
        order_lock_timeout = config.ORDER_LOCK_TIMEOUT_SEC,
    )
    # FIX 1: Load historical bricks to warm up all indicators before 09:15
    # This will now include NIFTY ENERGY and other indices in the logs.
    exec_guard.warm_up_all()

    # Synchronize guard buffers back to LiveRenkoState to ensure consistency
    for sym, st in renko_states.items():
        if sym in exec_guard.buffers:
            st.bricks = list(exec_guard.buffers[sym]._buffer)
    for sym, st in sector_renko.items():
        if sym in exec_guard.buffers:
            st.bricks = list(exec_guard.buffers[sym]._buffer)


    # -- Wait for Market Open ------------------------------------------------
    ot = datetime.now().replace(hour=config.MARKET_OPEN_HOUR, minute=config.MARKET_OPEN_MINUTE, second=0, microsecond=0)
    if datetime.now() < ot:
        sleep_sec = (ot - datetime.now()).total_seconds()
        logger.info(f"Brick sizes calculating complete. Sleeping {sleep_sec:.0f}s until {config.MARKET_OPEN_HOUR:02d}:{config.MARKET_OPEN_MINUTE:02d} AM Market Open...")
        time.sleep(sleep_sec)
    logger.info(f"{config.MARKET_OPEN_HOUR:02d}:{config.MARKET_OPEN_MINUTE:02d} - TRADING LOOP STARTED")

    tick_provider = TickProvider(list(renko_states) + list(sector_renko))
    tick_provider.connect()

    last_write = 0.0
    last_entry_minutes = {}  # Hyper-trading protection
    _already_squared_off = False
    
    try:
        while True:
            t0 = time.time()
            now = datetime.now()

            # Shutdown check
            if now.hour > config.SYSTEM_SHUTDOWN_HOUR or \
               (now.hour == config.SYSTEM_SHUTDOWN_HOUR and now.minute >= config.SYSTEM_SHUTDOWN_MINUTE):
                logger.info(f"{config.SYSTEM_SHUTDOWN_HOUR}:{config.SYSTEM_SHUTDOWN_MINUTE:02d} - MARKET CLOSED. Bye.")
                tick_provider.disconnect(); sys.exit(0)

            # -- Intraday Auto-Square-Off (Cutoff) -------------------------
            # FIX #10: Added parentheses around the `or` condition to fix an operator precedence bug that caused wasteful EOD re-evaluation.
            if not _already_squared_off and (now.hour > config.EOD_SQUARE_OFF_HOUR or \
               (now.hour == config.EOD_SQUARE_OFF_HOUR and now.minute >= config.EOD_SQUARE_OFF_MIN)):
                # FIX 1: Use Simulator for EOD Auto-Square-Off
                if _simulator is not None:
                    _simulator.square_off_all(now)
                _already_squared_off = True

            # -- Instant Kill (App Button) ------------------------------------
            # Re-read control state to check if user pressed instant kill
            from trading_engine.src.control_state import CONTROL_STATE
            if CONTROL_STATE.get("GLOBAL_KILL", False):
                logger.critical("GLOBAL_KILL ACTIVE. Forcing close of all positions immediately.")
                if _simulator is not None:
                    _simulator.square_off_all(now)

            ticks = tick_provider.get_latest_ticks()

            # -- Circuit Breaker (Stale Data Protection) ----------
            # Only applies to live configuration. If the WebSocket disconnects, the newest tick stops updating.
            # If the entire market feed is older than 5 seconds, freeze the engine.
            if getattr(tick_provider, "_use_live", False) and ticks:
                latest_tick_time = max((t["timestamp"] for t in ticks.values() if "timestamp" in t), default=now)
                max_tick_age = (now - latest_tick_time).total_seconds()
                
                if max_tick_age > config.CIRCUIT_BREAKER_STALE_SEC:
                    logger.warning(f"CIRCUIT BREAKER: Market data is {max_tick_age:.1f}s stale (limit {config.CIRCUIT_BREAKER_STALE_SEC}s). Engine paused.")
                    time.sleep(1)
                    continue

            # Process sector ticks
            for sym, st in sector_renko.items():
                # FIX 3: Inject heartbeat if WebSocket has gone silent
                exec_guard.heartbeat.check_and_inject(sym, st, now)

                if sym in ticks:
                    t = ticks[sym]
                    # FIX 3: Register this live tick so heartbeat knows it's alive
                    exec_guard.heartbeat.register_tick(sym, t["ltp"])

                    prev_cnt = len(st.bricks)
                    st.process_tick(t["ltp"], t["high"], t["low"], t["timestamp"], volume=t.get("volume", 0))

                    # FIX 4: Append new brick to rolling buffer
                    if len(st.bricks) > prev_cnt:
                        new_brick = st.bricks[-1]
                        exec_guard.buffers[sym].append(new_brick)
                        exec_guard.splicers[sym].append_live_brick(new_brick)

            sector_dirs = get_sector_directions(sector_renko)

            # Process stock ticks
            all_signals = []
            executable_signals = []  # Collecting signals for priority execution

            for sym, st in renko_states.items():
                # FIX 3: Inject heartbeat if WebSocket has gone silent for 60s
                exec_guard.heartbeat.check_and_inject(sym, st, now)

                if sym not in ticks:
                    continue
                t = ticks[sym]

                # Update live MTM PnL for active positions
                if _simulator is not None:
                    _simulator.update_active_price(sym, t["ltp"])

                # FIX 3: Register this live tick so heartbeat knows it's alive
                exec_guard.heartbeat.register_tick(sym, t["ltp"])

                prev_cnt = len(st.bricks)
                st.process_tick(t["ltp"], t["high"], t["low"], t["timestamp"], volume=t.get("volume", 0))
                if len(st.bricks) <= prev_cnt or len(st.bricks) < 2:
                    continue

                # Gate 0: Penny Stock Filter
                if t["ltp"] < MIN_PRICE_FILTER:
                    continue

                # FIX 4: Append new brick to rolling buffer (O(1), not DataFrame)
                if st.bricks:
                    new_brick = st.bricks[-1]
                    exec_guard.buffers[sym].append(new_brick)
                    # FIX 1 (PERMANENT): Mark this as a live brick in the splicer
                    # This is the ONLY correct way to count today's bricks.
                    # The old brick_timestamp.date() comparison was broken because
                    # parquet timestamps after tz-strip could match today's date.
                    exec_guard.splicers[sym].append_live_brick(new_brick)

                sec_sym = sector_index_map.get(st.sector, "")
                sec_bdf = exec_guard.buffers[sec_sym].to_dataframe() if sec_sym in exec_guard.buffers else pd.DataFrame()
                
                # Use faster rolling buffer for features instead of full history
                bdf = compute_features_live(exec_guard.buffers[sym].to_dataframe(), sec_bdf)
                latest = bdf.iloc[-1]

                # Microsecond O(1) execution swap: Bypass pandas overhead
                latest_dict = latest.to_dict()
                
                # CNN Inference (3D sliding window + Scaling)
                win_df = exec_guard.buffers[sym].to_dataframe().tail(config.CNN_WINDOW_SIZE)
                if len(win_df) < config.CNN_WINDOW_SIZE:
                    continue # Wait for more bricks
                
                # Apply scaler to 2D features before 3D stacking
                feats_2d = bdf[EXPECTED_FEATURES].tail(config.CNN_WINDOW_SIZE).fillna(0)
                scaled_2d = scaler.transform(feats_2d)
                feat_3d   = np.array([scaled_2d], dtype=np.float32)
                
                # Dual Brain Inference: LONG and SHORT
                p_long  = float(brain1_long.predict(feat_3d, verbose=0)[0])
                p_short = float(brain1_short.predict(feat_3d, verbose=0)[0])
                
                # Signal Selection Logic (Highest probability)
                signal_str = "FLAT"
                if p_long >= p_short:
                    b1p, b1d = p_long, 1
                else:
                    b1p, b1d = p_short, -1
                
                # Static Threshold Selection (Now only controls signal_str, not b1p)
                thresh_long  = config.LONG_ENTRY_PROB_THRESH if USE_CALIBRATED_MODELS else config.RAW_LONG_ENTRY_PROB_THRESH
                thresh_short = config.SHORT_ENTRY_PROB_THRESH if USE_CALIBRATED_MODELS else config.RAW_SHORT_ENTRY_PROB_THRESH

                long_ok  = p_long  >= thresh_long
                short_ok = p_short >= thresh_short

                if long_ok and short_ok:
                    if p_long >= p_short:
                        signal_str = "LONG"
                    else:
                        signal_str = "SHORT"
                elif long_ok:
                    signal_str = "LONG"
                elif short_ok:
                    signal_str = "SHORT"
                
                # Reset direction if neither passed (for safety) 
                if signal_str == "FLAT":
                    # We still keep b1p/b1d for the log, 
                    # but logic below uses signal_str to gate entries.
                    pass
                
                if signal_str != "FLAT":
                    p_type = "Calibrated" if USE_CALIBRATED_MODELS else "Raw"
                    # Log high probability signals even if they get filtered later
                    if b1p > 0.5:
                        logger.info(f"[{sym}] {signal_str} {p_type} Prob: {b1p:.4f} | RS: {float(latest.get('relative_strength', 0)):.2f} | Wick: {float(latest.get('wick_pressure', 0)):.2f}")


                # Build the feature matrix for Brain 2 dynamically from config.BRAIN2_FEATURES
                b2_vals = []
                for f_name in config.BRAIN2_FEATURES:
                    if f_name == "brain1_prob_long": b2_vals.append(p_long)
                    elif f_name == "brain1_prob_short": b2_vals.append(p_short)
                    elif f_name == "trade_direction": b2_vals.append(float(b1d))
                    else: b2_vals.append(float(latest.get(f_name, 0)))
                
                # Booster requires DMatrix
                dm_meta = xgb.DMatrix([b2_vals], feature_names=config.BRAIN2_FEATURES)
                b2c = float(np.clip(brain2.predict(dm_meta)[0], 0, config.TARGET_CLIPPING_BPS))

                sec_dir = sector_dirs.get(st.sector, 0)
                score = risk.score_signal(b1p, b2c, b1d, sec_dir)
                rel_str_val = float(latest.get("relative_strength", 0))

                # Fix 3: Position Sizing (Capital Allocation)
                alloc     = config.STARTING_CAPITAL * config.POSITION_SIZE_PCT * config.INTRADAY_LEVERAGE
                calc_qty  = max(1, int(alloc / float(t["ltp"])))

                sig = {
                    "symbol": sym, "sector": st.sector,
                    "direction": "BUY" if signal_str == "LONG" else ("SELL" if signal_str == "SHORT" else "FLAT"),
                    "qty": calc_qty,
                    "brain1_prob": round(b1p, 4),
                    "brain2_conviction": round(b2c, 2),
                    "score": round(score, 2),
                    "velocity": round(float(latest.get("velocity",0)), 4),
                    "wick_pressure": round(float(latest.get("wick_pressure",0)), 4),
                    "rs": round(rel_str_val, 4),
                    "price": round(float(t["ltp"]), 2),
                    "brick_size": st.brick_size,
                    "brick_count": len(st.bricks),
                    "is_vetoed": not passes_soft_veto(signal_str, rel_str_val),
                    "timestamp": now.isoformat(),
                }
                all_signals.append(sig)

                # -- EXIT GATES --
                if _simulator is not None and sym in _simulator.active_trades:
                    order = _simulator.active_trades[sym]
                    ltp_val = float(t["ltp"])
                    
                    exit_reason = check_exit_conditions(
                        order_side = order.side,
                        entry_price = order.entry_price,
                        current_price = ltp_val,
                        brick_size = st.brick_size,
                        b2c = b2c,
                        p_long = p_long,
                        p_short = p_short
                    )
                        
                    if exit_reason:
                        _simulator.close_position(sym, ltp_val, now, exit_reason)
                        logger.info(f"[Engine->Sim] EXIT {sym} @ {ltp_val:.2f} | reason={exit_reason} (Backtest Synced)")
                        
                        # Log the EXIT event
                        log_brick_event(
                            ts=now, symbol=sym, sector=st.sector, price=ltp_val,
                            brick_dir=int(latest.get("direction", 0)), sec_dir=sec_dir,
                            new_bricks=len(st.bricks),
                            velocity=float(latest.get("velocity", 0)),
                            wick_pressure=float(latest.get("wick_pressure", 0)),
                            relative_strength=rel_str_val,
                            brick_size=st.brick_size,
                            duration_seconds=float(latest.get("duration", 0)),
                            consecutive_same=int(latest.get("consecutive_same_dir", 0)),
                            brain1_prob=b1p, brain2_conv=b2c,
                            signal=signal_str, score=score,
                            action="EXIT", reason=exit_reason,
                            open_positions=len(_simulator.active_trades) if _simulator else 0,
                            live_pnl=_simulator.get_live_pnl() if _simulator else 0.0
                        )
                    continue # Skip entry logic if already in position

                # -- ENTRY GATES --
                is_already_in_position = (_simulator is not None and sym in _simulator.active_trades)
                stock_losses = sum(1 for trade in _simulator.trade_history if trade.symbol == sym and trade.net_pnl < 0) if _simulator else 0
                portfolio_size = len(_simulator.active_trades) if _simulator else 0
                
                recent_bricks = st.bricks[-config.MIN_CONSECUTIVE_BRICKS:] if len(st.bricks) >= config.MIN_CONSECUTIVE_BRICKS else []
                recent_dirs = [rb["direction"] for rb in recent_bricks]
                
                brick_dir = int(latest.get("direction", 0))

                gate_pass, gate_reason, gate_audit = check_entry_gates(
                    symbol = sym,
                    now = now,
                    price = float(t["ltp"]),
                    b1p = b1p,
                    b2c = b2c,
                    signal_str = signal_str,
                    rel_str = rel_str_val,
                    wick_p = float(latest.get("wick_pressure", 0)),
                    z_vwap = float(latest.get("vwap_zscore", 0)),
                    streak_count = int(latest.get("consecutive_same_dir", 0)),
                    brick_dir = brick_dir,
                    recent_dirs = recent_dirs,
                    stock_losses = stock_losses,
                    portfolio_size = portfolio_size,
                    is_already_in_position = is_already_in_position,
                    structural_score = float(latest.get("structural_score", 0.0))
                )

                # -- Log every brick event "Behind the Scenes" --
                log_brick_event(
                    ts=now, symbol=sym, sector=st.sector, price=float(t["ltp"]),
                    brick_dir=brick_dir, sec_dir=sec_dir,
                    new_bricks=len(st.bricks),
                    velocity=float(latest.get("velocity", 0)),
                    wick_pressure=float(latest.get("wick_pressure", 0)),
                    relative_strength=rel_str_val,
                    brick_size=st.brick_size,
                    duration_seconds=float(latest.get("duration", 0)),
                    consecutive_same=int(latest.get("consecutive_same_dir", 0)),
                    brain1_prob=b1p, brain2_conv=b2c,
                    signal=signal_str, score=score,
                    structural_score=float(latest.get("structural_score", 0.0)),
                    # Spread the audit gates
                    **gate_audit,
                    action="ENTRY" if gate_pass else "SKIP",
                    reason=gate_reason if not gate_pass else "ALL_PASS",
                    open_positions=len(_simulator.active_trades) if _simulator else 0,
                    live_pnl=_simulator.get_live_pnl() if _simulator else 0.0
                )

                if gate_pass:
                    # Same-minute duplicate check
                    current_minute = now.replace(second=0, microsecond=0)
                    if last_entry_minutes.get(sym) != current_minute:
                        executable_signals.append(sig)
                        last_entry_minutes[sym] = current_minute
                else:
                    if b1p > 0.5:
                        logger.info(f"[{sym}] [DROP] Gate Failed: {gate_reason} (Prob: {b1p:.4f})")

            # -- PRIORITY EXECUTION QUEUE ---------------------------------
            # Sort all collected signals by their mathematical score (Highest first)
            # This ensures limited margin goes to the best setups, not just the fastest ticks.
            if executable_signals:
                executable_signals.sort(key=lambda x: x["score"], reverse=True)
                logger.info(f"Priority Queue: {len(executable_signals)} signals firing. Executing highest score first.")
                
                for sig in executable_signals:
                    if is_trading_active():
                        execute_trade(sig)
                    else:
                        logger.info(f"TRADE SUPPRESSED: Engine Paused by User | {sig['symbol']}")

            top = risk.rank_signals(all_signals)
            latency = (time.time() - t0) * 1000

            if (time.time() - last_write) >= config.STATE_WRITE_INTERVAL:
                write_live_state(top, renko_states, risk, latency)
                last_write = time.time()

            elapsed = time.time() - t0
            if elapsed < config.HEARTBEAT_INJECT_SEC / 100: # Adaptive sleep
                time.sleep(max(0.01, config.HEARTBEAT_INJECT_SEC / 100 - elapsed))

    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        tick_provider.disconnect()
        logger.info("Engine shut down.")


if __name__ == "__main__":
    run_live_engine()

