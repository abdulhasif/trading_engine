"""
src/live/engine_main.py - Phase 5: Real-Time Trading Engine (Master Router)
============================================================================
Daily lifecycle: 08:50 wake -> 09:00 connect -> 09:05 warmup -> 09:15 trade -> 15:35 shutdown.

Order of Operations per tick (The Master Router):
  Step 0: PULSE       — Register tick, update MTM, process brick physics
  Step 1: EXIT FIRST  — If symbol has active position -> exit check -> continue
  Step 2: SNIPER      — O(1) volume pre-check. If vol > 500 -> skip to execution
  Step 3: MEMORY      — Record b2c into DynamicThresholdTracker
  Step 4: CHAMELEON   — Get percentile threshold
  Step 5: RIVER       — Calculate structural trend + pullback
  Step 6: STANDARD    — Run full 12-gate check_entry_gates()
  Step 7: EXECUTE     — EntryStateLock.try_enter() + BrickCooldown + fill
"""

import sys
if sys.version_info < (3, 12) or sys.version_info >= (3, 13):
    print("ERROR: This application requires Python 3.12 specifically.")
    print(f"Current version: {sys.version}")
    print("Please run using: py -3.12 src/engine_main.py")
    sys.exit(1)

import os
os.environ["KERAS_BACKEND"] = "torch"
import time
import logging
import pandas as pd
from datetime import datetime

from trading_core.core.config import base_config as config
from trading_core.core.risk.risk_fortress import RiskFortress
from trading_engine.src.data.tick_provider import TickProvider
from trading_core.core.risk.execution_guard import LiveExecutionGuard, SyncPendingOrderGuard
from trading_engine.src.execution.upstox_simulator import UpstoxSimulator
from trading_engine.src.core.daily_logger import log_brick_event
from trading_core.core.features import compute_features_live
from trading_core.core.risk.execution_guard import calculate_river_state

# Modular Imports
from trading_engine.src.models.inference_engine import InferenceEngine
from trading_engine.src.strategy.strategy_manager import StrategyManager
from trading_engine.src.execution.execution_manager import ExecutionManager
from trading_engine.src.core.physics_manager import PhysicsManager
from trading_engine.src.utils.state_manager import write_live_state, is_trading_active

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.LIVE_LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

def run_live_engine():
    logger.info("=" * 70)
    logger.info("LIVE ENGINE - Starting (Modular Orchestrator)")
    logger.info("=" * 70)

    # -- Phase 1: SYSTEM_WAKE (Default 08:50) --------------------------------
    now = datetime.now()
    wake_target = now.replace(hour=config.SYSTEM_WAKE_HOUR, minute=config.SYSTEM_WAKE_MINUTE, second=0, microsecond=0)
    if now < wake_target:
        logger.info(f"SYSTEM SLEEP: Waiting {(wake_target-now).total_seconds():.0f}s until {config.SYSTEM_WAKE_HOUR:02d}:{config.SYSTEM_WAKE_MINUTE:02d} Wake...")
        time.sleep((wake_target - now).total_seconds())

    # -- Load universe & models ---------------------------------------------
    universe = pd.read_csv(config.UNIVERSE_CSV)
    universe["is_index"] = universe["is_index"].astype(str).str.lower().isin(["true","1","yes"])
    stocks  = universe[~universe["is_index"]].reset_index(drop=True)
    indices = universe[ universe["is_index"]].reset_index(drop=True)

    # Initialize Components
    inference = InferenceEngine()
    inference.load_models()
    
    risk_fortress = RiskFortress()
    strategy = StrategyManager(risk_fortress)
    
    simulator = UpstoxSimulator(starting_capital=config.STARTING_CAPITAL)
    order_guard = SyncPendingOrderGuard(lock_timeout_seconds=config.ORDER_LOCK_TIMEOUT_SEC)
    execution = ExecutionManager(simulator, order_guard)
    
    all_sym_sector_map = {r["symbol"]: r["sector"] for _, r in universe.iterrows()}
    exec_guard = LiveExecutionGuard(
        symbols   = list(stocks["symbol"]) + list(indices["symbol"]),
        sectors   = all_sym_sector_map,
        silence_threshold  = config.HEARTBEAT_INJECT_SEC,
        order_lock_timeout = config.ORDER_LOCK_TIMEOUT_SEC,
    )
    
    physics = PhysicsManager(universe, exec_guard)
    exec_guard.warm_up_all() # Load historical bricks into buffers BEFORE initializing states

    # -- Phase 2: CONNECTIVITY_CHECK (Default 09:00) -------------------------
    ct = datetime.now().replace(hour=config.CONNECTIVITY_CHECK_HOUR, minute=config.CONNECTIVITY_CHECK_MINUTE, second=0, microsecond=0)
    if datetime.now() < ct:
        sleep_sec = (ct - datetime.now()).total_seconds()
        logger.info(f"INITIALIZED. Sleeping {sleep_sec:.0f}s until {config.CONNECTIVITY_CHECK_HOUR:02d}:{config.CONNECTIVITY_CHECK_MINUTE:02d} AM Connectivity Check...")
        time.sleep(sleep_sec)
    
    logger.info("CONNECTIVITY CHECK: Priming Renko physics & starting WebSocket...")
    physics.warmup_brick_sizes()
    physics.initialize_states(stocks, indices)
    
    tick_provider = TickProvider(list(physics.renko_states.keys()) + list(physics.sector_renko.keys()))
    tick_provider.connect()

    # -- Phase 3: WARMUP (Default 09:05) --------------------------------------
    wt = datetime.now().replace(hour=config.WARMUP_HOUR, minute=config.WARMUP_MINUTE, second=0, microsecond=0)
    if datetime.now() < wt:
        sleep_sec = (wt - datetime.now()).total_seconds()
        logger.info(f"CONNECTING. Sleeping {sleep_sec:.0f}s until {config.WARMUP_HOUR:02d}:{config.WARMUP_MINUTE:02d} AM Warmup...")
        time.sleep(sleep_sec)
    
    logger.info("WARMUP: Priming risk buffers...")
    exec_guard.warm_up_all()

    # -- Phase 4: MARKET_OPEN (Default 09:15) ---------------------------------
    ot = datetime.now().replace(hour=config.MARKET_OPEN_HOUR, minute=config.MARKET_OPEN_MINUTE, second=0, microsecond=0)
    if datetime.now() < ot:
        sleep_sec = (ot - datetime.now()).total_seconds()
        logger.info(f"WARMUP COMPLETE. Sleeping {sleep_sec:.0f}s until {config.MARKET_OPEN_HOUR:02d}:{config.MARKET_OPEN_MINUTE:02d} AM Market Open...")
        time.sleep(sleep_sec)
    logger.info(f"MARKET OPEN: Starting TRADING LOOP at {config.MARKET_OPEN_HOUR:02d}:{config.MARKET_OPEN_MINUTE:02d}")

    # TickProvider moved to Connectivity Check phase

    last_write = 0.0
    last_heartbeat = 0.0
    last_lock_log = 0.0
    _already_squared_off = False
    sector_index_map = {r["sector"]: r["symbol"] for _, r in indices.iterrows()}
    
    # Phase 8: Ignition Velocity Memory
    prev_b1p_memo = {} # symbol -> last known B1 probability

    try:
        while True:
            t0 = time.time()
            now = datetime.now()

            # Shutdown check
            if now.hour > config.SYSTEM_SHUTDOWN_HOUR or \
               (now.hour == config.SYSTEM_SHUTDOWN_HOUR and now.minute >= config.SYSTEM_SHUTDOWN_MINUTE):
                logger.info(f"{config.SYSTEM_SHUTDOWN_HOUR}:{config.SYSTEM_SHUTDOWN_MINUTE:02d} - MARKET CLOSED. Bye.")
                tick_provider.disconnect(); sys.exit(0)

            # EOD Square-off
            if not _already_squared_off and (now.hour > config.EOD_SQUARE_OFF_HOUR or \
               (now.hour == config.EOD_SQUARE_OFF_HOUR and now.minute >= config.EOD_SQUARE_OFF_MIN)):
                execution.square_off_all(now)
                _already_squared_off = True

            # Global Kill
            from trading_engine.src.control_state import CONTROL_STATE
            if CONTROL_STATE.get("GLOBAL_KILL", False):
                logger.critical("GLOBAL_KILL ACTIVE. Forcing close of all positions immediately.")
                execution.square_off_all(now)

            ticks = tick_provider.get_latest_ticks()

            # Circuit Breaker
            if getattr(tick_provider, "_use_live", False) and ticks:
                latest_tick_time = max((t["timestamp"] for t in ticks.values() if "timestamp" in t), default=now)
                max_tick_age = (now - latest_tick_time).total_seconds()
                if max_tick_age > config.CIRCUIT_BREAKER_STALE_SEC:
                    logger.warning(f"CIRCUIT BREAKER: Market data is {max_tick_age:.1f}s stale. Engine paused.")
                    time.sleep(1); continue

            # Sector Processing
            for sym in physics.sector_renko:
                if sym in ticks:
                    physics.process_sector_tick(sym, ticks[sym], now)

            sector_dirs = physics.get_sector_directions()

            # ============================================================
            # THE MASTER ROUTER — Stock Processing (Strict Order of Ops)
            # ============================================================
            all_signals = []
            executable_signals = []

            for sym, st in physics.renko_states.items():
                if sym not in ticks: continue
                t = ticks[sym]

                # ── Step 0: PULSE ─────────────────────────────────────────
                # Register tick, update MTM, process Renko physics.
                execution.update_active_price(sym, t["ltp"])
                brick_formed = physics.process_stock_tick(sym, t, now)
                if not brick_formed: continue

                # ALWAYS log every brick formed, even during WARMUP
                if len(st.bricks) < config.CNN_WINDOW_SIZE:
                    last_brick = st.bricks[-1]
                    log_brick_event(
                        ts=now, symbol=sym, sector=st.sector, price=float(t["ltp"]),
                        brick_dir=last_brick["direction"], sec_dir=sector_dirs.get(st.sector, 0),
                        new_bricks=1, action="SKIP", reason="WARMUP"
                    )
                    continue

                # ── Step 1: EXIT FIRST ────────────────────────────────────
                # If symbol has an active position, evaluate exit BEFORE
                # wasting CPU on feature computation and AI inference.
                if sym in execution.simulator.active_trades:
                    order = execution.simulator.active_trades[sym]
                    t_type = exec_guard.entry_lock.get_trade_type(sym)

                    # Lightweight feature extraction for exit (volume + price only)
                    sec_sym = sector_index_map.get(st.sector, "")
                    sec_bdf = exec_guard.buffers[sec_sym].to_dataframe() if sec_sym in exec_guard.buffers else pd.DataFrame()
                    bdf = compute_features_live(exec_guard.buffers[sym].to_dataframe(), sec_bdf)
                    latest_row = bdf.iloc[-1].to_dict()
                    v_int = float(latest_row.get("volume_intensity_per_sec", 0))

                    # Inference for exit evaluation
                    p_long, p_short = inference.predict_brain1(bdf)
                    if p_long >= p_short: b1d = 1
                    else: b1d = -1
                    b2c = inference.predict_brain2(p_long, p_short, b1d, latest_row)

                    # Step 3 (Memory): Record b2c even during exit evaluation
                    exec_guard.dyn_thresh.record(sym, b2c)

                    exit_reason = strategy.check_exit(
                        order, float(t["ltp"]), st, b2c, p_long, p_short,
                        trade_type=t_type, vol_intensity=v_int
                    )

                    if exit_reason:
                        execution.close_position(sym, float(t["ltp"]), now, exit_reason)
                        # Wire safety locks: release position + start cooldown
                        exec_guard.entry_lock.confirm_exit(sym)
                        exec_guard.cooldown.record_exit(sym, exec_guard.buffers[sym]._total_bricks_seen)
                        logger.info(f"[Router] EXIT {sym} @ {float(t['ltp']):.2f} | type={t_type} | reason={exit_reason}")
                        last_brick = st.bricks[-1]
                        log_brick_event(
                            ts=now, symbol=sym, sector=st.sector, price=float(t["ltp"]),
                            brick_dir=last_brick["direction"], sec_dir=sector_dirs.get(st.sector, 0), 
                            new_bricks=1, action="EXIT", reason=exit_reason, 
                            trade_type=t_type, **latest_row
                        )
                    continue  # Skip entry evaluation for active positions

                # ── Step 2-6: ENTRY EVALUATION (no active position) ───────

                # Feature Computation (required for all entry paths)
                sec_sym = sector_index_map.get(st.sector, "")
                sec_bdf = exec_guard.buffers[sec_sym].to_dataframe() if sec_sym in exec_guard.buffers else pd.DataFrame()
                bdf = compute_features_live(exec_guard.buffers[sym].to_dataframe(), sec_bdf)
                latest_row = bdf.iloc[-1].to_dict()

                # AI Inference
                p_long, p_short = inference.predict_brain1(bdf)
                signal_str = "FLAT"
                if p_long >= p_short: b1p, b1d = p_long, 1
                else: b1p, b1d = p_short, -1

                thresh_l = config.LONG_ENTRY_PROB_THRESH if config.USE_CALIBRATED_MODELS else config.RAW_LONG_ENTRY_PROB_THRESH
                thresh_s = config.SHORT_ENTRY_PROB_THRESH if config.USE_CALIBRATED_MODELS else config.RAW_SHORT_ENTRY_PROB_THRESH
                if p_long >= thresh_l and p_long >= p_short: signal_str = "LONG"
                elif p_short >= thresh_s: signal_str = "SHORT"

                b2c = inference.predict_brain2(p_long, p_short, b1d, latest_row)

                # ── Step 2.5: VELOCITY CALCULATION (Phase 8) ─────────────
                # Calculate the speed at which Brain 1 is converging.
                last_b1p = prev_b1p_memo.get(sym, b1p)
                delta_b1p = b1p - last_b1p
                prev_b1p_memo[sym] = b1p

                # ── Step 3: MEMORY UPDATE ─────────────────────────────────
                exec_guard.dyn_thresh.record(sym, b2c)

                # ── Step 4: CHAMELEON (Dynamic Threshold) ─────────────────
                dyn_conv = exec_guard.dyn_thresh.get_dynamic_threshold(sym, signal_str) if signal_str != "FLAT" else None

                # ── Step 5: THE RIVER (Structural Trend) ──────────────────
                river_wr, river_pb = calculate_river_state(exec_guard.buffers[sym]._buffer, signal_str) if signal_str != "FLAT" else (0.0, False)

                # ── Step 6: LIFECYCLE GATES ──────────────────────────────
                # Skip if within the morning "Wait for Range" lock period
                minutes_since_open = (now.hour - config.MARKET_OPEN_HOUR) * 60 + (now.minute - config.MARKET_OPEN_MINUTE)
                if minutes_since_open < config.ENTRY_LOCK_MINUTES:
                    last_brick = st.bricks[-1]
                    log_brick_event(
                        ts=now, symbol=sym, sector=st.sector, price=float(t["ltp"]),
                        brick_dir=last_brick["direction"], sec_dir=sector_dirs.get(st.sector, 0),
                        new_bricks=1, action="SKIP", reason="MORNING_LOCK", **latest_row
                    )
                    if now_ts - last_lock_log > 300.0: # Log to console every 5 mins
                         logger.info(f"MORNING LOCK: Skipping {sym} (Market stabilizing until 09:30)")
                         last_lock_log = now_ts
                    continue

                # Stop new entries after the "No New Entry" threshold
                if now.hour > config.NO_NEW_ENTRY_HOUR or \
                   (now.hour == config.NO_NEW_ENTRY_HOUR and now.minute >= config.NO_NEW_ENTRY_MIN):
                    last_brick = st.bricks[-1]
                    log_brick_event(
                        ts=now, symbol=sym, sector=st.sector, price=float(t["ltp"]),
                        brick_dir=last_brick["direction"], sec_dir=sector_dirs.get(st.sector, 0),
                        new_bricks=1, action="SKIP", reason="TIME_CUTOFF", **latest_row
                    )
                    continue

                # ── Step 7: STANDARD GATES (12-Gate Check) ────────────────
                portfolio_size = len(execution.simulator.active_trades)
                stock_losses = sum(1 for tr in execution.simulator.trade_history if tr.symbol == sym and tr.net_pnl < 0)

                gate_pass, gate_reason, gate_audit, sig = strategy.evaluate_entry(
                    sym, st.sector, signal_str, b1p, b2c, latest_row, st,
                    sector_dirs.get(st.sector, 0), portfolio_size, stock_losses, now,
                    dynamic_conv_thresh=dyn_conv,
                    river_win_ratio=river_wr,
                    river_pullback_cleared=river_pb,
                    volume_intensity=float(latest_row.get("volume_intensity_per_sec", 0)),
                    cvd_divergence=float(latest_row.get("feature_cvd_divergence", 0)),
                    delta_b1p=delta_b1p
                )

                # Position Sizing
                alloc = config.STARTING_CAPITAL * config.POSITION_SIZE_PCT * config.INTRADAY_LEVERAGE
                sig["qty"] = max(1, int(alloc / float(t["ltp"])))
                all_signals.append(sig)

                # Log entry attempt
                last_brick = st.bricks[-1]
                log_brick_event(
                    ts=now, symbol=sym, sector=st.sector, price=float(t["ltp"]),
                    brick_dir=last_brick["direction"], sec_dir=sector_dirs.get(st.sector, 0),
                    new_bricks=1, action="ENTRY" if gate_pass else "SKIP", 
                    reason=gate_reason if not gate_pass else "ALL_PASS", 
                    signal=signal_str,
                    trade_type=sig.get("trade_type", "NORMAL"), 
                    brain1_prob=b1p, brain2_conv=b2c, **latest_row
                )

                if gate_pass and not strategy.check_duplicate_minute(sym, now):
                    executable_signals.append(sig)

            # ── Step 7: EXECUTION (with safety locks) ─────────────────────
            if executable_signals:
                executable_signals.sort(key=lambda x: x["score"], reverse=True)
                for sig in executable_signals:
                    sym = sig["symbol"]
                    trade_type = sig.get("trade_type", "NORMAL")

                    # Safety Lock 1: Brick Cooldown (anti-churn)
                    if not exec_guard.cooldown.is_cooled_down(sym, exec_guard.buffers[sym]._total_bricks_seen):
                        continue

                    # Safety Lock 2: EntryStateLock (position deduplication)
                    if not exec_guard.entry_lock.try_enter(sym, trade_type):
                        continue

                    # Execute trade
                    if is_trading_active():
                        execution.execute_trade(sig)

            # State Update
            top = risk_fortress.rank_signals(all_signals)
            latency = (time.time() - t0) * 1000
            if (time.time() - last_write) >= config.STATE_WRITE_INTERVAL:
                write_live_state(top, physics.renko_states, risk_fortress, latency, execution)
                last_write = time.time()

            # --- HEARTBEAT (Phase 5 Visibility) ---
            if (time.time() - last_heartbeat) > 60.0:
                mins_open = (now.hour - config.MARKET_OPEN_HOUR) * 60 + (now.minute - config.MARKET_OPEN_MINUTE)
                status = "LOCKED" if mins_open < config.ENTRY_LOCK_MINUTES else "ACTIVE"
                logger.info(f"[Heartbeat] {now.strftime('%H:%M:%S')} | Ticks: {len(ticks)} | Latency: {latency:.2f}ms | Mode: {status}")
                last_heartbeat = time.time()

            time.sleep(max(0.001, 0.01 - (time.time() - t0)))

    except KeyboardInterrupt: logger.info("Stopped by user")
    finally: tick_provider.disconnect(); logger.info("Engine shut down.")

if __name__ == "__main__":
    run_live_engine()
