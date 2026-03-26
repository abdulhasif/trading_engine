"""
src/live/engine_main.py - Phase 4: Real-Time Trading Engine (Modular Orchestrator)
==================================================================================
Daily lifecycle: 08:50 wake -> 09:08 warmup -> 09:15 15:30 trade -> 15:35 shutdown.
Main orchestrator that coordinates modular components.

STRICT LOGIC PRESERVATION from original monolithic engine_main.py.
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

from trading_engine import config
from trading_core.core.risk.risk_fortress import RiskFortress
from trading_engine.src.data.tick_provider import TickProvider
from trading_core.core.risk.execution_guard import LiveExecutionGuard, SyncPendingOrderGuard
from trading_engine.src.execution.upstox_simulator import UpstoxSimulator
from trading_engine.src.core.daily_logger import log_brick_event

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

    # -- Warmup -------------------------------------------------------------
    wt = datetime.now().replace(hour=config.WARMUP_HOUR, minute=config.WARMUP_MINUTE, second=0, microsecond=0)
    if datetime.now() < wt:
        sleep_sec = (wt - datetime.now()).total_seconds()
        logger.info(f"Sleeping {sleep_sec:.0f}s until {config.WARMUP_HOUR:02d}:{config.WARMUP_MINUTE:02d} AM Warmup...")
        time.sleep(sleep_sec)
    
    physics.warmup_brick_sizes()
    physics.initialize_states(stocks, indices)
    exec_guard.warm_up_all()

    # -- Wait for Market Open ------------------------------------------------
    ot = datetime.now().replace(hour=config.MARKET_OPEN_HOUR, minute=config.MARKET_OPEN_MINUTE, second=0, microsecond=0)
    if datetime.now() < ot:
        sleep_sec = (ot - datetime.now()).total_seconds()
        logger.info(f"Warmup complete. Sleeping {sleep_sec:.0f}s until {config.MARKET_OPEN_HOUR:02d}:{config.MARKET_OPEN_MINUTE:02d} AM Market Open...")
        time.sleep(sleep_sec)
    logger.info(f"{config.MARKET_OPEN_HOUR:02d}:{config.MARKET_OPEN_MINUTE:02d} - TRADING LOOP STARTED")

    tick_provider = TickProvider(list(physics.renko_states.keys()) + list(physics.sector_renko.keys()))
    tick_provider.connect()

    last_write = 0.0
    _already_squared_off = False
    sector_index_map = {r["sector"]: r["symbol"] for _, r in indices.iterrows()}
    
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

            # Stock Processing
            all_signals = []
            executable_signals = []

            for sym, st in physics.renko_states.items():
                if sym not in ticks: continue
                t = ticks[sym]
                
                # MTM Updates
                execution.update_active_price(sym, t["ltp"])

                # Physics Processing
                brick_formed = physics.process_stock_tick(sym, t, now)
                if not brick_formed: continue
                
                # Minimum data check
                if len(st.bricks) < config.CNN_WINDOW_SIZE: continue

                # Feature Computation
                sec_sym = sector_index_map.get(st.sector, "")
                sec_bdf = exec_guard.buffers[sec_sym].to_dataframe() if sec_sym in exec_guard.buffers else pd.DataFrame()
                bdf = compute_features_live(exec_guard.buffers[sym].to_dataframe(), sec_bdf)
                latest_row = bdf.iloc[-1].to_dict()

                # Inference
                p_long, p_short = inference.predict_brain1(bdf)
                signal_str = "FLAT"
                if p_long >= p_short: b1p, b1d = p_long, 1
                else: b1p, b1d = p_short, -1

                thresh_l = config.LONG_ENTRY_PROB_THRESH if config.USE_CALIBRATED_MODELS else config.RAW_LONG_ENTRY_PROB_THRESH
                thresh_s = config.SHORT_ENTRY_PROB_THRESH if config.USE_CALIBRATED_MODELS else config.RAW_SHORT_ENTRY_PROB_THRESH

                if p_long >= thresh_l and p_long >= p_short: signal_str = "LONG"
                elif p_short >= thresh_s: signal_str = "SHORT"

                b2c = inference.predict_brain2(p_long, p_short, b1d, latest_row)

                # Phase 5: Record b2c into per-symbol rolling memory (EVERY prediction, not just entries)
                exec_guard.dyn_thresh.record(sym, b2c)

                # Phase 5: Get the auto-calibrated conviction threshold for this symbol + direction
                dyn_conv = exec_guard.dyn_thresh.get_dynamic_threshold(sym, signal_str) if signal_str != "FLAT" else None

                # Phase 3: The River — Structural Trend Calculator
                from trading_core.core.risk.execution_guard import calculate_river_state
                river_wr, river_pb = calculate_river_state(exec_guard.buffers[sym]._buffer, signal_str) if signal_str != "FLAT" else (0.0, False)

                # Strategy Gating
                portfolio_size = len(execution.simulator.active_trades)
                stock_losses = sum(1 for tr in execution.simulator.trade_history if tr.symbol == sym and tr.net_pnl < 0)
                
                gate_pass, gate_reason, gate_audit, sig = strategy.evaluate_entry(
                    sym, st.sector, signal_str, b1p, b2c, latest_row, st, 
                    sector_dirs.get(st.sector, 0), portfolio_size, stock_losses, now,
                    dynamic_conv_thresh=dyn_conv,
                    river_win_ratio=river_wr,               # Phase 3
                    river_pullback_cleared=river_pb          # Phase 3
                )
                
                # Position Sizing
                alloc = config.STARTING_CAPITAL * config.POSITION_SIZE_PCT * config.INTRADAY_LEVERAGE
                sig["qty"] = max(1, int(alloc / float(t["ltp"])))
                all_signals.append(sig)

                # -- EXIT GATES --
                if sym in execution.simulator.active_trades:
                    order = execution.simulator.active_trades[sym]
                    exit_reason = check_exit_conditions(order.side, order.entry_price, float(t["ltp"]), st.brick_size, b2c, p_long, p_short)
                    if exit_reason:
                        execution.close_position(sym, float(t["ltp"]), now, exit_reason)
                        logger.info(f"[Engine->Sim] EXIT {sym} @ {float(t['ltp']):.2f} | reason={exit_reason}")
                        log_brick_event(ts=now, symbol=sym, sector=st.sector, price=float(t["ltp"]), action="EXIT", reason=exit_reason, **latest_row)
                    continue

                # -- ENTRY GATES --
                log_brick_event(ts=now, symbol=sym, sector=st.sector, action="ENTRY" if gate_pass else "SKIP", reason=gate_reason if not gate_pass else "ALL_PASS", **latest_row)

                if gate_pass and not strategy.check_duplicate_minute(sym, now):
                    executable_signals.append(sig)

            # Execution
            if executable_signals:
                executable_signals.sort(key=lambda x: x["score"], reverse=True)
                for sig in executable_signals:
                    if is_trading_active(): execution.execute_trade(sig)

            # State Update
            top = risk_fortress.rank_signals(all_signals)
            latency = (time.time() - t0) * 1000
            if (time.time() - last_write) >= config.STATE_WRITE_INTERVAL:
                write_live_state(top, physics.renko_states, risk_fortress, latency, execution)
                last_write = time.time()

            time.sleep(max(0.001, 0.01 - (time.time() - t0)))

    except KeyboardInterrupt: logger.info("Stopped by user")
    finally: tick_provider.disconnect(); logger.info("Engine shut down.")

if __name__ == "__main__":
    run_live_engine()
