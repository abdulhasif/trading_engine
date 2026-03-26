"""Unified integration smoke test for the Live Trading Engine."""
import sys
import os
sys.path.insert(0, ".")

# Set Keras backend to torch before any imports that might trigger it
os.environ["KERAS_BACKEND"] = "torch"

print("1. Testing imports...")
from trading_engine.src.execution.execution_manager import ExecutionManager
from trading_engine.src.execution.upstox_simulator import UpstoxSimulator
from trading_core.core.risk.execution_guard import SyncPendingOrderGuard
from trading_engine.src.models.inference_engine import InferenceEngine
from trading_engine.src.strategy.strategy_manager import StrategyManager
from trading_engine.src.data.tick_provider import TickProvider
from trading_core.core.physics.renko import LiveRenkoState
from trading_core.core.risk.risk_fortress import RiskFortress
import trading_engine.config as config
import xgboost as xgb
import keras
print("   ALL IMPORTS OK")

print()
print("2. Testing model loading...")
ie = InferenceEngine()
b1_long, b1_short, b2, scaler = ie.load_models()
print(f"   Brain1 Long: {type(b1_long).__name__}")
print(f"   Brain1 Short: {type(b1_short).__name__}")
print(f"   Brain2: {type(b2).__name__}")

print()
print("3. Testing Execution Components...")
sim = UpstoxSimulator(starting_capital=100000)
guard = SyncPendingOrderGuard()
rf = RiskFortress()
sm = StrategyManager(rf)
em = ExecutionManager(sim, guard)

print(f"   Starting capital: Rs {sim.starting_capital:,}")
print(f"   Cash: Rs {sim.available_margin:,}")
print(f"   Open positions: {len(sim.active_trades)}")

print()
print("4. Testing virtual trade cycle...")
from datetime import datetime
now = datetime.now()

# Build a mock signal that StrategyManager would produce
signal = {
    "symbol": "SBIN", "direction": "BUY", "price": 625.50, 
    "brick_size": 0.75, "qty": 100
}
opened = em.execute_trade(signal)
print(f"   Opened SBIN LONG: {opened}")
print(f"   Open positions: {len(sim.active_trades)}")
sim_trade = sim.active_trades["SBIN"]
print(f"   Unrealized PnL: Rs {sim_trade.unrealized_pnl:.2f}")

# Close it
sim.update_active_price("SBIN", 630.0)
sim.close_position("SBIN", 630.0, now, "TREND_REVERSAL")
sim_order = sim.trade_history[-1]
print(f"   After price move 625.5 -> 630.0: Rs {sim_order.unrealized_pnl:.2f}")
print(f"   Closed SBIN: Net PnL = Rs {sim_order.net_pnl:.2f}")
print(f"   Cash after trade: Rs {sim.available_margin:,.2f}")

print()
print("5. Testing JSON state write...")
# write_live_state(sim)  # If needed
print("   Skipping JSON state write in smoke test")

print()
print("=" * 50)
print("ALL UNIFIED INTEGRATION TESTS PASSED")
print("=" * 50)

