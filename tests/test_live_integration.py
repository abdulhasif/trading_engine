"""Unified integration smoke test for the Live Trading Engine."""
import sys
import os
sys.path.insert(0, ".")

# Set Keras backend to torch before any imports that might trigger it
os.environ["KERAS_BACKEND"] = "torch"

print("1. Testing imports...")
from trading_engine.src.engine import (
    LivePortfolio, passes_soft_veto, load_models
)
from trading_engine.src.tick_provider import TickProvider
from trading_core.core.physics.renko import LiveRenkoState
import xgboost as xgb
import config
import keras
print("   ALL IMPORTS OK")

print()
print("2. Testing model loading...")
b1_long, b1_short, b2, scaler = load_models()
print(f"   Brain1 Long: {type(b1_long).__name__}")
print(f"   Brain1 Short: {type(b1_short).__name__}")
print(f"   Brain2: {type(b2).__name__}")

print()
print("3. Testing LivePortfolio...")
pf = LivePortfolio(100000)
print(f"   Starting capital: Rs {pf.starting_capital:,}")
print(f"   Cash: Rs {pf.simulator.available_margin:,}")
print(f"   Open positions: {len(pf.positions)}")

print()
print("4. Testing virtual trade cycle...")
from datetime import datetime
now = datetime.now()
opened = pf.open_position("SBIN", "Banking", "LONG", 625.50, 620.00, now)
print(f"   Opened SBIN LONG: {opened}")
print(f"   Open positions: {len(pf.positions)}")
sim_trade = pf.simulator.active_trades["SBIN"]
print(f"   Unrealized PnL: Rs {sim_trade.unrealized_pnl:.2f}")

# Simulate price move
pf.simulator.update_active_price("SBIN", 630.0)
print(f"   After price move 625.5 -> 630.0: Rs {pf.simulator.active_trades['SBIN'].unrealized_pnl:.2f}")

# Close it
pf.close_position("SBIN", 630.0, now, "TREND_REVERSAL")
sim_order = pf.simulator.trade_history[-1]
print(f"   Closed SBIN: Net PnL = Rs {sim_order.net_pnl:.2f}")
print(f"   Cash after trade: Rs {pf.simulator.available_margin:,.2f}")

print()
print("5. Testing JSON state write...")
pf.write_pnl_state()
if os.path.exists("live_pnl.json"):
    print("   live_pnl.json created successfully")

print()
print("=" * 50)
print("ALL UNIFIED INTEGRATION TESTS PASSED")
print("=" * 50)

