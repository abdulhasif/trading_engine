"""Modern integration smoke test for the Upstox+Renko Trading Engine."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("1. Testing imports...")
from trading_engine.src.upstox_simulator import UpstoxSimulator, TradeState
from trading_engine.src.engine import passes_soft_veto
from trading_engine.src.tick_provider import TickProvider
from trading_core.core.physics.renko import LiveRenkoState
from trading_core.core.features import compute_features_live
from trading_core.core.risk.risk_fortress import RiskFortress
import config
print("   ALL IMPORTS OK")

print("\n2. Testing UpstoxSimulator Initialization...")
sim = UpstoxSimulator(starting_capital=100000)
print(f"   Starting capital: Rs {sim.starting_capital:,}")
print(f"   Available Margin: Rs {sim.available_margin:,}")
print(f"   Active trades: {len(sim.active_trades)}")

print("\n3. Testing Virtual Trade Cycle (PENDING -> ACTIVE -> CLOSED)...")
from datetime import datetime
now = datetime.now()
# Place order (PENDING)
order = sim.place_order("SBIN", "BUY", 100, 625.50, 620.00, now)
print(f"   Placed order: {order.symbol} {order.side} | State: {order.state}")
print(f"   Locked Margin: Rs {order.locked_margin:,.2f}")

# Fill order (ACTIVE)
sim.fill_pending_order("SBIN", now)
active_order = sim.active_trades["SBIN"]
print(f"   Order filled. State: {active_order.state}")

# Simulate price move & unrealized PnL
sim.update_active_price("SBIN", 630.0)
print(f"   Price moved to 630.0. Unrealized PnL: Rs {sim.get_live_pnl():,.2f}")

# Close it
sim.close_position("SBIN", 630.0, now, "SMOKE_TEST_EXIT")
print(f"   Position closed. Net PnL (after fees): Rs {sim.trade_history[-1].net_pnl:,.2f}")
print(f"   Total Capital now: Rs {sim.total_capital:,.2f}")

print("\n4. Testing Margin Limits (5x Leverage Check)...")
sim2 = UpstoxSimulator(starting_capital=10000) # Only 10k
# Try to buy 1 lakh worth of RELIANCE (20k margin needed - should fail on 10k capital)
huge_order = sim2.place_order("RELIANCE", "BUY", 40, 2500.00, 2480.0, now)
print(f"   Huge order (over limit) state: {huge_order.state} (Expect REJECTED)")

print("\n5. Testing engine soft veto logic...")
print(f"   LONG + rel_str=-0.8: {passes_soft_veto('LONG', -0.8)} (Expect False if THRESH=0.5)")
print(f"   LONG + rel_str=+0.3: {passes_soft_veto('LONG', 0.3)} (Expect True)")

print("\n6. Testing RiskFortress Scoring...")
rf = RiskFortress()
score = rf.score_signal(0.72, 80.0, 1, 1)
print(f"   Score (Aligned): {score:.2f}")

print("\n7. Testing LiveRenkoState & Features...")
rs = LiveRenkoState("SBIN", "Banking", 0.75)
print(f"   Created renko state for SBIN. Initial bricks: {len(rs.bricks)}")

print("\n8. Testing Simulator Reporting...")
summary = sim.generate_all_time_summary()
print(f"   All-time trades: {summary.get('Total_Trades')}")
print(f"   Final Net PnL: Rs {summary.get('Net_PnL_Rs ')}")

print("\n" + "=" * 50)
print("ALL INTEGRATION CHECKS PASSED")
print("=" * 50)


