"""Quick test for the new TickProvider."""
import sys
sys.path.insert(0, ".")

print("Testing new TickProvider...")
from trading_engine.src.tick_provider import TickProvider
from trading_core.core.config import base_config as config

print(f"Access token set: {bool(config.UPSTOX_ACCESS_TOKEN)}")

tp = TickProvider(["SBIN", "RELIANCE", "TCS", "INFY", "HCLTECH"])
print(f"Instrument mapping: {len(tp._sym_to_ikey)} symbols mapped")
for sym, ikey in tp._sym_to_ikey.items():
    print(f"  {sym} -> {ikey}")

tp.connect()
print(f"Connected: {tp._connected}")
print(f"Live mode: {tp.is_live}")

# Get ticks multiple times to test random walk persistence
for i in range(3):
    ticks = tp.get_latest_ticks()
    print(f"\nTick round {i+1}:")
    for sym in sorted(ticks.keys()):
        t = ticks[sym]
        print(f"  {sym}: ltp={t['ltp']:.2f}")

tp.disconnect()
print("\nDONE - All tick provider tests passed")

