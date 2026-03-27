"""
src/live/execution_manager.py - Order Execution & Management
=============================================================
Wraps the UpstoxSimulator and SyncPendingOrderGuard.
STRICT LOGIC PRESERVATION from engine_main.py.
"""

import logging
from datetime import datetime
from trading_core.core.config import base_config as config

logger = logging.getLogger(__name__)

class ExecutionManager:
    def __init__(self, simulator, order_guard):
        self.simulator = simulator
        self.order_guard = order_guard

    def execute_trade(self, signal):
        """
        STRICT: Verbatim from engine_main.py.
        """
        symbol    = signal["symbol"]
        side      = signal["direction"]   # "BUY" or "SELL"
        price     = signal["price"]
        ts        = datetime.now()

        # Fix 4: Real-World Execution Latency (Slippage Mirage)
        slippage   = config.T1_SLIPPAGE_PCT
        fill_price = price * (1.0 + slippage) if side == "BUY" else price * (1.0 - slippage)

        # -- Fix 5: Non-blocking mutex - drop signal if already pending --------
        if not self.order_guard.try_acquire(symbol, side):
            return False

        # Calculate SL level: Entry +/- (STRUCTURAL_REVERSAL_BRICKS * brick_size)
        brick_size = signal["brick_size"]
        sl_dist    = config.STRUCTURAL_REVERSAL_BRICKS * brick_size
        sl_price   = fill_price - sl_dist if side == "BUY" else fill_price + sl_dist

        try:
            # Delegate to UpstoxSimulator
            req = self.simulator.place_order(
                symbol   = symbol,
                side     = side,
                qty      = signal.get("qty", 1),
                price    = fill_price,
                sl_price = sl_price,
                ts       = ts
            )
            if req and req.state != "REJECTED":
                self.simulator.fill_pending_order(symbol, ts)
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
            self.order_guard.release(symbol)

    def square_off_all(self, now):
        """STRICT: End-of-day square-off."""
        if self.simulator is not None:
            self.simulator.square_off_all(now)

    def close_position(self, symbol, ltp, now, reason):
        """STRICT: Exit condition execution."""
        if self.simulator is not None:
            self.simulator.close_position(symbol, ltp, now, reason)

    def update_active_price(self, symbol, ltp):
        """STRICT: Live MTM update."""
        if self.simulator is not None:
            self.simulator.update_active_price(symbol, ltp)

    def get_portfolio_state(self):
        """STRICT: Returns data for live_state.json."""
        return {
            "active_trades": self.simulator.active_trades,
            "live_pnl": self.simulator.get_live_pnl(),
            "margin_usage": self.simulator.get_margin_usage()
        }
