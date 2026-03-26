"""
src/live/upstox_simulator.py - Upstox Live Trading Simulator
============================================================
Production-ready, Object-Oriented simulator that perfectly mirrors
the live trading mechanics of the Upstox API for Intraday (MIS) Equity.

Structural Pillars:
  1. Margin & Buying Power Management (5x Leverage Rule)
  2. Hyper-Accurate Transaction Friction (Upstox Intraday Math)
  3. Live Trade Lifecycle & State Tracking (PENDING, ACTIVE, CLOSED)
  4. Institutional Reporting & Analytics Dashboard

Author: Quant & Execution Architecture Team
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

from trading_engine import config

logger = logging.getLogger(__name__)


@dataclass
class TradeState:
    PENDING = "PENDING"
    ACTIVE  = "ACTIVE"
    CLOSED  = "CLOSED"
    REJECTED= "REJECTED"


@dataclass
class SimulatedOrder:
    trade_id:       int
    symbol:         str
    side:           str        # "BUY" or "SELL"
    qty:            int
    entry_price:    float
    created_at:     datetime
    sl_price:       float      = 0.0

    state:          str        = TradeState.PENDING
    locked_margin:  float      = 0.0

    # Active State Stats
    filled_at:      Optional[datetime] = None
    last_price:     float      = 0.0

    # Closed State Stats
    closed_at:      Optional[datetime] = None
    exit_price:     Optional[float]    = None
    exit_reason:    str        = ""

    # Taxation & Friction
    brokerage_paid: float      = 0.0
    stt_paid:       float      = 0.0
    exchange_txn:   float      = 0.0
    gst_paid:       float      = 0.0
    sebi_paid:      float      = 0.0
    stamp_duty:     float      = 0.0
    total_friction: float      = 0.0

    # PnL
    gross_pnl:      float      = 0.0
    net_pnl:        float      = 0.0

    @property
    def unrealized_pnl(self) -> float:
        """MTM PnL for an ACTIVE position using the latest tick price."""
        if self.state != TradeState.ACTIVE or self.last_price == 0.0:
            return 0.0
        if self.side in ("BUY", "LONG"):  # BUG FIX: paper trader passes 'LONG'/'SHORT'
            return (self.last_price - self.entry_price) * self.qty
        return (self.entry_price - self.last_price) * self.qty


class UpstoxSimulator:
    """
    Simulates the Upstox Intraday (MIS) Equity trading environment.
    Enforces strict 5x margin limits and exact government/broker taxation.
    """

    # Upstox MIS Equity Constants - Centralized in config.py
    LEVERAGE         = config.SIM_LEVERAGE
    BROKERAGE_MAX    = config.SIM_BROKERAGE_MAX
    BROKERAGE_PCT    = config.SIM_BROKERAGE_PCT
    STT_SELL_PCT     = config.SIM_STT_SELL_PCT
    STAMP_BUY_PCT    = config.SIM_STAMP_BUY_PCT
    EXCHANGE_PCT     = config.SIM_EXCHANGE_PCT
    SEBI_PCT         = config.SIM_SEBI_PCT
    GST_PCT          = config.SIM_GST_PCT

    def __init__(self, starting_capital: float = config.SIM_STARTING_CAPITAL):
        # 1. Margin & Buying Power State
        self.starting_capital: float = starting_capital
        self.total_capital:    float = starting_capital
        self.locked_margin:    float = 0.0
        self.available_margin: float = starting_capital

        # Memory / Ledger
        self._trade_id_counter: int = 0
        self.active_trades:  Dict[str, SimulatedOrder] = {}  # symbol -> Order
        self.pending_orders: Dict[str, SimulatedOrder] = {}  # symbol -> Order
        self.trade_history:  List[SimulatedOrder]      = []

    # ========================================================================
    # PILLAR 1: MARGIN & BUYING POWER MANAGEMENT
    # ========================================================================

    @property
    def total_buying_power(self) -> float:
        """Maximum possible exposure allowed by the broker (5x MIS)."""
        return self.available_margin * self.LEVERAGE

    def _lock_margin(self, required_margin: float) -> bool:
        """Attempt to lock margin. Returns True if successful, False if NSF."""
        if required_margin > self.available_margin:
            return False
        self.available_margin -= required_margin
        self.locked_margin += required_margin
        return True

    def _release_margin(self, locked_amount: float, net_pnl: float) -> None:
        """Release margin back to capital, factoring in the realized PnL/Loss."""
        self.locked_margin -= locked_amount
        self.total_capital += net_pnl
        self.available_margin = self.total_capital - self.locked_margin

    # ========================================================================
    # PILLAR 2: HYPER-ACCURATE TRANSACTION FRICTION
    # ========================================================================

    def _calculate_taxes(self, order: SimulatedOrder) -> None:
        """
        Calculates exact Indian taxation and Upstox brokerage for a round trip.
        Mutates the order object with the specific fee breakdown.
        """
        if order.exit_price is None:
            return

        buy_turnover  = (order.entry_price * order.qty) if order.side in ("BUY", "LONG") else (order.exit_price * order.qty)
        sell_turnover = (order.exit_price * order.qty) if order.side in ("BUY", "LONG") else (order.entry_price * order.qty)
        total_turnover = buy_turnover + sell_turnover

        # 1. Brokerage: Lower of Rs 20 or 0.05% per side
        brok_entry = min(self.BROKERAGE_MAX, (order.entry_price * order.qty) * self.BROKERAGE_PCT)
        brok_exit  = min(self.BROKERAGE_MAX, (order.exit_price * order.qty)  * self.BROKERAGE_PCT)
        order.brokerage_paid = brok_entry + brok_exit

        # 2. STT: 0.025% on Sell Side Only
        order.stt_paid = sell_turnover * self.STT_SELL_PCT

        # 3. Stamp Duty: 0.003% on Buy Side Only
        order.stamp_duty = buy_turnover * self.STAMP_BUY_PCT

        # 4. Exchange Transaction Charge: 0.00297% on both sides
        order.exchange_txn = total_turnover * self.EXCHANGE_PCT

        # 5. SEBI Turnover Fee: Rs 10 per Crore
        order.sebi_paid = total_turnover * self.SEBI_PCT

        # 6. GST: 18% applied strictly on (Brokerage + Exchange)
        order.gst_paid = (order.brokerage_paid + order.exchange_txn) * self.GST_PCT

        # Total Friction
        order.total_friction = (
            order.brokerage_paid + order.stt_paid + order.stamp_duty +
            order.exchange_txn + order.sebi_paid + order.gst_paid
        )

        # Calculate PnL
        if order.side in ("BUY", "LONG"):
            order.gross_pnl = (order.exit_price - order.entry_price) * order.qty
        else:
            order.gross_pnl = (order.entry_price - order.exit_price) * order.qty

        order.net_pnl = order.gross_pnl - order.total_friction

    # ========================================================================
    # PILLAR 3: LIVE TRADE LIFECYCLE & STATE TRACKING
    # ========================================================================

    def place_order(self, symbol: str, side: str, qty: int, price: float, sl_price: float, ts: datetime) -> SimulatedOrder:
        """
        Step 1: PENDING State.
        Creates an order and attempts to lock margin. Rejects if funds insufficient.
        """
        self._trade_id_counter += 1
        notional_value = price * qty
        required_margin = notional_value / self.LEVERAGE

        order = SimulatedOrder(
            trade_id=self._trade_id_counter,
            symbol=symbol,
            side=side.upper(),
            qty=qty,
            entry_price=price,
            sl_price=sl_price,
            created_at=ts,
            locked_margin=required_margin
        )

        if symbol in self.active_trades or symbol in self.pending_orders:
            logger.warning(f"Order REJECTED: {symbol} already has an active/pending position.")
            order.state = TradeState.REJECTED
            return order

        if not self._lock_margin(required_margin):
            logger.warning(f"Order REJECTED: Insufficient Funds. "
                           f"Req: Rs {required_margin:,.2f} | Avail: Rs {self.available_margin:,.2f}")
            order.state = TradeState.REJECTED
            return order

        order.state = TradeState.PENDING
        self.pending_orders[symbol] = order
        logger.info(f"Order PENDING: {side} {qty} {symbol} @ Rs {price:,.2f}. "
                    f"Locked Margin: Rs {required_margin:,.2f}")
        return order

    def fill_pending_order(self, symbol: str, ts: datetime) -> bool:
        """
        Step 2: ACTIVE State.
        Simulates the broker confirming the fill. Moves from PENDING to ACTIVE.
        """
        if symbol not in self.pending_orders:
            return False

        order = self.pending_orders.pop(symbol)
        order.state = TradeState.ACTIVE
        order.filled_at = ts
        order.last_price = order.entry_price
        self.active_trades[symbol] = order
        logger.info(f"Order FILLED/ACTIVE: {order.side} {order.symbol}.")
        return True

    def cancel_pending_order(self, symbol: str, ts: datetime, reason: str) -> bool:
        """Cancel a PENDING order and release its locked margin."""
        if symbol not in self.pending_orders:
            return False

        order = self.pending_orders.pop(symbol)
        order.state = TradeState.REJECTED
        order.closed_at = ts
        order.exit_reason = reason
        self._release_margin(order.locked_margin, 0.0)
        self.trade_history.append(order)
        logger.info(f"Order CANCELLED: {order.side} {order.symbol} | Reason: {reason}")
        return True

    def update_active_price(self, symbol: str, current_price: float) -> None:
        """Update the MTM (Mark-to-Market) price of an active position."""
        if symbol in self.active_trades:
            self.active_trades[symbol].last_price = current_price

    def close_position(self, symbol: str, exit_price: float, ts: datetime, reason: str) -> bool:
        """
        Step 3: CLOSED State.
        Closes position, calculates exact taxes, records PnL, and releases margin.
        """
        if symbol not in self.active_trades:
            return False

        order = self.active_trades.pop(symbol)
        order.state = TradeState.CLOSED
        order.exit_price = exit_price
        order.closed_at = ts
        order.exit_reason = reason

        self._calculate_taxes(order)
        self._release_margin(order.locked_margin, order.net_pnl)
        self.trade_history.append(order)

        logger.info(f"Position CLOSED: {order.symbol} @ Rs {exit_price:,.2f} | "
                    f"Reason: {reason} | Net PnL: Rs {order.net_pnl:,.2f} "
                    f"(Friction: Rs {order.total_friction:,.2f})")
        return True

    # ========================================================================
    # PILLAR 4: INSTITUTIONAL REPORTING & ANALYTICS DASHBOARD
    # ========================================================================

    def get_trade_ledger(self) -> pd.DataFrame:
        """Returns the complete historical trade-by-trade ledger."""
        if not self.trade_history:
            return pd.DataFrame()

        records = []
        cumu_pnl = 0.0
        for t in self.trade_history:
            cumu_pnl += t.net_pnl
            records.append({
                "Trade_ID":      t.trade_id,
                "Symbol":        t.symbol,
                "Side":          t.side,
                "Qty":           t.qty,
                "Entry_Time":    t.filled_at,
                "Entry_Price":   round(t.entry_price, 2),
                "Exit_Time":     t.closed_at,
                "Exit_Price":    round(t.exit_price, 2) if t.exit_price else None,
                "Reason":        t.exit_reason,
                "Gross_PnL":     round(t.gross_pnl, 2),
                "Brokerage":     round(t.brokerage_paid, 2),
                "STT":           round(t.stt_paid, 2),
                "GST_18%":       round(t.gst_paid, 2),
                "Total_Friction":round(t.total_friction, 2),
                "Net_PnL":       round(t.net_pnl, 2),
                "Capital_Curve": round(self.starting_capital + cumu_pnl, 2),
            })
        return pd.DataFrame(records)

    def generate_daily_summary(self, target_date: datetime.date = None) -> dict:
        """Daily aggregation of PnL, taxes, and win rates."""
        if target_date is None:
            target_date = datetime.now().date()

        df = self.get_trade_ledger()
        if df.empty:
            return {"Date": str(target_date), "Trades": 0, "Net_PnL": 0.0}

        # Filter for the specific day
        df["Exit_Date"] = pd.to_datetime(df["Exit_Time"]).dt.date
        day_df = df[df["Exit_Date"] == target_date]

        if day_df.empty:
            return {"Date": str(target_date), "Trades": 0, "Net_PnL": 0.0}

        trades    = len(day_df)
        wins      = len(day_df[day_df["Net_PnL"] > 0])
        win_rate  = (wins / trades * 100) if trades > 0 else 0.0
        gross_pnl = day_df["Gross_PnL"].sum()
        total_tax = day_df["Total_Friction"].sum()
        net_pnl   = day_df["Net_PnL"].sum()

        return {
            "Date":              str(target_date),
            "Total_Trades":      trades,
            "Win_Ratio_%":       round(win_rate, 2),
            "Gross_PnL_Rs ":       round(gross_pnl, 2),
            "Total_Taxes_Paid_Rs ":round(total_tax, 2),
            "Net_PnL_Rs ":         round(net_pnl, 2),
            "Ending_Capital_Rs ":  round(self.total_capital, 2)
        }

    def generate_all_time_summary(self) -> dict:
        """Overall historical summary for the entire simulator run."""
        df = self.get_trade_ledger()
        if df.empty:
            return {"Status": "No trades taken"}

        trades    = len(df)
        wins      = len(df[df["Net_PnL"] > 0])
        win_rate  = (wins / trades * 100) if trades > 0 else 0.0
        
        # Max Drawdown calculation using the capital curve
        rolling_max = df["Capital_Curve"].cummax()
        drawdown    = (df["Capital_Curve"] - rolling_max) / rolling_max
        max_dd      = drawdown.min() * 100

        return {
            "Total_Trades":       trades,
            "All_Time_Win_Ratio_%": round(win_rate, 2),
            "Max_Drawdown_%":     round(max_dd, 2),
            "Gross_PnL_Rs ":        round(df["Gross_PnL"].sum(), 2),
            "Total_Taxes_Paid_Rs ": round(df["Total_Friction"].sum(), 2),
            "Net_PnL_Rs ":          round(df["Net_PnL"].sum(), 2),
            "Starting_Capital_Rs ": round(self.starting_capital, 2),
            "Current_Capital_Rs ":  round(self.total_capital, 2),
        }

    # ========================================================================
    # PILLAR 5: ANDROID API INTERFACE (Kill Switch + Telemetry Helpers)
    # ========================================================================

    def square_off_all(self, ts: datetime = None) -> int:
        """
        Emergency square-off: closes every active position at its last_price.
        Called by the GLOBAL_KILL switch from the Android app.
        Returns the number of positions closed.
        """
        if ts is None:
            ts = datetime.now()
        symbols = list(self.active_trades.keys())
        for symbol in symbols:
            order = self.active_trades.get(symbol)
            if order and order.last_price > 0:
                self.close_position(symbol, order.last_price, ts, "GLOBAL_KILL")
            elif order:
                self.close_position(symbol, order.entry_price, ts, "GLOBAL_KILL")
        logger.warning(f"square_off_all: closed {len(symbols)} position(s) via GLOBAL_KILL.")
        return len(symbols)

    def get_live_pnl(self) -> float:
        """
        Returns the aggregate unrealized MTM PnL across all active positions.
        Broadcast over the WebSocket telemetry feed every second.
        """
        return round(sum(o.unrealized_pnl for o in self.active_trades.values()), 2)

    def get_margin_usage(self) -> dict:
        """
        Returns a margin snapshot for the WebSocket telemetry feed.
        margin_usage_pct = locked / total_capital * 100
        """
        usage_pct = (
            (self.locked_margin / self.total_capital * 100)
            if self.total_capital > 0 else 0.0
        )
        return {
            "total_capital":    round(self.total_capital, 2),
            "available_margin": round(self.available_margin, 2),
            "locked_margin":    round(self.locked_margin, 2),
            "margin_usage_pct": round(usage_pct, 2),
        }

