"""
src/live/state_manager.py - Dashboard State & JSON Updates
===========================================================
Handles serialization and writing for live_state.json.
STRICT LOGIC PRESERVATION from engine_main.py.
"""

import json
import logging
from datetime import datetime
from trading_engine import config
from trading_api.src.services.market_service import compute_market_regime
from trading_api.src.services.news_service import get_sentiment_feed as _get_sentiment_feed

logger = logging.getLogger(__name__)

def write_live_state(top_signals, renko_states, risk_fortress, latency_ms, execution_manager):
    """
    STRICT: Verbatim from engine_main.py.
    """
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
    active_trades = _serialize_active_trades(execution_manager.simulator)
    margin_usage  = _serialize_margin(execution_manager.simulator)
    live_pnl = 0.0
    try:
        live_pnl = execution_manager.simulator.get_live_pnl()
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
            "drift_accuracy": risk_fortress.drift_accuracy,
            "yellow_alert": risk_fortress.yellow_alert,
            "active_symbols": len(renko_states),
        },
    }
    try:
        with open(config.LIVE_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"State write failed: {e}")

def _serialize_active_trades(simulator):
    """STRICT: Verbatim from engine_main.py."""
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

def _serialize_margin(simulator):
    """STRICT: Verbatim from engine_main.py."""
    if simulator is None:
        return {"total_capital": 0.0, "available_margin": 0.0,
                "locked_margin": 0.0, "margin_usage_pct": 0.0}
    try:
        return simulator.get_margin_usage()
    except Exception:
        return {"total_capital": 0.0, "available_margin": 0.0,
                "locked_margin": 0.0, "margin_usage_pct": 0.0}

def is_trading_active() -> bool:
    """STRICT: Verbatim from engine_main.py."""
    if not config.TRADE_CONTROL_FILE.exists():
        return True
    try:
        with open(config.TRADE_CONTROL_FILE, "r") as f:
            data = json.load(f)
            return data.get("active", True)
    except Exception:
        return True
