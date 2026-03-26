"""
src/live/strategy_manager.py - Strategy Logic & Gating
======================================================
Centralizes entry/exit gates and signal evaluation.
STRICT LOGIC PRESERVATION from engine_main.py and strategy.py.
"""

import logging
from datetime import datetime
from trading_engine import config
from trading_core.core.risk.strategy import check_entry_gates, check_exit_conditions

logger = logging.getLogger(__name__)

class StrategyManager:
    def __init__(self, risk_fortress):
        self.risk = risk_fortress
        self.last_entry_minutes = {}

    def evaluate_entry(self, symbol, sector, signal_str, b1p, b2c, latest_row_dict, st, sector_dir, portfolio_size, stock_losses, now, dynamic_conv_thresh=None, river_win_ratio=0.0, river_pullback_cleared=False):
        """
        Combined logic for gates and signal building.
        Phase 5: dynamic_conv_thresh from DynamicThresholdTracker.
        Phase 3: river_win_ratio + river_pullback_cleared from The River.
        """
        rel_str_val = float(latest_row_dict.get("relative_strength", 0))
        score = self.risk.score_signal(b1p, b2c, (1 if signal_str == "LONG" else -1), sector_dir)

        # Build signal dict
        sig = {
            "symbol": symbol, "sector": sector,
            "direction": "BUY" if signal_str == "LONG" else ("SELL" if signal_str == "SHORT" else "FLAT"),
            "qty": 0,
            "brain1_prob": round(b1p, 4),
            "brain2_conviction": round(b2c, 2),
            "score": round(score, 2),
            "velocity": round(float(latest_row_dict.get("velocity",0)), 4),
            "wick_pressure": round(float(latest_row_dict.get("wick_pressure",0)), 4),
            "rs": round(rel_str_val, 4),
            "price": round(float(latest_row_dict.get("brick_close", 0)), 2),
            "brick_size": st.brick_size,
            "brick_count": len(st.bricks),
            "is_vetoed": self._passes_soft_veto(signal_str, rel_str_val),
            "timestamp": now.isoformat(),
        }

        # Entry Gate Check
        recent_bricks = st.bricks[-config.MIN_CONSECUTIVE_BRICKS:] if len(st.bricks) >= config.MIN_CONSECUTIVE_BRICKS else []
        recent_dirs = [rb["direction"] for rb in recent_bricks]
        
        gate_pass, gate_reason, gate_audit = check_entry_gates(
            symbol = symbol,
            now = now,
            price = sig["price"],
            b1p = b1p,
            b2c = b2c,
            signal_str = signal_str,
            rel_str = rel_str_val,
            wick_p = float(latest_row_dict.get("wick_pressure", 0)),
            z_vwap = float(latest_row_dict.get("vwap_zscore", 0)),
            streak_count = int(latest_row_dict.get("consecutive_same_dir", 0)),
            brick_dir = int(latest_row_dict.get("direction", 0)),
            recent_dirs = recent_dirs,
            stock_losses = stock_losses,
            portfolio_size = portfolio_size,
            is_already_in_position = False,
            structural_score = float(latest_row_dict.get("structural_score", 0.0)),
            dynamic_conv_thresh = dynamic_conv_thresh,
            river_win_ratio = river_win_ratio,               # Phase 3
            river_pullback_cleared = river_pullback_cleared   # Phase 3
        )

        # Phase 3: Propagate trade_type from gate audit into signal dict
        sig["trade_type"] = gate_audit.get("trade_type", "NORMAL")

        return gate_pass, gate_reason, gate_audit, sig

    def _passes_soft_veto(self, signal, rel_strength):
        """STRICT: Verbatim from engine_main.py"""
        if signal == "LONG" and rel_strength < -config.SOFT_VETO_THRESHOLD:
            return False
        if signal == "SHORT" and rel_strength > config.SOFT_VETO_THRESHOLD:
            return False
        return True

    def check_duplicate_minute(self, symbol, now):
        """STRICT: Hyper-trading protection."""
        current_minute = now.replace(second=0, microsecond=0)
        if self.last_entry_minutes.get(symbol) == current_minute:
            return True
        self.last_entry_minutes[symbol] = current_minute
        return False
