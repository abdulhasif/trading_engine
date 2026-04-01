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

# Rule-Based Strategy Imports
from trading_core.core.strategy.institutional_breakout import InstitutionalBreakout
from trading_core.core.strategy.river_mean_reversion import RiverMeanReversion

logger = logging.getLogger(__name__)

class StrategyManager:
    def __init__(self, risk_fortress):
        self.risk = risk_fortress
        self.last_entry_minutes = {}
        # Pure rule-based execution paths
        self.strats = [
            InstitutionalBreakout(),
            RiverMeanReversion()
        ]

    def evaluate_entry(self, symbol, sector, signal_str, b1p, b2c, latest_row_dict, st, sector_dir, portfolio_size, stock_losses, now):
        """
        STRICT: Execution restricted to specific rule-based strategies.
        Bypasses ML Base Probs, defaults to NO_STRAT_SIGNAL if none trigger.
        """
        # Execute Strategy Logic First
        strat_signal = None
        for strat in self.strats:
            res = strat.should_enter(
                brick_data={}, # Dummy for interface compat
                features=latest_row_dict,
                brain1_prob=b1p,
                brain2_conv=b2c,
            )
            if res:
                strat_signal = res
                break
                
        if not strat_signal:
            # Must return a fully populated dummy signal to prevent KeyError inside risk_fortress
            dummy_sig = {
                "symbol": symbol, "sector": sector, "direction": "FLAT",
                "qty": 0, "brain1_prob": b1p, "brain2_conviction": b2c, "score": 0.0,
                "velocity": float(latest_row_dict.get("velocity", 0)),
                "wick_pressure": float(latest_row_dict.get("wick_pressure", 0)),
                "rs": float(latest_row_dict.get("relative_strength", 0)),
                "price": float(latest_row_dict.get("brick_close", 0)),
                "brick_size": st.brick_size, "brick_count": len(st.bricks),
                "is_vetoed": False, "timestamp": now.isoformat(),
                "strategy_name": "NONE", "strategy_reason": "NO_STRAT_SIGNAL"
            }
            return False, "NO_STRAT_SIGNAL", {}, dummy_sig

        # Override ML default signals & force probability bypass
        signal_str = strat_signal.direction
        b1p = 1.0  # Force to bypass LOW_PROB gate
        b2c = 1.0  # Force to bypass LOW_CONVICTION gate
        
        rel_str_val = float(latest_row_dict.get("relative_strength", 0))
        score = self.risk.score_signal(b1p, b2c, (1 if signal_str == "LONG" else -1), sector_dir)

        # Build signal dict (STRICT format)
        sig = {
            "symbol": symbol, "sector": sector,
            "direction": "BUY" if signal_str == "LONG" else ("SELL" if signal_str == "SHORT" else "FLAT"),
            "qty": 0, # Calculated in main loop or ExecutionManager
            "brain1_prob": round(b1p, 4),
            "brain2_conviction": round(b2c, 2),
            "score": round(score, 2),
            "velocity": round(float(latest_row_dict.get("velocity",0)), 4),
            "wick_pressure": round(float(latest_row_dict.get("wick_pressure",0)), 4),
            "rs": round(rel_str_val, 4),
            "price": round(float(latest_row_dict.get("brick_close", 0)), 2), # Using close for signal price
            "brick_size": st.brick_size,
            "brick_count": len(st.bricks),
            "is_vetoed": self._passes_soft_veto(signal_str, rel_str_val),
            "timestamp": now.isoformat(),
            "strategy_name": strat_signal.strategy_name,
            "strategy_reason": strat_signal.entry_reason
        }

        # Entry Gate Check (Universal Guardrails Like Time, Portfolio, Streak)
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
            is_already_in_position = False, # Handled by loop
            structural_score = float(latest_row_dict.get("structural_score", 0.0))
        )
        
        # Append strat metadata into gate_reason if it passes
        if gate_pass:
            gate_reason = f"STRAT_PASS_({strat_signal.strategy_name})"

        return gate_pass, gate_reason, gate_audit, sig

    def evaluate_exit(self, order, current_price, brick_size, b2c, p_long, p_short, latest_row_dict):
        """
        Evaluates both universal structural exits and strategy-specific dynamic exits.
        """
        # Universal ML/Structural Stop Loss logic
        default_exit = check_exit_conditions(order.side, order.entry_price, current_price, brick_size, b2c, p_long, p_short)
        if default_exit:
            return default_exit
            
        # Iteration-specific dynamic exits
        for strat in self.strats:
            reason = strat.should_exit(
                position=order,
                brick_data={},
                features=latest_row_dict,
                brain1_prob=p_long,
                brain2_conv=b2c,
            )
            if reason:
                return reason
                
        return None

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
