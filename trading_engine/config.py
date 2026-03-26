from trading_core.core.config.base_config import *

# (Inherited from base_config: UPSTOX_API_BASE, UPSTOX_ACCESS_TOKEN, API_DELAY_BETWEEN_CALLS)

TICK_RECONNECT_DELAYS   = [5, 10, 20, 40, 60]
TICK_FLUSH_INTERVAL     = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# 3. MARKET HOURS & TRADING WINDOWS (IST)
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_WAKE_HOUR       = 8;   SYSTEM_WAKE_MINUTE       = 50
WARMUP_HOUR            = 9;   WARMUP_MINUTE            = 5
MARKET_OPEN_HOUR       = 9;   MARKET_OPEN_MINUTE       = 15
MARKET_CLOSE_HOUR      = 15;  MARKET_CLOSE_MINUTE      = 30
SYSTEM_SHUTDOWN_HOUR   = 15;  SYSTEM_SHUTDOWN_MINUTE   = 35

ENTRY_LOCK_MINUTES     = 30
NO_NEW_ENTRY_HOUR      = 14
NO_NEW_ENTRY_MIN       = 30
EOD_SQUARE_OFF_HOUR    = 15
EOD_SQUARE_OFF_MIN     = 14

# ─────────────────────────────────────────────────────────────────────────────
# 6. TRADING STRATEGY & EXECUTION (SNIPER SETTINGS)
# ─────────────────────────────────────────────────────────────────────────────
LONG_ENTRY_PROB_THRESH   = 0.60
SHORT_ENTRY_PROB_THRESH  = 0.60
RAW_LONG_ENTRY_PROB_THRESH  = 0.72
RAW_SHORT_ENTRY_PROB_THRESH = 0.72
ENTRY_CONV_THRESH        = 0.40
STRONG_CONVICTION_THRESH = 1.0
BIAS_ENTRY_THRESHOLD     = 0.65
VETO_BYPASS_CONV         = 4.0

ENTRY_RS_THRESHOLD     = -0.5
MAX_VWAP_ZSCORE        = 3.0
MAX_ENTRY_WICK         = 0.50
MIN_PRICE_FILTER       = 100.0
MIN_CONSECUTIVE_BRICKS = 1
MIN_BRICKS_TODAY       = 0
STREAK_LIMIT           = 7
BRICK_COOLDOWN         = 3

STRUCTURAL_WINDOW        = 50
STRUCTURAL_THRESHOLD     = 0.85
STRUCTURAL_PROB_BUMP     = 0.10

VOLUME_LIMIT_PCT       = 0.05
MIN_CANDLE_VOLUME      = 500

STRUCTURAL_REVERSAL_BRICKS = 6
TRAIL_ACTIVATION_BRICKS    = 8.0
TRAIL_DISTANCE_BRICKS      = 3.0
MAX_HOLD_BRICKS            = 300
HYST_LONG_SELL_FLOOR       = 0.40
HYST_SHORT_SELL_CEIL       = 0.60
EXIT_CONV_THRESH           = 0.0

# Execution Realism
T1_SLIPPAGE_PCT          = 0.0005
TRANSACTION_COST_PCT     = 0.00075
JITTER_SECONDS           = 1.0
PATH_CONFLICT_PESSIMISM  = True

# ─────────────────────────────────────────────────────────────────────────────
# 9. SIMULATOR & FRICTION MECHANICS
# ─────────────────────────────────────────────────────────────────────────────
SIM_STARTING_CAPITAL = 100000.0
SIM_LEVERAGE         = 5.0
SIM_BROKERAGE_MAX    = 20.0
SIM_BROKERAGE_PCT    = 0.0005
SIM_STT_SELL_PCT     = 0.00025
SIM_STAMP_BUY_PCT    = 0.00003
SIM_EXCHANGE_PCT     = 0.0000297
SIM_SEBI_PCT         = 0.000001
SIM_GST_PCT          = 0.18

