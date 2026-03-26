import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 1. SYSTEM PATHS & DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORAGE_DIR  = PROJECT_ROOT / "storage"
DATA_DIR     = STORAGE_DIR  / "data"        # [SHARED: Pipeline, Core]
FEATURES_DIR = STORAGE_DIR  / "features"    # [SHARED: Pipeline, Core]
MODELS_DIR   = STORAGE_DIR  / "models"      # [SHARED: Pipeline]
LOGS_DIR     = STORAGE_DIR  / "logs"        # [SHARED: API, Pipeline]
CONFIG_DIR   = PROJECT_ROOT / "config_data"

UNIVERSE_CSV = CONFIG_DIR / "sector_universe.csv" # [SHARED: API, Pipeline]
LIVE_STATE_FILE = PROJECT_ROOT / "live_state.json" # [SHARED: API]
LIVE_LOG_FILE   = LOGS_DIR / "live_engine.log"
TRADE_CONTROL_FILE = LOGS_DIR / "trade_control.json"

# ─────────────────────────────────────────────────────────────────────────────
# 2. UPSTOX API & CONNECTION SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
UPSTOX_ACCESS_TOKEN   = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI2R0I1OTUiLCJqdGkiOiI2OWMzNTBhYjk4MGMyODExZDNlOTgxYWEiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6ZmFsc2UsImlhdCI6MTc3NDQwNzg1MSwiaXNzIjoidWRhcGktZ2F0ZXdheS1zZXJ2aWNlIiwiZXhwIjoxNzc0NDc2MDAwfQ.dWqoTa0yt0HIipi3Z4PuhznpNpzfoWBI3fcPN1AbiG0" # [SHARED: Pipeline]
TICK_RECONNECT_DELAYS   = [5, 10, 20, 40, 60]
TICK_FLUSH_INTERVAL     = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# 3. MARKET HOURS & TRADING WINDOWS (IST)
# ─────────────────────────────────────────────────────────────────────────────
WARMUP_HOUR            = 9
MARKET_OPEN_HOUR       = 9;   MARKET_OPEN_MINUTE       = 15 # [SHARED: Pipeline, Core]
SYSTEM_SHUTDOWN_HOUR   = 15

# ─────────────────────────────────────────────────────────────────────────────
# 5. ML MODEL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BRAIN1_CNN_LONG_PATH          = MODELS_DIR / "brain1_cnn_long.keras" # [SHARED: Pipeline]
BRAIN1_CNN_SHORT_PATH         = MODELS_DIR / "brain1_cnn_short.keras" # [SHARED: Pipeline]
BRAIN1_SCALER_PATH            = MODELS_DIR / "scaler.pkl"             # [SHARED: Pipeline]
BRAIN2_MODEL_PATH             = MODELS_DIR / "brain2_meta.json"       # [SHARED: Pipeline]
BRAIN2_FEATURES               = ["brain1_prob_long", "brain1_prob_short", "trade_direction", "velocity", "momentum_acceleration", "feature_tib_zscore", "wick_pressure", "relative_strength", "feature_vpb_roc", "regime_morning", "regime_midday", "regime_afternoon", "feature_cvd_divergence"] # [SHARED: Pipeline]
CNN_WINDOW_SIZE               = 15 # [SHARED: Pipeline]
USE_CALIBRATED_MODELS         = True # [SHARED: Pipeline, Core]

# ─────────────────────────────────────────────────────────────────────────────
# 6. TRADING STRATEGY & EXECUTION (SNIPER SETTINGS)
# ─────────────────────────────────────────────────────────────────────────────
LONG_ENTRY_PROB_THRESH   = 0.60 # [SHARED: Pipeline, Core]
SHORT_ENTRY_PROB_THRESH  = 0.60 # [SHARED: Pipeline, Core]
RAW_LONG_ENTRY_PROB_THRESH  = 0.72 # [SHARED: Pipeline, Core]
RAW_SHORT_ENTRY_PROB_THRESH = 0.72 # [SHARED: Pipeline, Core]
ENTRY_CONV_THRESH        = 0.40 # [SHARED: Pipeline, Core]

ENTRY_RS_THRESHOLD     = -0.5 # [SHARED: Pipeline, Core]
MAX_ENTRY_WICK         = 0.50 # [SHARED: Pipeline, Core]
MIN_PRICE_FILTER       = 100.0 # [SHARED: Pipeline, Core]
MIN_CONSECUTIVE_BRICKS = 1 # [SHARED: Pipeline, Core]
MIN_BRICKS_TODAY       = 0 # [SHARED: Pipeline]

# ─────────────────────────────────────────────────────────────────────────────
# 7. CORE PHYSICS & RISK
# ─────────────────────────────────────────────────────────────────────────────
NATR_BRICK_PERCENT       = 0.0015 # [SHARED: Pipeline, Core]
HEARTBEAT_INJECT_SEC      = 60.0 # [SHARED: Core]
ORDER_LOCK_TIMEOUT_SEC    = 30 # [SHARED: Core]
RENKO_HISTORY_LIMIT      = 100 # [SHARED: Core]
STRUCTURAL_REVERSAL_BRICKS = 6 # [SHARED: Pipeline, Core]
SOFT_VETO_THRESHOLD       = 0.9 # [SHARED: Pipeline, Core]
FEATURE_COLS = ["velocity", "momentum_acceleration", "feature_tib_zscore", "vwap_zscore", "feature_vpb_roc", "feature_cvd_divergence", "vpt_acceleration", "relative_strength", "fracdiff_price", "wick_pressure", "hurst", "consecutive_same_dir", "streak_exhaustion", "true_gap_pct", "regime_morning", "regime_midday", "regime_afternoon"] # [SHARED: Pipeline, Core]

# ─────────────────────────────────────────────────────────────────────────────
# 8. RISK MANAGEMENT & LATENCY
# ─────────────────────────────────────────────────────────────────────────────
STARTING_CAPITAL      = 20000 # [SHARED: Pipeline]
INTRADAY_LEVERAGE     = 5 # [SHARED: Pipeline]
POSITION_SIZE_PCT     = 0.10 # [SHARED: Pipeline]
STATE_WRITE_INTERVAL  = 0.5
CIRCUIT_BREAKER_STALE_SEC = 30.0
TARGET_CLIPPING_BPS      = 250.0

# ─────────────────────────────────────────────────────────────────────────────
# 9. SIMULATOR & FRICTION
# ─────────────────────────────────────────────────────────────────────────────
SIM_BROKERAGE_MAX    = 20.0 # [SHARED: Pipeline]
SIM_BROKERAGE_PCT    = 0.0005 # [SHARED: Pipeline]
SIM_STT_SELL_PCT     = 0.00025 # [SHARED: Pipeline]
SIM_STAMP_BUY_PCT    = 0.00003 # [SHARED: Pipeline]
SIM_EXCHANGE_PCT     = 0.0000297 # [SHARED: Pipeline]
SIM_SEBI_PCT         = 0.000001 # [SHARED: Pipeline]
SIM_GST_PCT          = 0.18 # [SHARED: Pipeline]
SIM_LEVERAGE         = 5.0
SIM_STARTING_CAPITAL = 100000.0
T1_SLIPPAGE_PCT      = 0.0005 # [SHARED: Pipeline]

# ─────────────────────────────────────────────────────────────────────────────
# 10. UI & DASHBOARD (Inherited from base for now, can be overridden)
# ─────────────────────────────────────────────────────────────────────────────
REGIME_WINDOW         = 40 # [SHARED: API]

def to_naive_ist(ts):
    import pandas as pd
    if ts is None: return None
    if hasattr(ts, "dt"):
        if ts.dt.tz is None: return ts
        return ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    ts_scalar = pd.to_datetime(ts)
    if ts_scalar.tz is None: return ts_scalar
    return ts_scalar.tz_convert("Asia/Kolkata").tz_localize(None)

