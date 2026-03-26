"""
src/live/inference.py - Phase 4: Dual-Brain Live Inference Engine
================================================================
Orchestrates the 1D-CNN (Direction) and XGBoost (Conviction) pipeline.
Synchronized with config.FEATURE_COLS (17 features).
"""

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import keras
import logging
from pathlib import Path

from trading_engine import config
from trading_core.core.features import compute_features_live

logger = logging.getLogger(__name__)

class DualBrainInferenceEngine:
    """
    Surgical Fix: Dual-Brain Inference.
    Loads production models and scales/shapes live data for prediction.
    """
    def __init__(self):
        logger.info("Initializing Dual-Brain Inference Engine...")
        
        # 1. Load Robust Scaler
        if not config.BRAIN1_SCALER_PATH.exists():
            raise FileNotFoundError(f"Scaler missing at {config.BRAIN1_SCALER_PATH}")
        self.scaler = joblib.load(config.BRAIN1_SCALER_PATH)
        
        # 2. Load Brain 1 CNNs (Long & Short)
        logger.info("Loading Brain 1 CNNs...")
        self.b1_long = keras.models.load_model(str(config.BRAIN1_CNN_LONG_PATH))
        self.b1_short = keras.models.load_model(str(config.BRAIN1_CNN_SHORT_PATH))
        
        # 3. Load Brain 2 Meta-Regressor (XGBoost)
        logger.info("Loading Brain 2 Meta-Regressor...")
        self.b2 = xgb.Booster()
        self.b2.load_model(str(config.BRAIN2_MODEL_PATH))
        
        self.feature_cols = config.FEATURE_COLS
        self.window_size  = config.CNN_WINDOW_SIZE
        
    def predict_conviction(self, bricks_df: pd.DataFrame, sector_df: pd.DataFrame = None) -> tuple[float, float, float, int]:
        """
        Runs the full inference pipeline.
        Returns: (conviction_score, prob_long, prob_short, trade_direction)
        """
        if len(bricks_df) < self.window_size:
            return 0.0, 0.0, 0.0, 0
            
        # 1. Compute Features (In-Memory Delta)
        # We compute on the last W+N rows to ensure rolling windows are warm
        lookback = max(self.window_size, 30)
        recent_df = bricks_df.tail(lookback).copy()
        
        feat_df = compute_features_live(recent_df, sector_df)
        
        # 2. Extract 3D Window for CNN
        # Shape: (1, 15, 17)
        win_df = feat_df.tail(self.window_size)
        if len(win_df) < self.window_size:
            return 0.0, 0.0, 0.0, 0
            
        # Scaler expects 2D
        scaled_2d = self.scaler.transform(win_df[self.feature_cols].fillna(0))
        
        # Reshape to 3D Tensor
        X_3d = np.array([scaled_2d], dtype=np.float32)
        
        # 3. Predict Brain 1 (Direction)
        p_long  = float(self.b1_long.predict(X_3d, verbose=0)[0])
        p_short = float(self.b1_short.predict(X_3d, verbose=0)[0])
        
        # Determine internal direction (Categorical)
        # 1=LONG, 2=SHORT, 0=NONE
        b1_dir = 0
        if p_long >= 0.5 or p_short >= 0.5:
            b1_dir = 1 if p_long >= p_short else 2
            
        # 4. Predict Brain 2 (Conviction)
        # Build meta-feature vector matching training: [prob_long, prob_short, trade_direction, ...config.BRAIN2_FEATURES]
        latest = win_df.iloc[-1]
        
        meta_vals = []
        for feat in config.BRAIN2_FEATURES:
            if feat == "brain1_prob_long": meta_vals.append(p_long)
            elif feat == "brain1_prob_short": meta_vals.append(p_short)
            elif feat == "trade_direction": meta_vals.append(float(b1_dir))
            else:
                meta_vals.append(float(latest.get(feat, 0)))
                
        # XGBoost Inference
        dm = xgb.DMatrix([meta_vals], feature_names=config.BRAIN2_FEATURES)
        conviction = float(self.b2.predict(dm)[0])
        
        # Optional: Clip and floor
        conviction = max(0.0, min(conviction, config.TARGET_CLIPPING_BPS))
        
        return conviction, p_long, p_short, b1_dir

