"""
src/live/inference_engine.py - Model Inference Wrapper
=======================================================
Handles loading and prediction for Brain 1 (CNN) and Brain 2 (XGBoost).
STRICT LOGIC PRESERVATION from engine_main.py.
"""

import logging
import numpy as np
import xgboost as xgb
import keras
import torch
from trading_engine import config

logger = logging.getLogger(__name__)

class InferenceEngine:
    def __init__(self):
        self.brain1_long = None
        self.brain1_short = None
        self.brain2 = None
        self.scaler = None
        self.EXPECTED_FEATURES = config.FEATURE_COLS

    def load_models(self):
        """STRICT: Verbatim from engine_main.py"""
        self.brain1_long = keras.models.load_model(str(config.BRAIN1_CNN_LONG_PATH))
        self.brain1_short = keras.models.load_model(str(config.BRAIN1_CNN_SHORT_PATH))
        self.scaler = keras.utils.custom_object_scope({}) # placeholder if needed, but joblib used below
        import joblib
        self.scaler = joblib.load(str(config.BRAIN1_SCALER_PATH))
        
        # Use Booster for class-agnostic JSON loading
        self.brain2 = xgb.Booster()
        self.brain2.load_model(str(config.BRAIN2_MODEL_PATH))
        logger.info(f"Models loaded (Brain1: CNN, Brain2: JSON, Scaler: Robust)")
        return self.brain1_long, self.brain1_short, self.brain2, self.scaler

    def predict_brain1(self, bdf_features_2d):
        """
        STRICT: 3D sliding window + Scaling logic.
        Args:
            bdf_features_2d: DataFrame containing latest features for the symbol.
        """
        # Apply scaler to 2D features before 3D stacking
        feats_2d = bdf_features_2d[self.EXPECTED_FEATURES].tail(config.CNN_WINDOW_SIZE).fillna(0)
        scaled_2d = self.scaler.transform(feats_2d)
        feat_3d   = np.array([scaled_2d], dtype=np.float32)
        
        # Dual Brain Inference: LONG and SHORT
        p_long  = float(self.brain1_long.predict(feat_3d, verbose=0)[0])
        p_short = float(self.brain1_short.predict(feat_3d, verbose=0)[0])
        
        return p_long, p_short

    def predict_brain2(self, p_long, p_short, b1d, latest_row_dict):
        """
        STRICT: Brain 2 feature matrix construction and prediction.
        """
        b2_vals = []
        for f_name in config.BRAIN2_FEATURES:
            if f_name == "brain1_prob_long": b2_vals.append(p_long)
            elif f_name == "brain1_prob_short": b2_vals.append(p_short)
            elif f_name == "trade_direction": b2_vals.append(float(b1d))
            else: b2_vals.append(float(latest_row_dict.get(f_name, 0)))
        
        dm_meta = xgb.DMatrix([b2_vals], feature_names=config.BRAIN2_FEATURES)
        b2c = float(np.clip(self.brain2.predict(dm_meta)[0], 0, config.TARGET_CLIPPING_BPS))
        return b2c
