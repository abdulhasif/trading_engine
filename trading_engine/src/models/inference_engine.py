"""
src/live/inference_engine.py - Model Inference Wrapper
=======================================================
Handles loading and prediction for Brain 1 (CNN) and Brain 2 (XGBoost).
STRICT LOGIC PRESERVATION from engine_main.py.
"""

import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import keras
import torch
from trading_core.core.config import base_config as config

logger = logging.getLogger(__name__)

class InferenceEngine:
    def __init__(self):
        self.brain1_long = None
        self.brain1_short = None
        self.brain2 = None
        self.scaler = None
        self.EXPECTED_FEATURES = config.FEATURE_COLS

    def load_models(self):
        """
        STRICT: Verbatim from engine_main.py + Phase 3: Calibrated Loader.
        Attempts to load calibrated Isotonic wrappers only if enabled in config.
        """
        from trading_core.core.physics.quant_fixes import IsotonicCalibrationWrapper
        import joblib

        # 1. Load Brain 1 (Directional)
        try:
            # ONLY load calibrated if toggle is ON
            if config.USE_CALIBRATED_MODELS and config.BRAIN1_CALIBRATED_LONG_PATH.exists():
                self.brain1_long = IsotonicCalibrationWrapper.load(config.BRAIN1_CALIBRATED_LONG_PATH)
                logger.info("Loaded CALIBRATED Brain1 LONG model.")
            else:
                self.brain1_long = keras.models.load_model(str(config.BRAIN1_CNN_LONG_PATH))
                logger.info("Loaded RAW Brain1 LONG model.")
        except Exception as e:
            logger.error(f"Failed to load Brain1 LONG: {e}")
            self.brain1_long = keras.models.load_model(str(config.BRAIN1_CNN_LONG_PATH))

        try:
            if config.USE_CALIBRATED_MODELS and config.BRAIN1_CALIBRATED_SHORT_PATH.exists():
                self.brain1_short = IsotonicCalibrationWrapper.load(config.BRAIN1_CALIBRATED_SHORT_PATH)
                logger.info("Loaded CALIBRATED Brain1 SHORT model.")
            else:
                self.brain1_short = keras.models.load_model(str(config.BRAIN1_CNN_SHORT_PATH))
                logger.info("Loaded RAW Brain1 SHORT model.")
        except Exception as e:
            logger.error(f"Failed to load Brain1 SHORT: {e}")
            self.brain1_short = keras.models.load_model(str(config.BRAIN1_CNN_SHORT_PATH))

        # 2. Load Scaler
        self.scaler = joblib.load(str(config.BRAIN1_SCALER_PATH))
        
        # 3. Load Brain 2 (Conviction Meta-Regressor)
        self.brain2 = xgb.Booster()
        self.brain2.load_model(str(config.BRAIN2_MODEL_PATH))
        
        logger.info(f"Inference Engine Ready (Brain1: Dual-Head, Brain2: Meta-Booster)")
        return self.brain1_long, self.brain1_short, self.brain2, self.scaler

    def predict_brain1(self, bdf_features_2d):
        """
        STRICT: 3D sliding window + Scaling logic.
        Handles both raw Keras models and IsotonicCalibrationWrapper.
        """
        # Apply scaler to 2D features before 3D stacking
        feats_2d = bdf_features_2d[self.EXPECTED_FEATURES].tail(config.CNN_WINDOW_SIZE).fillna(0)
        scaled_2d = self.scaler.transform(feats_2d)
        feat_3d   = np.array([scaled_2d], dtype=np.float32)
        
        # 1. LONG Prediction
        if hasattr(self.brain1_long, "_calibrator"):
            # Calibrated path: Keras -> Isotonic
            # The Wrapper's base_estimator is a KerasClassifierWrapper, which has .model
            m = self.brain1_long._base_estimator.model
            raw_p = float(m.predict(feat_3d, verbose=0)[0][0])
            p_long = float(self.brain1_long._calibrator.transform([raw_p])[0])
        else:
            # Raw path
            p_long = float(self.brain1_long.predict(feat_3d, verbose=0)[0][0])

        # 2. SHORT Prediction
        if hasattr(self.brain1_short, "_calibrator"):
            m = self.brain1_short._base_estimator.model
            raw_p = float(m.predict(feat_3d, verbose=0)[0][0])
            p_short = float(self.brain1_short._calibrator.transform([raw_p])[0])
        else:
            p_short = float(self.brain1_short.predict(feat_3d, verbose=0)[0][0])
        
        return p_long, p_short

    def predict_brain2(self, p_long, p_short, b1d, latest_row_dict):
        """
        STRICT: Brain 2 feature matrix construction and prediction.
        PhD Audit Fix: Technical features MUST be scaled for parity with the backtester.
        """
        # 1. Scale technical features from latest_row_dict
        # latest_row_dict contains raw technical indicators (e.g. relative_strength=20.0)
        # We need a 2D DataFrame for the scaler
        raw_tech_df = pd.DataFrame([latest_row_dict])[self.EXPECTED_FEATURES].fillna(0)
        scaled_tech_vals = self.scaler.transform(raw_tech_df)[0]
        scaled_tech_map = dict(zip(self.EXPECTED_FEATURES, scaled_tech_vals))

        b2_vals = []
        for f_name in config.BRAIN2_FEATURES:
            if f_name == "brain1_prob_long": 
                b2_vals.append(p_long)
            elif f_name == "brain1_prob_short": 
                b2_vals.append(p_short)
            elif f_name == "trade_direction": 
                b2_vals.append(float(b1d))
            elif f_name in self.EXPECTED_FEATURES:
                # Use the SCALED version of the technical indicator
                b2_vals.append(float(scaled_tech_map[f_name]))
            else: 
                # Meta-features that aren't technical indicators (e.g. true_gap_pct)
                b2_vals.append(float(latest_row_dict.get(f_name, 0)))
        
        dm_meta = xgb.DMatrix([b2_vals], feature_names=config.BRAIN2_FEATURES)
        b2_pred = self.brain2.predict(dm_meta)
        b2c = float(np.clip(b2_pred[0], 0, config.TARGET_CLIPPING_BPS))
        return b2c
