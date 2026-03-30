"""
src/live/inference_engine.py - Model Inference Wrapper
=======================================================
Handles loading and prediction for Brain 1 (CNN) and Brain 2 (XGBoost).
Supports LONG_ONLY_MODE and calibrated [0,1] probability from Brain 2.
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
        self.calibrator_long = None
        self.calibrator_short = None
        self.EXPECTED_FEATURES = config.FEATURE_COLS

    def load_models(self):
        """Load all models with LONG_ONLY_MODE support."""
        import joblib
        
        # Brain 1 LONG (always loaded)
        self.brain1_long = keras.models.load_model(str(config.BRAIN1_CNN_LONG_PATH))
        
        # Brain 1 SHORT (skip if LONG_ONLY_MODE)
        if not config.LONG_ONLY_MODE:
            self.brain1_short = keras.models.load_model(str(config.BRAIN1_CNN_SHORT_PATH))
        else:
            self.brain1_short = None
            logger.info("LONG_ONLY_MODE: Skipped loading Brain 1 SHORT model")
        
        # Scaler
        self.scaler = joblib.load(str(config.BRAIN1_SCALER_PATH))
        
        # Load calibrators if available
        if config.USE_CALIBRATED_MODELS:
            try:
                from trading_core.core.physics.quant_fixes import IsotonicCalibrationWrapper
                if config.BRAIN1_CALIBRATED_LONG_PATH.exists():
                    self.calibrator_long = IsotonicCalibrationWrapper.load(config.BRAIN1_CALIBRATED_LONG_PATH)
                if not config.LONG_ONLY_MODE and config.BRAIN1_CALIBRATED_SHORT_PATH.exists():
                    self.calibrator_short = IsotonicCalibrationWrapper.load(config.BRAIN1_CALIBRATED_SHORT_PATH)
            except Exception as e:
                logger.warning(f"Calibrator loading failed (using raw probabilities): {e}")
        
        # Brain 2 (XGBoost — Booster for class-agnostic JSON loading)
        self.brain2 = xgb.Booster()
        self.brain2.load_model(str(config.BRAIN2_MODEL_PATH))
        
        logger.info(f"Models loaded (Brain1: CNN, Brain2: Classifier, Scaler: Robust, LONG_ONLY={config.LONG_ONLY_MODE})")
        return self.brain1_long, self.brain1_short, self.brain2, self.scaler

    def predict_brain1(self, bdf_features_2d):
        """
        3D sliding window + Scaling logic.
        Returns (p_long, p_short). In LONG_ONLY_MODE, p_short is always 0.0.
        """
        feats_2d = bdf_features_2d[self.EXPECTED_FEATURES].tail(config.CNN_WINDOW_SIZE).fillna(0)
        scaled_2d = self.scaler.transform(feats_2d)
        feat_3d   = np.array([scaled_2d], dtype=np.float32)
        
        # LONG prediction
        p_long = float(self.brain1_long.predict(feat_3d, verbose=0)[0])
        if self.calibrator_long:
            p_long = self.calibrator_long.predict_single(p_long)
        
        # SHORT prediction (skip in LONG_ONLY_MODE)
        if config.LONG_ONLY_MODE or self.brain1_short is None:
            p_short = 0.0
        else:
            p_short = float(self.brain1_short.predict(feat_3d, verbose=0)[0])
            if self.calibrator_short:
                p_short = self.calibrator_short.predict_single(p_short)
        
        return p_long, p_short

    def predict_brain2(self, p_long, p_short, b1d, latest_row_dict):
        """
        Brain 2 conviction prediction.
        FIX: Output is now [0, 1] probability (was [0, 100] on old regressor).
        """
        b2_vals = []
        for f_name in config.BRAIN2_FEATURES:
            if f_name == "brain1_prob_long": b2_vals.append(p_long)
            elif f_name == "brain1_prob_short": b2_vals.append(p_short)
            elif f_name == "trade_direction": b2_vals.append(float(b1d))
            else: b2_vals.append(float(latest_row_dict.get(f_name, 0)))
        
        dm_meta = xgb.DMatrix([b2_vals], feature_names=config.BRAIN2_FEATURES)
        # FIX: Clip to [0, 1] — Brain 2 is now a classifier outputting probability
        b2c = float(np.clip(self.brain2.predict(dm_meta)[0], 0.0, 1.0))
        return b2c
