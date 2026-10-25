"""Baseline predictors: Trivial mean/median and unsupervised zero-shot ESM-2 scoring."""

from typing import Any, Dict, Optional
import numpy as np
from scipy.stats import spearmanr

from evanthea.predictors.base import FitnessPredictor


class MeanPredictor(FitnessPredictor):
    """Predicts the constant arithmetic mean of training target labels."""

    def __init__(self):
        super().__init__(name="MeanPredictor")
        self.mean_value_: Optional[float] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "MeanPredictor":
        y_arr = np.asarray(y_train, dtype=np.float64)
        self.mean_value_ = float(np.mean(y_arr))
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("MeanPredictor must be fitted before predict().")
        X_arr = np.asarray(X)
        return np.full(shape=(len(X_arr),), fill_value=self.mean_value_, dtype=np.float32)

    def get_params(self) -> Dict[str, Any]:
        return {"name": self.name, "mean_value": self.mean_value_, "is_fitted": self.is_fitted}


class MedianPredictor(FitnessPredictor):
    """Predicts the constant median of training target labels."""

    def __init__(self):
        super().__init__(name="MedianPredictor")
        self.median_value_: Optional[float] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "MedianPredictor":
        y_arr = np.asarray(y_train, dtype=np.float64)
        self.median_value_ = float(np.median(y_arr))
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("MedianPredictor must be fitted before predict().")
        X_arr = np.asarray(X)
        return np.full(shape=(len(X_arr),), fill_value=self.median_value_, dtype=np.float32)

    def get_params(self) -> Dict[str, Any]:
        return {"name": self.name, "median_value": self.median_value_, "is_fitted": self.is_fitted}


class ZeroShotESM2Predictor(FitnessPredictor):
    """Unsupervised zero-shot baseline using ESM-2 masked-marginal log-odds score (Δlog p).

    Strict Scientific Safeguard:
        - By default, uses the raw documented score directly without sign flipping.
        - If calibrate_sign_on_val=True, tests Spearman rank correlation on validation data only.
          If negative on validation, flips orientation and records sign_flipped_on_val=True.
          Test results are never inspected to choose score orientation.
    """

    def __init__(
        self,
        score_column_idx: int = 1,
        calibrate_sign_on_val: bool = False,
    ):
        super().__init__(name="ZeroShotESM2Predictor")
        self.score_column_idx = score_column_idx
        self.calibrate_sign_on_val = calibrate_sign_on_val
        self.sign_multiplier_: float = 1.0
        self.val_spearman_: Optional[float] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "ZeroShotESM2Predictor":
        """Checks validation calibration if enabled; otherwise no-op parameter fitting."""
        if self.calibrate_sign_on_val and X_val is not None and y_val is not None:
            raw_val_scores = X_val[:, self.score_column_idx]
            corr, _ = spearmanr(raw_val_scores, y_val)
            self.val_spearman_ = float(corr) if not np.isnan(corr) else 0.0
            if self.val_spearman_ < 0:
                self.sign_multiplier_ = -1.0
            else:
                self.sign_multiplier_ = 1.0
        else:
            self.sign_multiplier_ = 1.0

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("ZeroShotESM2Predictor must be fitted before predict().")
        X_arr = np.asarray(X, dtype=np.float32)
        raw_scores = X_arr[:, self.score_column_idx]
        return (raw_scores * self.sign_multiplier_).astype(np.float32)

    def get_params(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score_column_idx": self.score_column_idx,
            "calibrate_sign_on_val": self.calibrate_sign_on_val,
            "sign_multiplier": self.sign_multiplier_,
            "val_spearman": self.val_spearman_,
            "is_fitted": self.is_fitted,
        }
