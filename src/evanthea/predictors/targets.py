"""Leakage-free target transformation and normalization for supervised fitness prediction."""

from typing import Any, Dict, Literal, Optional
import numpy as np


class TargetTransformer:
    """Transforms experimental target scores ensuring zero information leakage from test/validation sets.

    Supported modes:
        - "identity": No transformation (default, recommended initial baseline).
        - "standard": Z-score standardization using training mean and standard deviation.
        - "robust": Median/IQR scaling using training median and interquartile range.
    """

    def __init__(self, mode: Literal["identity", "standard", "robust"] = "identity"):
        if mode not in {"identity", "standard", "robust"}:
            raise ValueError(f"Unsupported mode '{mode}'. Choose from 'identity', 'standard', 'robust'.")
        self.mode = mode
        self.is_fitted = False
        self.mean_: Optional[float] = None
        self.scale_: Optional[float] = None
        self.n_samples_seen_: int = 0

    def fit(self, y: np.ndarray) -> "TargetTransformer":
        """Fits normalization parameters strictly on training target values.

        Args:
            y: 1D array of experimental target measurements from training split only.

        Returns:
            Fitted TargetTransformer instance.
        """
        arr = np.asarray(y, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D target array, got shape {arr.shape}.")
        if len(arr) == 0:
            raise ValueError("Target array cannot be empty.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Target array contains non-finite values (NaN or Inf).")

        self.n_samples_seen_ = len(arr)

        if self.mode == "identity":
            self.mean_ = 0.0
            self.scale_ = 1.0

        elif self.mode == "standard":
            self.mean_ = float(np.mean(arr))
            std = float(np.std(arr, ddof=0))
            # Guard against zero or near-zero variance
            self.scale_ = std if std > 1e-12 else 1.0

        elif self.mode == "robust":
            self.mean_ = float(np.median(arr))
            q75, q25 = np.percentile(arr, [75, 25])
            iqr = float(q75 - q25)
            self.scale_ = iqr if iqr > 1e-12 else 1.0

        self.is_fitted = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Transforms target values using parameters learned during fit."""
        if not self.is_fitted:
            raise RuntimeError("TargetTransformer must be fitted before calling transform().")

        arr = np.asarray(y, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D target array, got shape {arr.shape}.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Target array to transform contains non-finite values (NaN or Inf).")

        if self.mode == "identity":
            return arr.copy()

        return (arr - self.mean_) / self.scale_

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """Fits transformer on y and returns transformed values."""
        return self.fit(y).transform(y)

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        """Restores scaled targets or predictions back to original experimental units."""
        if not self.is_fitted:
            raise RuntimeError("TargetTransformer must be fitted before calling inverse_transform().")

        arr = np.asarray(y_scaled, dtype=np.float64)
        if self.mode == "identity":
            return arr.copy()

        return (arr * self.scale_) + self.mean_

    def get_params(self) -> Dict[str, Any]:
        """Returns fitted parameters for provenance tracking and model registry."""
        return {
            "mode": self.mode,
            "is_fitted": self.is_fitted,
            "mean": self.mean_,
            "scale": self.scale_,
            "n_samples_seen": self.n_samples_seen_,
        }
