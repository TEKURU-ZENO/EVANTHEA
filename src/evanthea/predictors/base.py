"""Abstract base class for all EVANTHEA fitness surrogate models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class FitnessPredictor(ABC):
    """Abstract interface for supervised and baseline fitness predictors."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "FitnessPredictor":
        """Fits the predictor on training data with optional validation-based model selection."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates continuous fitness predictions for candidate feature vectors."""
        raise NotImplementedError

    def fit_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Convenience method to fit on train/val and predict on test."""
        return self.fit(X_train, y_train, X_val=X_val, y_val=y_val).predict(X_test)

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Returns hyperparameters and fitted parameters for experiment provenance."""
        raise NotImplementedError
