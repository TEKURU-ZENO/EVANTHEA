"""Random Forest Regressor for ensemble fitness landscape prediction."""

from typing import Any, Dict, Optional
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from evanthea.predictors.base import FitnessPredictor


class RandomForestFitnessPredictor(FitnessPredictor):
    """Bagged ensemble predictor providing non-linear representations."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 8,
        min_samples_leaf: int = 2,
        random_state: int = 42,
    ):
        super().__init__(name="RandomForestFitnessPredictor")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model: Optional[RandomForestRegressor] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "RandomForestFitnessPredictor":
        X_tr = np.asarray(X_train, dtype=np.float32)
        y_tr = np.asarray(y_train, dtype=np.float32)

        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_tr, y_tr)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("RandomForestFitnessPredictor must be fitted before predict().")
        X_arr = np.asarray(X, dtype=np.float32)
        return self.model.predict(X_arr).astype(np.float32)

    def get_params(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
            "is_fitted": self.is_fitted,
        }
