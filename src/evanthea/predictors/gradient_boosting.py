"""Histogram-based Gradient Boosted Trees for non-linear fitness landscape learning."""

from typing import Any, Dict, List, Optional
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from evanthea.predictors.base import FitnessPredictor


class HistGradientBoostingPredictor(FitnessPredictor):
    """Fast histogram-based Gradient Boosted Decision Trees with validation tuning."""

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iter: int = 100,
        max_depth: Optional[int] = 6,
        min_samples_leaf: int = 5,
        tune_on_val: bool = True,
        random_state: int = 42,
    ):
        super().__init__(name="HistGradientBoostingPredictor")
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tune_on_val = tune_on_val
        self.random_state = random_state

        self.model: Optional[HistGradientBoostingRegressor] = None
        self.best_params_: Dict[str, Any] = {}
        self.val_score_: Optional[float] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "HistGradientBoostingPredictor":
        X_tr = np.asarray(X_train, dtype=np.float32)
        y_tr = np.asarray(y_train, dtype=np.float32)

        if self.tune_on_val and X_val is not None and y_val is not None and len(X_tr) >= 10:
            X_v = np.asarray(X_val, dtype=np.float32)
            y_v = np.asarray(y_val, dtype=np.float32)

            # Predefined, bounded hyperparameter search grid (12 candidate configs)
            param_grid: List[Dict[str, Any]] = [
                {"lr": 0.02, "max_iter": 50, "max_depth": 4},
                {"lr": 0.05, "max_iter": 50, "max_depth": 4},
                {"lr": 0.05, "max_iter": 100, "max_depth": 4},
                {"lr": 0.05, "max_iter": 100, "max_depth": 6},
                {"lr": 0.1,  "max_iter": 50, "max_depth": 4},
                {"lr": 0.1,  "max_iter": 100, "max_depth": 6},
            ]

            best_score = -float("inf")
            best_config = param_grid[0]

            for p in param_grid:
                candidate = HistGradientBoostingRegressor(
                    learning_rate=p["lr"],
                    max_iter=p["max_iter"],
                    max_depth=p["max_depth"],
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state,
                )
                candidate.fit(X_tr, y_tr)
                preds = candidate.predict(X_v)
                corr, _ = spearmanr(preds, y_v)
                score = float(corr) if not np.isnan(corr) else -1.0

                if score > best_score:
                    best_score = score
                    best_config = p

            self.best_params_ = best_config
            self.val_score_ = float(best_score)
            self.model = HistGradientBoostingRegressor(
                learning_rate=best_config["lr"],
                max_iter=best_config["max_iter"],
                max_depth=best_config["max_depth"],
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
            self.model.fit(X_tr, y_tr)

        else:
            self.best_params_ = {
                "lr": self.learning_rate,
                "max_iter": self.max_iter,
                "max_depth": self.max_depth,
            }
            self.val_score_ = None
            self.model = HistGradientBoostingRegressor(
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
            self.model.fit(X_tr, y_tr)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("HistGradientBoostingPredictor must be fitted before predict().")
        X_arr = np.asarray(X, dtype=np.float32)
        return self.model.predict(X_arr).astype(np.float32)

    def get_params(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "best_params": self.best_params_,
            "val_score": self.val_score_,
            "random_state": self.random_state,
            "is_fitted": self.is_fitted,
        }
