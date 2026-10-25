"""Ridge regression fitness predictor with validation-based regularization parameter selection."""

from typing import Any, Dict, List, Optional
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, RidgeCV

from evanthea.predictors.base import FitnessPredictor


class RidgeFitnessPredictor(FitnessPredictor):
    """Linear fitness predictor with L2 regularization and validation-driven hyperparameter tuning."""

    def __init__(
        self,
        alphas: Optional[List[float]] = None,
        random_state: int = 42,
    ):
        super().__init__(name="RidgeFitnessPredictor")
        self.alphas = alphas or list(np.logspace(-4, 4, 17))
        self.random_state = random_state
        self.model: Optional[Ridge] = None
        self.best_alpha_: Optional[float] = None
        self.val_score_: Optional[float] = None
        self.coef_norm_: Optional[float] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "RidgeFitnessPredictor":
        """Fits Ridge model and selects optimal alpha."""
        X_tr = np.asarray(X_train, dtype=np.float32)
        y_tr = np.asarray(y_train, dtype=np.float32)

        if X_val is not None and y_val is not None:
            # Explicit validation-set grid search using Spearman correlation
            X_v = np.asarray(X_val, dtype=np.float32)
            y_v = np.asarray(y_val, dtype=np.float32)

            best_alpha = self.alphas[0]
            best_score = -float("inf")

            for alpha in self.alphas:
                candidate = Ridge(alpha=alpha, random_state=self.random_state)
                candidate.fit(X_tr, y_tr)
                preds = candidate.predict(X_v)
                corr, _ = spearmanr(preds, y_v)
                score = float(corr) if not np.isnan(corr) else -1.0

                if score > best_score:
                    best_score = score
                    best_alpha = alpha

            self.best_alpha_ = float(best_alpha)
            self.val_score_ = float(best_score)
            self.model = Ridge(alpha=self.best_alpha_, random_state=self.random_state)
            self.model.fit(X_tr, y_tr)

        else:
            # Fallback to internal K-Fold CV on training split
            rcv = RidgeCV(alphas=self.alphas)
            rcv.fit(X_tr, y_tr)
            self.best_alpha_ = float(rcv.alpha_)
            self.val_score_ = None
            self.model = Ridge(alpha=self.best_alpha_, random_state=self.random_state)
            self.model.fit(X_tr, y_tr)

        self.coef_norm_ = float(np.linalg.norm(self.model.coef_))
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("RidgeFitnessPredictor must be fitted before predict().")
        X_arr = np.asarray(X, dtype=np.float32)
        return self.model.predict(X_arr).astype(np.float32)

    def get_params(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "best_alpha": self.best_alpha_,
            "val_score": self.val_score_,
            "coef_norm": self.coef_norm_,
            "intercept": float(self.model.intercept_) if self.model else None,
            "is_fitted": self.is_fitted,
        }
