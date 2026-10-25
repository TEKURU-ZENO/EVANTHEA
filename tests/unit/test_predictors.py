"""Unit tests for baseline and supervised fitness predictors."""

import numpy as np
import pytest
from evanthea.predictors.baselines import MeanPredictor, MedianPredictor, ZeroShotESM2Predictor
from evanthea.predictors.gradient_boosting import HistGradientBoostingPredictor
from evanthea.predictors.random_forest import RandomForestFitnessPredictor
from evanthea.predictors.ridge import RidgeFitnessPredictor


def test_mean_and_median_predictors():
    """Mean and median predictors produce expected constant outputs."""
    X_train = np.random.randn(20, 5).astype(np.float32)
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 10.0], dtype=np.float32)
    X_train = X_train[:5]

    mean_p = MeanPredictor().fit(X_train, y_train)
    preds = mean_p.predict(np.zeros((3, 5)))
    assert preds.shape == (3,)
    assert pytest.approx(preds[0]) == 4.0  # (1+2+3+4+10)/5 = 4.0

    med_p = MedianPredictor().fit(X_train, y_train)
    med_preds = med_p.predict(np.zeros((3, 5)))
    assert pytest.approx(med_preds[0]) == 3.0  # Median is 3.0


def test_zero_shot_esm2_predictor():
    """Zero-shot scorer passes through raw score and optionally calibrates sign on val."""
    # Feature matrix where col 1 is raw masked marginal score
    X = np.array([[0.0, -1.5], [0.0, -0.5], [0.0, 0.2]], dtype=np.float32)
    y = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    # 1. Uncalibrated (raw score direct)
    p_raw = ZeroShotESM2Predictor(score_column_idx=1, calibrate_sign_on_val=False).fit(X, y)
    preds = p_raw.predict(X)
    np.testing.assert_allclose(preds, [-1.5, -0.5, 0.2])

    # 2. Calibrated on val: since raw scores (-1.5, -0.5, 0.2) correlate positively with y (1, 2, 3), sign remains +1
    p_cal = ZeroShotESM2Predictor(score_column_idx=1, calibrate_sign_on_val=True).fit(X, y, X_val=X, y_val=y)
    assert p_cal.sign_multiplier_ == 1.0


def test_ridge_fitness_predictor():
    """Ridge predictor fits synthetic linear data and tunes alpha on validation set."""
    np.random.seed(42)
    X_tr = np.random.randn(50, 8).astype(np.float32)
    true_weights = np.array([1.5, -2.0, 0.5, 0.0, 1.0, -0.5, 0.2, -1.0], dtype=np.float32)
    y_tr = X_tr @ true_weights + np.random.normal(0, 0.1, size=50).astype(np.float32)

    X_v = np.random.randn(20, 8).astype(np.float32)
    y_v = X_v @ true_weights

    ridge = RidgeFitnessPredictor(random_state=42).fit(X_tr, y_tr, X_val=X_v, y_val=y_v)
    assert ridge.is_fitted
    assert ridge.best_alpha_ is not None

    preds = ridge.predict(X_v)
    corr = np.corrcoef(preds, y_v)[0, 1]
    assert corr > 0.95  # Strong recovery of linear signal


def test_hist_gradient_boosting_predictor():
    """HistGradientBoosting predictor trains on non-linear data and produces finite predictions."""
    np.random.seed(42)
    X_tr = np.random.randn(60, 4).astype(np.float32)
    y_tr = (X_tr[:, 0] ** 2 + np.sin(X_tr[:, 1])).astype(np.float32)

    X_v = np.random.randn(20, 4).astype(np.float32)
    y_v = (X_v[:, 0] ** 2 + np.sin(X_v[:, 1])).astype(np.float32)

    gbdt = HistGradientBoostingPredictor(random_state=42).fit(X_tr, y_tr, X_val=X_v, y_val=y_v)
    assert gbdt.is_fitted

    preds = gbdt.predict(X_v)
    assert len(preds) == 20
    assert np.all(np.isfinite(preds))


def test_random_forest_predictor():
    """RandomForest predictor fits and predicts accurately."""
    np.random.seed(42)
    X_tr = np.random.randn(30, 4).astype(np.float32)
    y_tr = X_tr[:, 0] * 2.0

    rf = RandomForestFitnessPredictor(n_estimators=10, max_depth=4, random_state=42).fit(X_tr, y_tr)
    assert rf.is_fitted
    preds = rf.predict(X_tr)
    assert len(preds) == 30
    assert np.all(np.isfinite(preds))
