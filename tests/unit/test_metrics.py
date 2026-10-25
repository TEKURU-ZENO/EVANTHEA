"""Unit tests for evaluation metrics suite and top-k enrichment."""

import numpy as np
import pytest
from evanthea.evaluation.metrics import (
    bootstrap_ci,
    compute_mae,
    compute_pearson,
    compute_r2,
    compute_rmse,
    compute_spearman,
    compute_top_k_enrichment,
    evaluate_predictions,
)


def test_standard_metrics_exact_agreement():
    """Identical predictions achieve 0 error and 1.0 correlations."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert pytest.approx(compute_spearman(y, y)) == 1.0
    assert pytest.approx(compute_pearson(y, y)) == 1.0
    assert pytest.approx(compute_mae(y, y)) == 0.0
    assert pytest.approx(compute_rmse(y, y)) == 0.0
    assert pytest.approx(compute_r2(y, y)) == 1.0


def test_constant_predictions_return_none_for_correlations():
    """Constant vectors return None (undefined), not silent 0.0."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_const = np.array([2.5, 2.5, 2.5, 2.5, 2.5])

    assert compute_spearman(y_true, y_const) is None
    assert compute_pearson(y_true, y_const) is None
    assert compute_r2(y_true, y_const) <= 0.0


def test_top_k_enrichment_perfect_and_inverse():
    """Top-k enrichment evaluates true vs predicted high-fitness variants."""
    # 20 samples: 2 samples in top 10%
    y_true = np.arange(20, dtype=float)
    y_pred_perfect = np.arange(20, dtype=float)

    # Perfect prediction captures all top 10%
    enrichment_perf = compute_top_k_enrichment(y_true, y_pred_perfect, percentile=0.10)
    assert enrichment_perf is not None
    # Overlap is 2, expected is 2 * (2/20) = 0.2 -> enrichment = 2 / 0.2 = 10.0
    assert pytest.approx(enrichment_perf) == 10.0

    # Inverse prediction captures 0 top 10%
    y_pred_inverse = -np.arange(20, dtype=float)
    enrichment_inv = compute_top_k_enrichment(y_true, y_pred_inverse, percentile=0.10)
    assert enrichment_inv == 0.0


def test_bootstrap_ci():
    """Bootstrap CI brackets point estimate."""
    np.random.seed(42)
    y_true = np.random.randn(50)
    y_pred = y_true + np.random.normal(0, 0.2, size=50)

    point_spearman = compute_spearman(y_true, y_pred)
    assert point_spearman is not None
    ci = bootstrap_ci(y_true, y_pred, compute_spearman, n_bootstraps=100, seed=42)
    assert ci is not None
    lower, upper = ci
    assert lower <= point_spearman <= upper


def test_evaluate_predictions_bundle():
    """MetricBundle returns all requested properties cleanly."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y_pred = y_true + 0.1

    bundle = evaluate_predictions(y_true, y_pred, compute_ci=False)
    assert bundle.n_samples == 10
    assert pytest.approx(bundle.spearman) == 1.0
    assert pytest.approx(bundle.pearson) == 1.0
    assert pytest.approx(bundle.mae) == 0.1
    assert bundle.r2 > 0.95
