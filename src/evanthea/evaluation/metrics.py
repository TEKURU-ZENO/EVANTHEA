"""Evaluation metrics suite: Spearman rho, Pearson r, calibration errors, Top-k enrichment, and bootstrap CIs."""

import math
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field
from scipy.stats import pearsonr, spearmanr


class MetricBundle(BaseModel):
    """Complete performance evaluation profile for a model on a designated test split."""
    spearman: Optional[float] = Field(None, description="Spearman rank correlation coefficient")
    spearman_ci: Optional[Tuple[float, float]] = Field(None, description="95% bootstrap confidence interval for Spearman")
    pearson: Optional[float] = Field(None, description="Pearson linear correlation coefficient")
    pearson_ci: Optional[Tuple[float, float]] = Field(None, description="95% bootstrap confidence interval for Pearson")
    mae: float = Field(..., description="Mean Absolute Error in original experimental units")
    rmse: float = Field(..., description="Root Mean Squared Error in original experimental units")
    r2: float = Field(..., description="Coefficient of determination (R^2)")
    top_k_enrichment: Optional[float] = Field(None, description="Enrichment of true top-10% in predicted top-10%")
    n_samples: int = Field(..., description="Number of evaluated test variants")


def compute_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    """Calculates Spearman rank correlation. Returns None if undefined (constant or n < 2)."""
    if len(y_true) < 2:
        return None
    if np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
        return None  # Undefined correlation for constant vectors
    res, _ = spearmanr(y_true, y_pred)
    return float(res) if not np.isnan(res) else None


def compute_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    """Calculates Pearson correlation. Returns None if undefined (constant or n < 2)."""
    if len(y_true) < 2:
        return None
    if np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
        return None
    res, _ = pearsonr(y_true, y_pred)
    return float(res) if not np.isnan(res) else None


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination R^2."""
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return 0.0  # Constant target
    ss_res = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - (ss_res / ss_tot))


def compute_top_k_enrichment(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    percentile: float = 0.10,
) -> Optional[float]:
    """Calculates enrichment of experimentally top-ranked variants within predicted top-ranked candidates.

    Procedure:
        1. Select predicted top-k variants where k = ceil(percentile * N).
        2. Select ground-truth top-k variants.
        3. Calculate overlap count |Pred_Top_k ∩ True_Top_k|.
        4. Random baseline expectation is k * (k / N).
        5. Enrichment = Overlap / Expectation.

    Returns:
        Enrichment ratio (1.0 = equal to random guessing, >1.0 = enriched, <1.0 = depleted).
        Returns None if N < 5 or insufficient sample size.
    """
    n = len(y_true)
    if n < 5:
        return None

    k = max(1, math.ceil(percentile * n))

    # Stable descending sort indices
    pred_top_indices = set(np.argsort(-y_pred, kind="stable")[:k])
    true_top_indices = set(np.argsort(-y_true, kind="stable")[:k])

    overlap = len(pred_top_indices.intersection(true_top_indices))
    expected_overlap = k * (k / n)

    if expected_overlap < 1e-12:
        return 1.0

    enrichment = overlap / expected_overlap
    return float(enrichment)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], Optional[float]],
    n_bootstraps: int = 500,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Optional[Tuple[float, float]]:
    """Calculates paired non-parametric bootstrap confidence interval over test variant indices."""
    n = len(y_true)
    if n < 5:
        return None

    rng = np.random.RandomState(seed)
    boot_estimates: List[float] = []

    for _ in range(n_bootstraps):
        boot_idx = rng.choice(n, size=n, replace=True)
        val = metric_fn(y_true[boot_idx], y_pred[boot_idx])
        if val is not None and not np.isnan(val):
            boot_estimates.append(val)

    if len(boot_estimates) < (n_bootstraps // 2):
        return None

    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.percentile(boot_estimates, alpha * 100))
    upper = float(np.percentile(boot_estimates, (1.0 - alpha) * 100))
    return lower, upper


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    compute_ci: bool = True,
    n_bootstraps: int = 200,
    seed: int = 42,
) -> MetricBundle:
    """Evaluates comprehensive ranking and regression calibration metrics."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    spearman = compute_spearman(y_t, y_p)
    pearson = compute_pearson(y_t, y_p)
    mae = compute_mae(y_t, y_p)
    rmse = compute_rmse(y_t, y_p)
    r2 = compute_r2(y_t, y_p)
    enrichment = compute_top_k_enrichment(y_t, y_p, percentile=0.10)

    spearman_ci = None
    pearson_ci = None

    if compute_ci and len(y_t) >= 10:
        spearman_ci = bootstrap_ci(y_t, y_p, compute_spearman, n_bootstraps=n_bootstraps, seed=seed)
        pearson_ci = bootstrap_ci(y_t, y_p, compute_pearson, n_bootstraps=n_bootstraps, seed=seed)

    return MetricBundle(
        spearman=spearman,
        spearman_ci=spearman_ci,
        pearson=pearson,
        pearson_ci=pearson_ci,
        mae=mae,
        rmse=rmse,
        r2=r2,
        top_k_enrichment=enrichment,
        n_samples=len(y_t),
    )
