"""Residual stratification, mutation-degree error breakdown, and top-variant error diagnostics."""

from typing import Any, Dict, List
import numpy as np
from pydantic import BaseModel, Field

from evanthea.data.schemas import VariantRecord
from evanthea.evaluation.metrics import compute_mae, compute_rmse, compute_spearman


class StratifiedErrorMetric(BaseModel):
    group_name: str
    count: int
    mae: float
    rmse: float
    mean_residual: float  # bias (y_true - y_pred)
    spearman: float | None = None


class ErrorAnalysisSummary(BaseModel):
    overall_mae: float
    overall_rmse: float
    by_mutation_order: Dict[int, StratifiedErrorMetric] = Field(default_factory=dict)
    top_10_percent_true_mae: float
    largest_errors: List[Dict[str, Any]] = Field(default_factory=list)


def analyze_prediction_errors(
    variants: List[VariantRecord],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> ErrorAnalysisSummary:
    """Performs deep residual stratification across mutation counts and fitness ranges."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    residuals = y_t - y_p

    overall_mae = compute_mae(y_t, y_p)
    overall_rmse = compute_rmse(y_t, y_p)

    # 1. Stratify by mutation count k
    order_dict: Dict[int, List[int]] = {}
    for idx, v in enumerate(variants):
        k = v.num_mutations
        if k not in order_dict:
            order_dict[k] = []
        order_dict[k].append(idx)

    stratified_order: Dict[int, StratifiedErrorMetric] = {}
    for k, indices in sorted(order_dict.items()):
        sub_t = y_t[indices]
        sub_p = y_p[indices]
        sub_res = residuals[indices]
        stratified_order[k] = StratifiedErrorMetric(
            group_name=f"{k}-mutant",
            count=len(indices),
            mae=compute_mae(sub_t, sub_p),
            rmse=compute_rmse(sub_t, sub_p),
            mean_residual=float(np.mean(sub_res)),
            spearman=compute_spearman(sub_t, sub_p),
        )

    # 2. Performance on top 10% highest true fitness variants
    n = len(y_t)
    k_top = max(1, int(np.ceil(0.10 * n)))
    top_10_indices = np.argsort(-y_t)[:k_top]
    top_mae = compute_mae(y_t[top_10_indices], y_p[top_10_indices])

    # 3. Top 5 largest absolute errors
    largest_err_indices = np.argsort(-np.abs(residuals))[:5]
    largest_errors = []
    for idx in largest_err_indices:
        largest_errors.append(
            {
                "variant_id": variants[idx].variant_id,
                "mutations": variants[idx].mutation_string,
                "y_true": float(y_t[idx]),
                "y_pred": float(y_p[idx]),
                "error": float(residuals[idx]),
            }
        )

    return ErrorAnalysisSummary(
        overall_mae=overall_mae,
        overall_rmse=overall_rmse,
        by_mutation_order=stratified_order,
        top_10_percent_true_mae=top_mae,
        largest_errors=largest_errors,
    )
