"""Benchmarking orchestrator executing matched model evaluations across leakage-resistant splits."""

from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field

from evanthea.data.schemas import DatasetSplitManifest, ProteinRecord, VariantRecord
from evanthea.evaluation.metrics import MetricBundle, evaluate_predictions
from evanthea.predictors.base import FitnessPredictor
from evanthea.predictors.baselines import MeanPredictor, MedianPredictor, ZeroShotESM2Predictor
from evanthea.predictors.features import FeatureConfig, FeatureMatrixBuilder, FeatureScaler
from evanthea.predictors.gradient_boosting import HistGradientBoostingPredictor
from evanthea.predictors.random_forest import RandomForestFitnessPredictor
from evanthea.predictors.ridge import RidgeFitnessPredictor
from evanthea.predictors.targets import TargetTransformer


class ModelBenchmarkResult(BaseModel):
    model_name: str
    feature_config: str
    split_strategy: str
    target_scaling: str
    test_metrics: MetricBundle
    val_score: Optional[float] = None
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)


def run_split_benchmark(
    reference_protein: ProteinRecord,
    all_variants: List[VariantRecord],
    split_manifest: DatasetSplitManifest,
    feature_config: FeatureConfig = FeatureConfig.EMB_DIFF,
    target_scaling: str = "identity",
    provided_embeddings: Optional[Dict[str, np.ndarray]] = None,
    scale_features: bool = False,
    models_to_run: Optional[List[str]] = None,
) -> List[ModelBenchmarkResult]:
    """Runs a rigorously controlled benchmark across a specific dataset split manifest.

    Scientific Protocol:
        1. Subsets variants into Train, Val, Test according to manifest variant IDs.
        2. Fits TargetTransformer strictly on Train labels.
        3. Transforms Val and Test targets using Train-fitted transformer.
        4. Builds feature matrices X_train, X_val, X_test under the specified FeatureConfig.
        5. Fits each candidate model on X_train, y_train (with validation selection on X_val, y_val).
        6. Predicts on X_test, inverts predictions back to original units.
        7. Evaluates Spearman, Pearson, MAE, RMSE, R^2, and Top-k enrichment.
    """
    id_to_var = {v.variant_id: v for v in all_variants}

    train_vars = [id_to_var[vid] for vid in split_manifest.train_variant_ids if vid in id_to_var]
    val_vars = [id_to_var[vid] for vid in split_manifest.val_variant_ids if vid in id_to_var]
    test_vars = [id_to_var[vid] for vid in split_manifest.test_variant_ids if vid in id_to_var]

    if not train_vars or not test_vars:
        raise ValueError(f"Split '{split_manifest.split_strategy}' lacks sufficient train or test variants.")

    # 1. Target Extraction & Leakage-Free Scaling
    y_train_raw = np.array([v.raw_score for v in train_vars], dtype=np.float64)
    y_val_raw = np.array([v.raw_score for v in val_vars], dtype=np.float64) if val_vars else None
    y_test_raw = np.array([v.raw_score for v in test_vars], dtype=np.float64)

    target_tf = TargetTransformer(mode=target_scaling).fit(y_train_raw)
    y_train_scaled = target_tf.transform(y_train_raw)
    y_val_scaled = target_tf.transform(y_val_raw) if y_val_raw is not None else None

    # 2. Feature Matrix Construction
    wt_emb = None
    if provided_embeddings:
        from evanthea.protein.sequence import compute_sequence_hash
        wt_hash = compute_sequence_hash(reference_protein.wildtype_sequence)
        wt_emb = provided_embeddings.get(wt_hash)

    builder = FeatureMatrixBuilder(reference_protein=reference_protein, wt_embedding=wt_emb)

    X_train, _ = builder.build_matrix(train_vars, config=feature_config, provided_embeddings=provided_embeddings)
    X_val, _ = builder.build_matrix(val_vars, config=feature_config, provided_embeddings=provided_embeddings) if val_vars else (None, None)
    X_test, _ = builder.build_matrix(test_vars, config=feature_config, provided_embeddings=provided_embeddings)

    if scale_features:
        f_scaler = FeatureScaler().fit(X_train)
        X_train = f_scaler.transform(X_train)
        if X_val is not None:
            X_val = f_scaler.transform(X_val)
        X_test = f_scaler.transform(X_test)

    # 3. Model Suite Setup
    default_models = ["Mean", "Median", "ZeroShot", "Ridge", "HistGBDT", "RandomForest"]
    selected_models = models_to_run or default_models

    model_instances: List[FitnessPredictor] = []
    if "Mean" in selected_models:
        model_instances.append(MeanPredictor())
    if "Median" in selected_models:
        model_instances.append(MedianPredictor())
    if "ZeroShot" in selected_models:
        # ZeroShot uses score column from feature matrix (Config D/E) or raw scores
        # Default index 1 is masked marginal score in mutation_features / hybrid
        col_idx = 1 if feature_config in {FeatureConfig.MUTATION_FEATURES, FeatureConfig.HYBRID} else 0
        model_instances.append(ZeroShotESM2Predictor(score_column_idx=col_idx, calibrate_sign_on_val=False))
    if "Ridge" in selected_models:
        model_instances.append(RidgeFitnessPredictor())
    if "HistGBDT" in selected_models:
        model_instances.append(HistGradientBoostingPredictor())
    if "RandomForest" in selected_models:
        model_instances.append(RandomForestFitnessPredictor(n_estimators=50, max_depth=6))

    # 4. Evaluation Loop
    results: List[ModelBenchmarkResult] = []

    for model in model_instances:
        model.fit(X_train, y_train_scaled, X_val=X_val, y_val=y_val_scaled)
        preds_scaled = model.predict(X_test)

        # Invert predictions back to original experimental units
        preds_original = target_tf.inverse_transform(preds_scaled)

        # Evaluate performance on test set
        metrics = evaluate_predictions(y_test_raw, preds_original, compute_ci=True)

        params = model.get_params()
        val_score = params.get("val_score")

        results.append(
            ModelBenchmarkResult(
                model_name=model.name,
                feature_config=str(feature_config.value if hasattr(feature_config, "value") else feature_config),
                split_strategy=split_manifest.split_strategy,
                target_scaling=target_scaling,
                test_metrics=metrics,
                val_score=val_score,
                hyperparameters=params,
            )
        )

    return results
