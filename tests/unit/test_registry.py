"""Unit tests for ModelRegistry saving, loading, and provenance tracking."""

import shutil
from pathlib import Path
import numpy as np
import pytest
from evanthea.evaluation.metrics import evaluate_predictions
from evanthea.predictors.registry import ModelRegistry
from evanthea.predictors.ridge import RidgeFitnessPredictor
from evanthea.predictors.targets import TargetTransformer

@pytest.fixture
def registry(tmp_path: Path):
    reg = ModelRegistry(tmp_path / "test_registry")
    yield reg


def test_model_registry_save_and_load(registry: ModelRegistry):
    """Saves model and scaler, then loads them and produces identical predictions."""
    X = np.random.randn(20, 4).astype(np.float32)
    y = (X[:, 0] * 3.0 + 1.0).astype(np.float32)

    # 1. TargetTransformer
    tf = TargetTransformer(mode="standard").fit(y)
    y_scaled = tf.transform(y)

    # 2. Fit Ridge
    ridge = RidgeFitnessPredictor().fit(X, y_scaled)
    preds_scaled = ridge.predict(X)
    preds_orig = tf.inverse_transform(preds_scaled)

    metrics = evaluate_predictions(y, preds_orig, compute_ci=False)

    # 3. Save to registry
    model_id = "ridge_test_v1"
    save_path = registry.save_model(
        model=ridge,
        target_transformer=tf,
        feature_config="seq_emb",
        metrics=metrics,
        model_id=model_id,
        extra_metadata={"notes": "unit test model"},
    )
    assert save_path.is_dir()
    assert (save_path / "model.joblib").is_file()
    assert (save_path / "target_transformer.json").is_file()
    assert (save_path / "metadata.json").is_file()

    # 4. List models
    all_models = registry.list_models()
    assert len(all_models) == 1
    assert all_models[0]["model_id"] == model_id

    # 5. Load model
    loaded_model, loaded_tf, meta = registry.load_model(model_id)
    assert meta["model_name"] == "RidgeFitnessPredictor"
    assert loaded_tf.mode == "standard"

    # 6. Verify predictions match exactly
    loaded_preds_scaled = loaded_model.predict(X)
    loaded_preds_orig = loaded_tf.inverse_transform(loaded_preds_scaled)
    np.testing.assert_allclose(loaded_preds_orig, preds_orig, rtol=1e-5)
