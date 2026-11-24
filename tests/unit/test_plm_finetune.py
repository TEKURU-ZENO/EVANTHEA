"""Unit tests for PLMFineTuningPredictor, LoRALinear, and CompositeRankLoss."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from evanthea.predictors.base import FitnessPredictor
from evanthea.predictors.baselines import TrueZeroShotESM2Predictor
from evanthea.predictors.plm_finetune import (
    AttentiveDeepRegressionHead,
    CompositeRankLoss,
    LoRALinear,
    PLMFineTuningPredictor,
)


def test_lora_linear_parameters_and_gradients():
    """Validates that LoRA freezes base weights and trains low-rank matrices."""
    base_linear = nn.Linear(32, 64)
    lora = LoRALinear(base_linear, rank=4, alpha=8.0, dropout=0.0)

    # Base weights must be frozen
    assert not lora.weight.requires_grad
    if lora.bias is not None:
        assert not lora.bias.requires_grad

    # LoRA matrices must be trainable
    assert lora.lora_A.requires_grad
    assert lora.lora_B.requires_grad
    assert lora.lora_A.shape == (4, 32)
    assert lora.lora_B.shape == (64, 4)

    # Forward pass
    x = torch.randn(8, 32)
    out = lora(x)
    assert out.shape == (8, 64)

    # Backward pass
    loss = out.sum()
    loss.backward()
    assert lora.lora_A.grad is not None
    assert lora.lora_B.grad is not None
    assert lora.weight.grad is None


def test_lora_linear_rank_zero():
    """Validates that rank 0 behaves as a frozen standard linear layer."""
    base_linear = nn.Linear(16, 16)
    lora = LoRALinear(base_linear, rank=0)
    x = torch.randn(4, 16)
    out = lora(x)
    assert out.shape == (4, 16)
    assert lora.lora_A is None
    assert lora.lora_B is None


def test_attentive_deep_regression_head_forward():
    """Validates that the deep attentive regression head produces correct scalar outputs."""
    head = AttentiveDeepRegressionHead(input_dim=48, hidden_dim=64, dropout=0.0)
    x = torch.randn(10, 48)
    out = head(x)
    assert out.shape == (10,)

    loss = out.sum()
    loss.backward()
    # Check that gradients flow to all projection layers
    for param in head.parameters():
        assert param.grad is not None


def test_composite_rank_loss_behavior():
    """Validates that CompositeRankLoss penalizes rank inversions."""
    loss_fn = CompositeRankLoss(rank_weight=2.0, temperature=1.0)

    # Targets: monotonic ascending [1.0, 2.0, 3.0, 4.0]
    y_true = torch.tensor([1.0, 2.0, 3.0, 4.0])

    # Case 1: Perfectly aligned predictions
    y_pred_good = torch.tensor([1.1, 1.9, 3.1, 3.9])
    loss_good = loss_fn(y_pred_good, y_true).item()

    # Case 2: Inverted predictions
    y_pred_bad = torch.tensor([4.0, 3.0, 2.0, 1.0])
    loss_bad = loss_fn(y_pred_bad, y_true).item()

    assert loss_bad > loss_good, f"Inverted loss {loss_bad} should exceed concordant loss {loss_good}"


def test_plm_finetuning_predictor_fit_predict():
    """Validates end-to-end fit and predict cycles of PLMFineTuningPredictor."""
    np.random.seed(42)
    torch.manual_seed(42)

    # Synthetic dataset with 64-dim features
    N_train = 80
    N_val = 20
    N_test = 20
    D = 32

    X_train = np.random.randn(N_train, D).astype(np.float32)
    w_true = np.random.randn(D).astype(np.float32)
    y_train = (X_train @ w_true + 0.1 * np.random.randn(N_train)).astype(np.float32)

    X_val = np.random.randn(N_val, D).astype(np.float32)
    y_val = (X_val @ w_true + 0.1 * np.random.randn(N_val)).astype(np.float32)

    X_test = np.random.randn(N_test, D).astype(np.float32)
    y_test = (X_test @ w_true + 0.1 * np.random.randn(N_test)).astype(np.float32)

    predictor = PLMFineTuningPredictor(
        name="TestPLMFineTune",
        hidden_dim=32,
        batch_size=16,
        max_epochs=10,
        early_stopping_patience=3,
        rank_loss_weight=1.0,
        device="cpu",
    )

    # Predict before fit must raise RuntimeError
    with pytest.raises(RuntimeError):
        predictor.predict(X_test)

    # Fit predictor
    fitted = predictor.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    assert fitted.is_fitted
    assert fitted.model is not None

    # Predict on test set
    preds = predictor.predict(X_test)
    assert preds.shape == (N_test,)
    assert isinstance(preds, np.ndarray)
    assert not np.isnan(preds).any()

    # Provenance params
    params = predictor.get_params()
    assert params["name"] == "TestPLMFineTune"
    assert params["hidden_dim"] == 32
    assert params["epochs_trained"] <= 10
    assert params["val_score"] is not None


def test_true_zero_shot_esm2_predictor():
    """Validates TrueZeroShotESM2Predictor behavior."""
    scores = np.array([[1.5], [-0.5], [2.0], [0.1]])
    y = np.array([1.0, -1.0, 2.0, 0.0])

    predictor = TrueZeroShotESM2Predictor()
    predictor.fit(scores, y)
    preds = predictor.predict(scores)

    assert preds.shape == (4,)
    assert np.allclose(preds, [1.5, -0.5, 2.0, 0.1])


def test_plm_finetuning_predictor_registry_persistence(tmp_path):
    """Validates that PLMFineTuningPredictor can be serialized and reloaded via ModelRegistry."""
    from evanthea.evaluation.metrics import evaluate_predictions
    from evanthea.predictors.registry import ModelRegistry
    from evanthea.predictors.targets import TargetTransformer

    X_train = np.random.randn(30, 16).astype(np.float32)
    y_train = (X_train[:, 0] * 2.0).astype(np.float32)
    X_test = np.random.randn(10, 16).astype(np.float32)

    tf = TargetTransformer(mode="identity").fit(y_train)
    predictor = PLMFineTuningPredictor(hidden_dim=16, batch_size=8, max_epochs=3, device="cpu")
    predictor.fit(X_train, y_train)

    preds_orig = predictor.predict(X_test)
    metrics = evaluate_predictions(y_train[:10], preds_orig, compute_ci=False)

    registry = ModelRegistry(tmp_path / "plm_registry")
    model_id = "plm_surrogate_test"
    registry.save_model(
        model=predictor,
        target_transformer=tf,
        feature_config="hybrid",
        metrics=metrics,
        model_id=model_id,
    )

    loaded_model, loaded_tf, meta = registry.load_model(model_id)
    assert meta["model_name"] == "PLMFineTune"
    loaded_preds = loaded_model.predict(X_test)
    assert np.allclose(preds_orig, loaded_preds, atol=1e-5)

