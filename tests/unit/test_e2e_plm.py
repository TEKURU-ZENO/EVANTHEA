"""Unit tests for the end-to-end PLM LoRA fine-tuning predictor."""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

from evanthea.predictors.e2e_plm import E2ELoRALinear, EndToEndPLMPredictor


def test_e2e_lora_linear():
    """Verifies that E2ELoRALinear wraps nn.Linear, preserves frozen base weights, and computes gradients."""
    base = nn.Linear(32, 64)
    lora = E2ELoRALinear(base, r=4, alpha=8.0, dropout=0.0)

    # Base weights must be frozen
    assert not base.weight.requires_grad
    if base.bias is not None:
        assert not base.bias.requires_grad

    # LoRA parameters must be trainable
    assert lora.lora_A.requires_grad
    assert lora.lora_B.requires_grad

    # Forward pass
    x = torch.randn(8, 32)
    out = lora(x)
    assert out.shape == (8, 64)

    # Backward pass
    loss = out.sum()
    loss.backward()

    assert lora.lora_A.grad is not None
    assert lora.lora_B.grad is not None
    assert base.weight.grad is None


def test_e2e_plm_init_and_parameter_isolation():
    """Verifies that EndToEndPLMPredictor injects LoRA into attention projections while freezing the base model."""
    predictor = EndToEndPLMPredictor(
        model_name="facebook/esm2_t6_8M_UR50D",
        lora_r=4,
        lora_alpha=8.0,
        target_modules=["query", "value"],
        device="cpu",
    )
    predictor._init_model()

    assert predictor.model is not None
    assert predictor.tokenizer is not None

    base_esm = predictor.model.base_model
    trainable_params = [p for p in predictor.model.parameters() if p.requires_grad]
    frozen_params = [p for p in predictor.model.parameters() if not p.requires_grad]

    assert len(trainable_params) > 0
    assert len(frozen_params) > 0

    # Ensure all base transformer parameters (excluding LoRA adapters) are frozen
    for name, param in base_esm.named_parameters():
        if "lora_" not in name:
            assert not param.requires_grad, f"Base parameter {name} was not frozen!"
        else:
            assert param.requires_grad, f"LoRA parameter {name} was not set to trainable!"


def test_e2e_plm_fit_and_predict():
    """Verifies full fit and predict pipeline with sequences and continuous fitness labels."""
    seqs = [
        "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTT",
        "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTK",
        "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTE",
        "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTA",
    ]
    targets = np.array([1.5, 0.2, -0.8, -1.5], dtype=np.float64)

    predictor = EndToEndPLMPredictor(
        model_name="facebook/esm2_t6_8M_UR50D",
        lora_r=4,
        lora_alpha=8.0,
        batch_size=2,
        max_epochs=2,
        device="cpu",
    )

    predictor.fit(seqs, targets, X_val=seqs[:2], y_val=targets[:2])
    assert predictor.is_fitted

    preds = predictor.predict(seqs)
    assert len(preds) == len(seqs)
    assert np.all(np.isfinite(preds))


def test_e2e_plm_serialization_roundtrip():
    """Verifies saving and loading of trained LoRA checkpoints."""
    seqs = [
        "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTT",
        "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTK",
    ]
    targets = np.array([1.0, 0.0], dtype=np.float64)

    predictor = EndToEndPLMPredictor(
        model_name="facebook/esm2_t6_8M_UR50D",
        lora_r=4,
        max_epochs=1,
        device="cpu",
    )
    predictor.fit(seqs, targets)

    preds_before = predictor.predict(seqs)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "e2e_lora.pt"
        predictor.save(ckpt_path)
        assert ckpt_path.exists()

        new_predictor = EndToEndPLMPredictor(device="cpu")
        new_predictor.load(ckpt_path)
        assert new_predictor.is_fitted

        preds_after = new_predictor.predict(seqs)
        np.testing.assert_allclose(preds_before, preds_after, atol=1e-5)
