"""Unit tests for TargetTransformer and leakage-free target normalization."""

import numpy as np
import pytest
from evanthea.predictors.targets import TargetTransformer


def test_target_transformer_identity():
    """Identity mode preserves values without alteration."""
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    transformer = TargetTransformer(mode="identity")
    transformer.fit(y_train)

    y_trans = transformer.transform(y_train)
    np.testing.assert_allclose(y_trans, y_train)

    y_inv = transformer.inverse_transform(y_trans)
    np.testing.assert_allclose(y_inv, y_train)


def test_target_transformer_standard():
    """Standard z-scoring uses training mean/std and inverts perfectly."""
    y_train = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    transformer = TargetTransformer(mode="standard")
    y_trans = transformer.fit_transform(y_train)

    # Expected: mean 30.0, std sqrt(200) ~ 14.1421
    assert pytest.approx(transformer.mean_) == 30.0
    assert pytest.approx(transformer.scale_) == np.std(y_train, ddof=0)
    assert pytest.approx(np.mean(y_trans)) == 0.0
    assert pytest.approx(np.std(y_trans, ddof=0)) == 1.0

    # Invert
    y_inv = transformer.inverse_transform(y_trans)
    np.testing.assert_allclose(y_inv, y_train)


def test_target_transformer_zero_test_leakage():
    """Parameters learned on y_train are strictly unaffected by y_test."""
    y_train = np.array([1.0, 2.0, 3.0])
    y_test = np.array([1000.0, 2000.0])

    transformer = TargetTransformer(mode="standard").fit(y_train)
    # y_test has huge values, but transformer's mean and scale must remain from y_train
    assert pytest.approx(transformer.mean_) == 2.0
    assert transformer.n_samples_seen_ == 3

    y_test_trans = transformer.transform(y_test)
    # Inversion must restore y_test exactly
    y_test_inv = transformer.inverse_transform(y_test_trans)
    np.testing.assert_allclose(y_test_inv, y_test)


def test_target_transformer_robust():
    """Robust mode scales using training median and IQR."""
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    transformer = TargetTransformer(mode="robust").fit(y_train)

    assert pytest.approx(transformer.mean_) == 3.0  # Median
    assert transformer.scale_ > 0.0

    y_inv = transformer.inverse_transform(transformer.transform(y_train))
    np.testing.assert_allclose(y_inv, y_train)


def test_target_transformer_error_handling():
    """Raises informative errors on non-finite, empty, or unfitted states."""
    transformer = TargetTransformer(mode="standard")
    with pytest.raises(RuntimeError):
        transformer.transform(np.array([1.0]))

    with pytest.raises(ValueError):
        transformer.fit(np.array([]))

    with pytest.raises(ValueError):
        transformer.fit(np.array([1.0, np.nan, 3.0]))

    with pytest.raises(ValueError):
        transformer.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
