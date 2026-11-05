"""Unit tests for ActorCriticNetwork with action masking and checkpointing."""

import numpy as np
import pytest
import torch

from evanthea.rl.models import ActorCriticNetwork


def test_actor_critic_forward_and_masking():
    obs_dim = 32
    action_dim = 10
    net = ActorCriticNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=64)

    obs = torch.randn(4, obs_dim)
    mask = torch.ones(4, action_dim, dtype=torch.bool)
    # Mask out action 2 for all samples
    mask[:, 2] = False

    logits, values = net(obs, action_mask=mask)

    assert logits.shape == (4, action_dim)
    assert values.shape == (4,)
    # Masked action logit should be ~ -1e9
    assert torch.all(logits[:, 2] < -1e8)


def test_actor_critic_get_action():
    net = ActorCriticNetwork(obs_dim=16, action_dim=5, hidden_dim=32)
    obs = np.random.randn(16).astype(np.float32)
    mask = np.array([True, False, False, False, False])  # Only action 0 valid

    # Both stochastic and deterministic must choose action 0
    act_det, lp_det, val_det = net.get_action(obs, action_mask=mask, deterministic=True)
    assert act_det == 0
    assert isinstance(lp_det, float)
    assert isinstance(val_det, float)

    act_stoch, lp_stoch, val_stoch = net.get_action(obs, action_mask=mask, deterministic=False)
    assert act_stoch == 0


def test_actor_critic_evaluate_actions():
    net = ActorCriticNetwork(obs_dim=20, action_dim=6, hidden_dim=32)
    batch_size = 8
    obs = torch.randn(batch_size, 20)
    actions = torch.randint(0, 6, (batch_size,))
    mask = torch.ones(batch_size, 6, dtype=torch.bool)

    log_probs, entropy, values = net.evaluate_actions(obs, actions, action_mask=mask)

    assert log_probs.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert values.shape == (batch_size,)
    assert torch.all(entropy >= 0.0)


def test_actor_critic_checkpointing(tmp_path):
    save_path = tmp_path / "test_policy.pt"
    net = ActorCriticNetwork(obs_dim=12, action_dim=4, hidden_dim=32)
    net.save(save_path)

    loaded = ActorCriticNetwork.load(save_path)
    assert loaded.obs_dim == 12
    assert loaded.action_dim == 4
    assert loaded.hidden_dim == 32

    # Verify identical output on sample observation
    obs = torch.randn(2, 12)
    with torch.no_grad():
        out1, v1 = net(obs)
        out2, v2 = loaded(obs)
        assert torch.allclose(out1, out2)
        assert torch.allclose(v1, v2)
