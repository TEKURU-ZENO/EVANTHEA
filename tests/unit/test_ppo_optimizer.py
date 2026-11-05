"""Unit tests for PPO training engine, rollout buffer, and policy evaluation."""

import numpy as np
import pytest

from evanthea.evolution.individual import Individual
from evanthea.evolution.objectives import MultiObjectiveEvaluator, Objective, ObjectiveDirection
from evanthea.rl.agent import RLAgent
from evanthea.rl.environment import ProteinDesignEnv
from evanthea.rl.evaluation import evaluate_policy
from evanthea.rl.models import ActorCriticNetwork
from evanthea.rl.ppo import PPOOptimizer, PPORolloutBuffer


@pytest.fixture
def test_env():
    wt = "MSKGEELFTG"
    evaluator = MultiObjectiveEvaluator(
        objectives=[
            Objective(name="predicted_fitness", direction=ObjectiveDirection.MAXIMIZE),
        ],
        evaluators={
            "predicted_fitness": lambda ind: float(ind.sequence.count("E") * 2.0),
        },
    )
    return ProteinDesignEnv(
        wildtype_sequence=wt,
        evaluator=evaluator,
        allowed_positions=[0, 1, 2],
        max_mutations=3,
        allow_stop_action=True,
        seed=42,
    )


def test_ppo_buffer_gae():
    buffer = PPORolloutBuffer(gamma=0.99, gae_lambda=0.95)
    for i in range(5):
        buffer.add(
            state=np.zeros(10, dtype=np.float32),
            action=i,
            reward=1.0,
            value=0.5,
            log_prob=-0.5,
            action_mask=np.ones(10, dtype=bool),
            done=(i == 4),
        )

    advantages, returns = buffer.compute_returns_and_advantages(last_value=0.0)
    assert len(advantages) == 5
    assert len(returns) == 5
    # Normalized advantages have zero mean
    assert pytest.approx(float(advantages.mean().item()), abs=1e-5) == 0.0


def test_ppo_optimizer_train_step(test_env):
    net = ActorCriticNetwork(obs_dim=test_env.obs_dim, action_dim=test_env.num_actions, hidden_dim=32)
    ppo = PPOOptimizer(network=net, lr=1e-3, n_epochs=2, batch_size=8)

    buffer = ppo.collect_rollouts(test_env, target_steps=16)
    assert len(buffer.states) >= 16

    metrics = ppo.train_step(buffer)
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert "approx_kl" in metrics
    assert metrics["samples_trained"] >= 16


def test_rl_agent_and_evaluation(test_env):
    net = ActorCriticNetwork(obs_dim=test_env.obs_dim, action_dim=test_env.num_actions, hidden_dim=32)
    agent = RLAgent(network=net, protein_id="TEST_PROT")

    # Rollout episode
    traj = agent.rollout_episode(test_env, deterministic=False)
    assert len(traj.steps) > 0
    assert len(traj.final_sequence) == len(test_env.wildtype_sequence)

    # Propose candidates
    candidates = agent.propose_candidates(test_env, n_candidates=2)
    assert len(candidates) == 2
    assert all(c.protein_id == "TEST_PROT" for c in candidates)

    # Policy evaluation
    summary = evaluate_policy(agent, test_env, n_episodes=3, deterministic=True)
    assert summary.episodes == 3
    assert summary.unique_variants >= 1
