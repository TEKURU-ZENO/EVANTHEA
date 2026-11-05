"""Integration test for end-to-end Reinforcement Learning training, memory caching, and variant generation."""

import numpy as np
import pytest

from evanthea.evolution.individual import Individual
from evanthea.evolution.objectives import MultiObjectiveEvaluator, Objective, ObjectiveDirection
from evanthea.memory.manager import EvolutionaryMemory
from evanthea.memory.storage import PersistentMemoryStore
from evanthea.rl.agent import RLAgent
from evanthea.rl.environment import ProteinDesignEnv
from evanthea.rl.evaluation import evaluate_policy
from evanthea.rl.models import ActorCriticNetwork
from evanthea.rl.ppo import PPOOptimizer


def test_end_to_end_rl_training_and_generation(tmp_path):
    wt = "MSKGEELFTG"
    eval_call_count = 0

    def mock_fitness(ind: Individual) -> float:
        nonlocal eval_call_count
        eval_call_count += 1
        return float(ind.sequence.count("E") * 2.5 + ind.sequence.count("D") * 1.5)

    def mock_esm(ind: Individual) -> float:
        return float(ind.sequence.count("G") * 0.4)

    evaluator = MultiObjectiveEvaluator(
        objectives=[
            Objective(name="predicted_fitness", direction=ObjectiveDirection.MAXIMIZE),
            Objective(name="esm2_plausibility", direction=ObjectiveDirection.MAXIMIZE),
        ],
        evaluators={
            "predicted_fitness": mock_fitness,
            "esm2_plausibility": mock_esm,
        },
    )

    db_path = tmp_path / "rl_mem.db"
    store = PersistentMemoryStore(db_path=db_path)
    memory = EvolutionaryMemory(store=store)

    env = ProteinDesignEnv(
        wildtype_sequence=wt,
        evaluator=evaluator,
        memory=memory,
        allowed_positions=[0, 1, 2, 3],
        max_mutations=3,
        allow_stop_action=True,
        seed=42,
    )

    net = ActorCriticNetwork(
        obs_dim=env.obs_dim,
        action_dim=env.num_actions,
        hidden_dim=32,
    )
    ppo = PPOOptimizer(network=net, lr=1e-3, n_epochs=2, batch_size=8)

    # 1. Train for 2 iterations of rollouts
    total_samples = 0
    for iteration in range(2):
        buffer = ppo.collect_rollouts(env, target_steps=16)
        metrics = ppo.train_step(buffer)
        total_samples += metrics["samples_trained"]
        assert not np.isnan(metrics["policy_loss"])
        assert not np.isnan(metrics["value_loss"])

    assert total_samples >= 32
    assert env.evaluations_count <= eval_call_count

    # 2. Deploy RLAgent for candidate proposal
    agent = RLAgent(network=net, protein_id="GFP_FRAGMENT", deterministic=False)
    candidates = agent.propose_candidates(env, n_candidates=4)

    assert len(candidates) == 4
    for c in candidates:
        assert c.protein_id == "GFP_FRAGMENT"
        assert len(c.mutated_sequence) == len(wt)
        assert "origin" in c.metadata
        assert c.metadata["origin"] == "rl_agent"

    # 3. Policy evaluation summary
    summary = evaluate_policy(agent, env, n_episodes=3, deterministic=True)
    assert summary.episodes == 3
    assert summary.best_fitness >= 0.0

    # 4. Evolutionary memory stats
    stats = memory.stats
    assert stats.unique_sequences > 0
