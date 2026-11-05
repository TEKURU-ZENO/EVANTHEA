"""Unit tests for CompositeRewardFunction and multi-objective scalarization."""

import pytest

from evanthea.data.schemas import Mutation
from evanthea.evolution.individual import Individual
from evanthea.evolution.objectives import MultiObjectiveEvaluator, Objective, ObjectiveDirection
from evanthea.memory.manager import EvolutionaryMemory
from evanthea.memory.storage import PersistentMemoryStore
from evanthea.rl.rewards import CompositeRewardFunction, RewardWeights


@pytest.fixture
def test_evaluator():
    def mock_fitness(ind: Individual) -> float:
        return float(ind.sequence.count("E") * 3.0)

    def mock_esm(ind: Individual) -> float:
        return float(ind.sequence.count("G") * 0.5)

    return MultiObjectiveEvaluator(
        objectives=[
            Objective(name="predicted_fitness", direction=ObjectiveDirection.MAXIMIZE),
            Objective(name="esm2_plausibility", direction=ObjectiveDirection.MAXIMIZE),
        ],
        evaluators={
            "predicted_fitness": mock_fitness,
            "esm2_plausibility": mock_esm,
        },
    )


def test_composite_reward_progression(test_evaluator):
    reward_fn = CompositeRewardFunction(
        evaluator=test_evaluator,
        weights=RewardWeights(fitness_weight=1.0, esm_plausibility_weight=1.0, conservation_penalty=0.1),
    )

    # Mutation increases 'E' count: fitness goes from 0.0 to 3.0
    mut = Mutation(ref_residue="M", position_1based=1, mut_residue="E")
    tot_rew, breakdown = reward_fn.compute_reward(
        current_sequence="MAAA",
        next_sequence="EAAA",
        mutations=[mut],
        prev_fitness=0.0,
        step=1,
    )

    assert breakdown["delta_fitness"] == 3.0
    assert breakdown["current_fitness"] == 3.0
    # total reward should be ~ (1.0 * 3.0) - 0.1 = 2.9
    assert pytest.approx(tot_rew, 0.01) == 2.9


def test_composite_reward_violation_penalty(test_evaluator):
    reward_fn = CompositeRewardFunction(
        evaluator=test_evaluator,
        weights=RewardWeights(violation_penalty=5.0),
    )

    mut = Mutation(ref_residue="M", position_1based=1, mut_residue="C")
    tot_rew, breakdown = reward_fn.compute_reward(
        current_sequence="MAAA",
        next_sequence="CAAA",
        mutations=[mut],
        prev_fitness=0.0,
        step=1,
        is_feasible=False,
        violations=["Disallowed cysteine introduced"],
    )

    assert breakdown["violation_penalty"] == 5.0
    assert tot_rew < 0.0


def test_composite_reward_clipping(test_evaluator):
    reward_fn = CompositeRewardFunction(
        evaluator=test_evaluator,
        clip_reward=(-2.0, 2.0),
    )

    # Massive delta fitness
    mut = Mutation(ref_residue="M", position_1based=1, mut_residue="E")
    tot_rew, _ = reward_fn.compute_reward(
        current_sequence="MAAA",
        next_sequence="EEEEE",
        mutations=[mut],
        prev_fitness=-100.0,
        step=1,
    )

    assert tot_rew <= 2.0
