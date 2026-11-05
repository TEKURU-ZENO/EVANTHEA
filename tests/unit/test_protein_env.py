"""Unit tests for Gymnasium-compliant ProteinDesignEnv with action masking and constraints."""

import numpy as np
import pytest

from evanthea.evolution.individual import Individual
from evanthea.evolution.objectives import MultiObjectiveEvaluator, Objective, ObjectiveDirection
from evanthea.mutation.constraints import ConstraintPipeline, DisallowedResiduesConstraint
from evanthea.rl.environment import ProteinDesignEnv


@pytest.fixture
def mock_evaluator():
    """Simple evaluator counting amino acids for testing."""
    def eval_fit(ind: Individual) -> float:
        return float(ind.sequence.count("E") * 2.0 + ind.sequence.count("A"))

    def eval_plm(ind: Individual) -> float:
        return float(ind.sequence.count("G") * 0.5)

    return MultiObjectiveEvaluator(
        objectives=[
            Objective(name="predicted_fitness", direction=ObjectiveDirection.MAXIMIZE),
            Objective(name="esm2_plausibility", direction=ObjectiveDirection.MAXIMIZE),
        ],
        evaluators={
            "predicted_fitness": eval_fit,
            "esm2_plausibility": eval_plm,
        },
    )


def test_env_initialization_and_spaces(mock_evaluator):
    wt = "MSKGEELFTG"
    mutable_positions = [1, 2, 4]  # S, K, E

    env = ProteinDesignEnv(
        wildtype_sequence=wt,
        evaluator=mock_evaluator,
        allowed_positions=mutable_positions,
        max_mutations=3,
        allow_stop_action=True,
        seed=42,
    )

    # 3 positions * 20 amino acids + 1 STOP action = 61 actions
    assert env.action_space.n == 61
    assert env.num_actions == 61
    assert env.stop_action_id == 60

    # Observation space: 3 positions * 20 AA + 4 scalars = 64
    assert env.observation_space.shape == (64,)

    obs, info = env.reset(seed=42)
    assert obs.shape == (64,)
    assert info["sequence"] == wt
    assert info["current_fitness"] >= 0.0
    assert len(info["action_mask"]) == 61


def test_env_action_masking_and_step(mock_evaluator):
    wt = "MSKGEELFTG"
    mutable_positions = [1, 2]  # pos 1 is 'S', pos 2 is 'K'

    env = ProteinDesignEnv(
        wildtype_sequence=wt,
        evaluator=mock_evaluator,
        allowed_positions=mutable_positions,
        max_mutations=2,
        allow_revisits=False,
        allow_stop_action=True,
        seed=123,
    )

    obs, info = env.reset()
    mask = env.get_action_mask()

    # Identity self-mutations must be masked False
    # Find action for pos 1 -> 'S' (no-op)
    act_s_to_s = env.def_to_action[(1, "S")]
    assert mask[act_s_to_s] is False or mask[act_s_to_s] == 0

    # Valid mutation: pos 1 ('S') -> 'E'
    act_s_to_e = env.def_to_action[(1, "E")]
    assert mask[act_s_to_e] is True or mask[act_s_to_e] == 1

    next_obs, reward, term, trunc, step_info = env.step(act_s_to_e)
    assert not term
    assert not trunc
    assert step_info["sequence"][1] == "E"
    assert len(step_info["mutations"]) == 1

    # Since allow_revisits=False, all actions for pos 1 must now be masked False
    new_mask = env.get_action_mask()
    for aa in env.amino_acids:
        assert new_mask[env.def_to_action[(1, aa)]] is False or new_mask[env.def_to_action[(1, aa)]] == 0


def test_env_truncation_at_horizon(mock_evaluator):
    wt = "MSKGEELFTG"
    env = ProteinDesignEnv(
        wildtype_sequence=wt,
        evaluator=mock_evaluator,
        allowed_positions=[0, 1, 2],
        max_mutations=2,
        seed=42,
    )

    env.reset()
    mask1 = env.get_action_mask()
    valid_acts = np.where(mask1)[0]
    act1 = [a for a in valid_acts if a != env.stop_action_id][0]

    _, _, term1, trunc1, _ = env.step(act1)
    assert not term1 and not trunc1

    mask2 = env.get_action_mask()
    valid_acts2 = np.where(mask2)[0]
    act2 = [a for a in valid_acts2 if a != env.stop_action_id][0]

    _, _, term2, trunc2, _ = env.step(act2)
    assert trunc2 is True  # Truncated after 2 steps


def test_env_stop_action(mock_evaluator):
    wt = "MSKGEELFTG"
    env = ProteinDesignEnv(
        wildtype_sequence=wt,
        evaluator=mock_evaluator,
        allowed_positions=[0, 1],
        max_mutations=5,
        allow_stop_action=True,
    )

    env.reset()
    _, reward, term, trunc, info = env.step(env.stop_action_id)
    assert term is True
    assert trunc is False
    assert info["stopped_early"] is True


def test_env_constraint_masking(mock_evaluator):
    wt = "MSKGEELFTG"
    # Pipeline disallowing Cysteine
    pipeline = ConstraintPipeline([DisallowedResiduesConstraint(disallowed_residues=["C"])])

    env = ProteinDesignEnv(
        wildtype_sequence=wt,
        evaluator=mock_evaluator,
        allowed_positions=[1],
        constraint_pipeline=pipeline,
        max_mutations=2,
    )

    env.reset()
    mask = env.get_action_mask()
    act_to_cys = env.def_to_action[(1, "C")]
    # Mutation to 'C' must be masked out
    assert mask[act_to_cys] is False or mask[act_to_cys] == 0
