"""Unit tests for CombinatorialMutationEngine (independent and sequential context-aware multi-mutants)."""

import pytest

from evanthea.data.schemas import Mutation
from evanthea.mutation.combination import CombinatorialMutationEngine
from evanthea.mutation.point_mutations import (
    BLOSUM62MutationGenerator,
    ESM2PointMutationGenerator,
    UniformRandomMutationGenerator,
)
from evanthea.protein.language_model import MockProteinLanguageModel
from evanthea.protein.sequence import apply_mutations


def test_combinatorial_engine_independent_fixed_degree():
    point_gen = UniformRandomMutationGenerator(seed=42)
    engine = CombinatorialMutationEngine(
        point_generator=point_gen,
        degree_mode="fixed",
        max_degree=3,
        proposal_mode="independent",
        seed=42,
    )
    parent = "MSKGEELFTG"

    props = engine.propose(parent, n_proposals=5)
    assert len(props) == 5

    for p in props:
        assert len(p.mutations) == 3
        # Check no duplicate positions
        positions = [m.position_1based for m in p.mutations]
        assert len(set(positions)) == 3
        # Mutations should be sorted by position
        assert positions == sorted(positions)
        # Check reconstruction
        expected_seq = apply_mutations(parent, p.mutations)
        assert p.child_sequence == expected_seq


def test_combinatorial_engine_sequential_context_aware():
    point_gen = BLOSUM62MutationGenerator(temperature=1.0, seed=10)
    engine = CombinatorialMutationEngine(
        point_generator=point_gen,
        degree_mode="uniform",
        min_degree=2,
        max_degree=4,
        proposal_mode="sequential",
        seed=10,
    )
    parent = "MSKGEELFTG"

    props = engine.propose(parent, n_proposals=5)
    assert len(props) == 5

    for p in props:
        k = len(p.mutations)
        assert 2 <= k <= 4
        positions = [m.position_1based for m in p.mutations]
        assert len(set(positions)) == k
        assert positions == sorted(positions)
        assert p.child_sequence == apply_mutations(parent, p.mutations)
        assert len(p.step_proposal_scores) == k


def test_combinatorial_engine_with_mock_plm():
    mock_plm = MockProteinLanguageModel()
    point_gen = ESM2PointMutationGenerator(
        language_model=mock_plm,
        temperature=1.0,
        seed=99,
    )
    engine = CombinatorialMutationEngine(
        point_generator=point_gen,
        degree_mode="poisson",
        min_degree=1,
        max_degree=4,
        poisson_lambda=2.0,
        proposal_mode="sequential",
        language_model=mock_plm,
        seed=99,
    )
    parent = "MSKGEELFTG"

    props = engine.propose(parent, n_proposals=5)
    assert len(props) == 5

    for p in props:
        assert 1 <= len(p.mutations) <= 4
        assert p.final_variant_score is not None
        assert p.child_sequence == apply_mutations(parent, p.mutations)


def test_combinatorial_engine_allowed_positions_bound():
    point_gen = UniformRandomMutationGenerator(seed=1)
    engine = CombinatorialMutationEngine(
        point_generator=point_gen,
        degree_mode="fixed",
        max_degree=3,
        min_degree=3,
    )
    # Only 2 allowed positions, but min_degree is 3 -> should raise ValueError
    with pytest.raises(ValueError):
        engine.propose("MSKGEELFTG", allowed_positions={1, 2})


def test_combinatorial_engine_reproducibility():
    point_gen1 = UniformRandomMutationGenerator()
    engine1 = CombinatorialMutationEngine(point_gen1, degree_mode="fixed", max_degree=2, seed=123)

    point_gen2 = UniformRandomMutationGenerator()
    engine2 = CombinatorialMutationEngine(point_gen2, degree_mode="fixed", max_degree=2, seed=123)

    props1 = engine1.propose("MSKGEELFTG", n_proposals=3)
    props2 = engine2.propose("MSKGEELFTG", n_proposals=3)

    for p1, p2 in zip(props1, props2):
        assert p1.child_sequence == p2.child_sequence
        assert [m.to_string() for m in p1.mutations] == [m.to_string() for m in p2.mutations]


def test_combinatorial_engine_parameter_errors():
    gen = UniformRandomMutationGenerator()
    with pytest.raises(ValueError):
        CombinatorialMutationEngine(gen, min_degree=0)

    with pytest.raises(ValueError):
        CombinatorialMutationEngine(gen, min_degree=3, max_degree=2)

    with pytest.raises(ValueError):
        CombinatorialMutationEngine(gen, poisson_lambda=-0.1)
