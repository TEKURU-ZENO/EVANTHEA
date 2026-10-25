"""Unit tests for single-point mutation generators (Uniform, BLOSUM62, ESM-2)."""

import pytest
import numpy as np

from evanthea.data.schemas import Mutation
from evanthea.exceptions import CoordinateOutOfBoundsError, SequenceValidationError
from evanthea.mutation.point_mutations import (
    BLOSUM62MutationGenerator,
    ESM2PointMutationGenerator,
    UniformRandomMutationGenerator,
)
from evanthea.protein.language_model import MockProteinLanguageModel


def test_uniform_random_generator_wildtype_exclusion_and_seed():
    gen1 = UniformRandomMutationGenerator(seed=42)
    gen2 = UniformRandomMutationGenerator(seed=42)
    parent = "MSKGE"

    props1 = gen1.propose(parent, n_proposals=5)
    props2 = gen2.propose(parent, n_proposals=5)

    assert len(props1) == 5
    assert len(props2) == 5

    # Check reproducibility
    for p1, p2 in zip(props1, props2):
        assert p1.child_sequence == p2.child_sequence
        assert p1.mutations[0].to_string() == p2.mutations[0].to_string()

    # Check wild-type exclusion
    for p in props1:
        mut = p.mutations[0]
        assert mut.ref_residue != mut.mut_residue
        assert mut.position_1based in [1, 2, 3, 4, 5]
        assert parent[mut.position_0based] == mut.ref_residue
        assert p.child_sequence[mut.position_0based] == mut.mut_residue
        assert len(p.child_sequence) == len(parent)


def test_uniform_random_generator_allowed_positions():
    gen = UniformRandomMutationGenerator(seed=123)
    parent = "MSKGEELFTG"
    allowed = {2, 4}

    props = gen.propose(parent, n_proposals=20, allowed_positions=allowed)
    for p in props:
        pos = p.mutations[0].position_1based
        assert pos in allowed
        assert pos != 1
        assert pos != 3


def test_blosum62_generator_sampling_and_temperature():
    gen_low_t = BLOSUM62MutationGenerator(temperature=0.1, seed=42)
    gen_high_t = BLOSUM62MutationGenerator(temperature=10.0, seed=42)
    parent = "IIIII"  # Isoleucine prefers Valine/Leucine in BLOSUM62

    # Low temperature should strongly concentrate on I -> V or I -> L
    props_low_t = gen_low_t.propose(parent, n_proposals=30)
    mut_aas_low = [p.mutations[0].mut_residue for p in props_low_t]
    assert all(aa in ["V", "L", "M"] for aa in mut_aas_low)

    # High temperature should explore more diverse residues
    props_high_t = gen_high_t.propose(parent, n_proposals=30)
    mut_aas_high = set(p.mutations[0].mut_residue for p in props_high_t)
    assert len(mut_aas_high) > 5


def test_esm2_point_generator_basic():
    mock_plm = MockProteinLanguageModel()
    gen = ESM2PointMutationGenerator(language_model=mock_plm, temperature=1.0, seed=42)
    parent = "MSKGE"

    props = gen.propose(parent, n_proposals=3)
    assert len(props) == 3
    for p in props:
        assert p.generator_name == "esm2_point"
        assert len(p.mutations) == 1
        mut = p.mutations[0]
        assert mut.ref_residue != mut.mut_residue
        assert len(p.step_proposal_scores) == 1
        assert isinstance(p.step_proposal_scores[0], float)
        assert p.final_variant_score == p.step_proposal_scores[0]


def test_esm2_point_generator_top_k():
    mock_plm = MockProteinLanguageModel()
    gen = ESM2PointMutationGenerator(
        language_model=mock_plm,
        temperature=1.0,
        top_k=2,
        seed=10,
    )
    parent = "MSKGE"
    allowed_pos = {1}
    props = gen.propose(parent, n_proposals=20, allowed_positions=allowed_pos)

    # Sampled residues at position 1 should be restricted to top-2 candidates
    sampled_aas = set(p.mutations[0].mut_residue for p in props)
    assert len(sampled_aas) <= 2


def test_esm2_point_generator_position_strategies():
    mock_plm = MockProteinLanguageModel()
    parent = "MSKGEELFTG"

    # Test all 3 strategies run and produce valid proposals
    for strat in ["uniform", "entropy_weighted", "conservation_weighted"]:
        gen = ESM2PointMutationGenerator(
            language_model=mock_plm,
            position_strategy=strat,
            seed=42,
        )
        props = gen.propose(parent, n_proposals=5)
        assert len(props) == 5
        for p in props:
            assert p.position_strategy == strat
            assert 1 <= p.mutations[0].position_1based <= len(parent)


def test_esm2_point_generator_top_p():
    mock_plm = MockProteinLanguageModel()
    gen = ESM2PointMutationGenerator(
        language_model=mock_plm,
        temperature=1.0,
        top_p=0.6,
        seed=10,
    )
    parent = "MSKGE"
    props = gen.propose(parent, n_proposals=10)
    assert len(props) == 10
    for p in props:
        assert p.top_p == 0.6


def test_proposal_record_properties():
    from evanthea.mutation.base import ProposalRecord
    p_wt = ProposalRecord(
        parent_hash="p",
        child_sequence="MSKGE",
        child_hash="p",
        mutations=[],
        generator_name="test",
    )
    assert p_wt.mutation_string == "WT"

    p_mut = ProposalRecord(
        parent_hash="p",
        child_sequence="ASKGE",
        child_hash="c",
        mutations=[Mutation(ref_residue="M", position_1based=1, mut_residue="A")],
        generator_name="test",
    )
    assert p_mut.mutation_string == "M1A"


def test_generator_error_handling():
    gen = UniformRandomMutationGenerator()
    with pytest.raises(SequenceValidationError):
        gen.propose("M123X")

    with pytest.raises(CoordinateOutOfBoundsError):
        gen.propose("MSKGE", allowed_positions={10})

    with pytest.raises(ValueError):
        gen.propose("MSKGE", allowed_positions=set())

    with pytest.raises(ValueError):
        BLOSUM62MutationGenerator(temperature=-1.0)

    mock_plm = MockProteinLanguageModel()
    with pytest.raises(ValueError):
        ESM2PointMutationGenerator(language_model=mock_plm, temperature=0.0)

    with pytest.raises(ValueError):
        ESM2PointMutationGenerator(language_model=mock_plm, top_k=25)

    with pytest.raises(ValueError):
        ESM2PointMutationGenerator(language_model=mock_plm, top_p=1.5)
