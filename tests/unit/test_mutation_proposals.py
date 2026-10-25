"""Unit tests for high-level CandidateProposalAPI facade."""

import pytest

from evanthea.data.schemas import Mutation, VariantRecord
from evanthea.mutation.constraints import (
    ConstraintPipeline,
    DisallowedResiduesConstraint,
    ProtectedPositionsConstraint,
)
from evanthea.mutation.point_mutations import UniformRandomMutationGenerator
from evanthea.mutation.proposals import CandidateProposalAPI


def test_candidate_proposal_api_basic():
    gen = UniformRandomMutationGenerator(seed=42)
    api = CandidateProposalAPI(
        operator=gen,
        protein_id="GFP_AEQVI",
    )
    parent = "MSKGEELFTG"

    candidates = api.propose_candidates(parent, n_candidates=5)
    assert len(candidates) == 5

    for c in candidates:
        assert isinstance(c, VariantRecord)
        assert c.protein_id == "GFP_AEQVI"
        assert len(c.mutations) == 1
        assert "proposal" in c.metadata
        prop_meta = c.metadata["proposal"]
        assert prop_meta["generator_name"] == "uniform_random"
        assert prop_meta["is_accepted"] is True


def test_candidate_proposal_api_with_constraints_and_resampling():
    gen = UniformRandomMutationGenerator(seed=10)
    # Disallow Cysteine and protect position 2
    pipeline = ConstraintPipeline([
        ProtectedPositionsConstraint({2}),
        DisallowedResiduesConstraint({"C"}),
    ])
    api = CandidateProposalAPI(
        operator=gen,
        protein_id="GFP_AEQVI",
        constraint_pipeline=pipeline,
        max_resample_factor=10,
    )
    parent = "MSKGEELFTG"

    candidates = api.propose_candidates(parent, n_candidates=4)
    assert len(candidates) == 4

    for c in candidates:
        for m in c.mutations:
            assert m.position_1based != 2
            assert m.mut_residue != "C"


def test_candidate_proposal_api_accepts_variant_record_parent():
    gen = UniformRandomMutationGenerator(seed=77)
    api = CandidateProposalAPI(operator=gen, protein_id="TEST_PROT")

    parent_record = VariantRecord(
        variant_id="var_wt",
        protein_id="TEST_PROT",
        mutations=[],
        mutated_sequence="MSKGEELFTG",
        is_wildtype=True,
    )

    candidates = api.propose_candidates(parent_record, n_candidates=2)
    assert len(candidates) == 2
    for c in candidates:
        assert c.protein_id == "TEST_PROT"
        assert len(c.mutations) == 1


def test_candidate_proposal_api_audit():
    gen = UniformRandomMutationGenerator(seed=5)
    pipeline = ConstraintPipeline([ProtectedPositionsConstraint({1, 2, 3})])
    api = CandidateProposalAPI(operator=gen, protein_id="TEST", constraint_pipeline=pipeline)

    accepted, rejected = api.propose_with_audit("MSKGE", n_proposals=10)
    assert len(accepted) + len(rejected) == 10

    for r in rejected:
        assert r.is_accepted is False
        assert any("Violated protected positions" in reason for reason in r.rejection_reasons)

    for a in accepted:
        assert a.is_accepted is True
        assert a.mutations[0].position_1based not in {1, 2, 3}
