"""Unit tests for proposal distribution analysis and statistical benchmarking."""

import pytest

from evanthea.data.schemas import Mutation
from evanthea.mutation.analysis import (
    compare_proposal_generators,
    compute_proposal_distribution_metrics,
)
from evanthea.mutation.base import ProposalRecord
from evanthea.mutation.constraints import ConstraintPipeline, ProtectedPositionsConstraint
from evanthea.mutation.point_mutations import (
    BLOSUM62MutationGenerator,
    UniformRandomMutationGenerator,
)


def test_compute_proposal_distribution_metrics_empty():
    res = compute_proposal_distribution_metrics([], "MSKGE")
    assert res["n_proposals"] == 0


def test_compute_proposal_distribution_metrics_basic():
    parent = "MSKGE"
    p1 = ProposalRecord(
        parent_hash="p",
        child_sequence="ASKGE",
        child_hash="c1",
        mutations=[Mutation(ref_residue="M", position_1based=1, mut_residue="A")],
        generator_name="test",
        step_proposal_scores=[-1.2],
        final_variant_score=-1.2,
        is_accepted=True,
    )
    p2 = ProposalRecord(
        parent_hash="p",
        child_sequence="VSKGE",
        child_hash="c2",
        mutations=[Mutation(ref_residue="M", position_1based=1, mut_residue="V")],
        generator_name="test",
        step_proposal_scores=[-0.5],
        final_variant_score=-0.5,
        is_accepted=False,
    )

    metrics = compute_proposal_distribution_metrics([p1, p2], parent)
    assert metrics["n_proposals"] == 2
    assert metrics["acceptance_rate"] == 0.5
    assert metrics["plausibility"]["mean_delta_log_p"] == pytest.approx(-0.85)
    assert metrics["residue_shannon_entropy_bits"] > 0.0
    assert metrics["mean_pairwise_hamming"] == 1.0


def test_compare_proposal_generators():
    parent = "MSKGEELFTG"
    generators = {
        "uniform": UniformRandomMutationGenerator(seed=42),
        "blosum62": BLOSUM62MutationGenerator(temperature=1.0, seed=42),
    }
    pipeline = ConstraintPipeline([ProtectedPositionsConstraint({1, 2})])

    comp = compare_proposal_generators(
        generators=generators,
        parent_sequence=parent,
        n_proposals=20,
        constraint_pipeline=pipeline,
    )

    assert comp["parent_length"] == len(parent)
    assert comp["n_proposals_per_generator"] == 20
    assert "uniform" in comp["generators"]
    assert "blosum62" in comp["generators"]

    u_metrics = comp["generators"]["uniform"]
    b_metrics = comp["generators"]["blosum62"]

    assert u_metrics["n_proposals"] == 20
    assert b_metrics["n_proposals"] == 20
    assert 0.0 <= u_metrics["acceptance_rate"] <= 1.0
    assert 0.0 <= b_metrics["acceptance_rate"] <= 1.0
