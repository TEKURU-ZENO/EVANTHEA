"""Unit tests for biological and biophysical constraints pipeline."""

import pytest

from evanthea.data.schemas import Mutation
from evanthea.mutation.base import ProposalRecord
from evanthea.mutation.constraints import (
    AllowedResiduesPerPositionConstraint,
    ChargeDeltaConstraint,
    ConstraintPipeline,
    DisallowedResiduesConstraint,
    MaxMutationCountConstraint,
    MinimumLikelihoodConstraint,
    ProtectedPositionsConstraint,
    compute_charge_delta,
    compute_net_charge,
)


def test_protected_positions_constraint():
    constraint = ProtectedPositionsConstraint(protected_positions={65, 66, 67})
    parent = "A" * 100

    # Non-protected mutation should pass
    mut_pass = [Mutation(ref_residue="A", position_1based=10, mut_residue="V")]
    valid, reason = constraint.evaluate(parent, mut_pass, "A" * 9 + "V" + "A" * 90)
    assert valid is True
    assert reason is None

    # Mutation at position 66 (chromophore) should be rejected
    mut_fail = [Mutation(ref_residue="A", position_1based=66, mut_residue="G")]
    valid, reason = constraint.evaluate(parent, mut_fail, "A" * 65 + "G" + "A" * 34)
    assert valid is False
    assert "Violated protected positions" in reason
    assert "pos 66" in reason


def test_disallowed_residues_constraint():
    constraint = DisallowedResiduesConstraint(disallowed_residues={"C"})
    parent = "MSKGE"

    mut_pass = [Mutation(ref_residue="M", position_1based=1, mut_residue="A")]
    valid, reason = constraint.evaluate(parent, mut_pass, "ASKGE")
    assert valid is True

    mut_fail = [Mutation(ref_residue="M", position_1based=1, mut_residue="C")]
    valid, reason = constraint.evaluate(parent, mut_fail, "CSKGE")
    assert valid is False
    assert "Disallowed amino acid" in reason


def test_allowed_residues_per_position_constraint():
    constraint = AllowedResiduesPerPositionConstraint(allowed_map={1: {"A", "V"}, 2: {"S", "T"}})
    parent = "MSKGE"

    mut_pass = [Mutation(ref_residue="M", position_1based=1, mut_residue="A")]
    valid, _ = constraint.evaluate(parent, mut_pass, "ASKGE")
    assert valid is True

    mut_fail = [Mutation(ref_residue="M", position_1based=1, mut_residue="W")]
    valid, reason = constraint.evaluate(parent, mut_fail, "WSKGE")
    assert valid is False
    assert "not in allowed set" in reason


def test_max_mutation_count_constraint():
    constraint = MaxMutationCountConstraint(max_mutations=2)
    parent = "MSKGE"

    muts_2 = [
        Mutation(ref_residue="M", position_1based=1, mut_residue="A"),
        Mutation(ref_residue="S", position_1based=2, mut_residue="T"),
    ]
    valid, _ = constraint.evaluate(parent, muts_2, "ATKGE")
    assert valid is True

    muts_3 = muts_2 + [Mutation(ref_residue="K", position_1based=3, mut_residue="R")]
    valid, reason = constraint.evaluate(parent, muts_3, "ATRGE")
    assert valid is False
    assert "exceeds maximum allowed" in reason


def test_charge_delta_constraint():
    # R (+1), K (+1), D (-1), E (-1)
    muts = [
        Mutation(ref_residue="D", position_1based=1, mut_residue="R"),  # -1 -> +1 = +2
        Mutation(ref_residue="E", position_1based=2, mut_residue="K"),  # -1 -> +1 = +2
    ]
    delta = compute_charge_delta(muts)
    assert delta == 4.0

    constraint = ChargeDeltaConstraint(max_abs_delta=2.0)
    valid, reason = constraint.evaluate("DEMMM", muts, "RKMMM")
    assert valid is False
    assert "Net charge shift |4.00| exceeds maximum allowed" in reason

    # Neutral shift (D -> E: -1 -> -1 = 0)
    mut_neutral = [Mutation(ref_residue="D", position_1based=1, mut_residue="E")]
    valid, _ = constraint.evaluate("DEMMM", mut_neutral, "EEMMM")
    assert valid is True


def test_minimum_likelihood_constraint():
    constraint = MinimumLikelihoodConstraint(min_delta_log_p=-3.0)

    # Proposal with good score
    good_prop = ProposalRecord(
        parent_hash="p",
        child_sequence="ASKGE",
        child_hash="c",
        mutations=[Mutation(ref_residue="M", position_1based=1, mut_residue="A")],
        generator_name="esm2",
        step_proposal_scores=[-0.5],
        final_variant_score=-0.5,
    )
    valid, _ = constraint.evaluate("MSKGE", good_prop.mutations, good_prop.child_sequence, good_prop)
    assert valid is True

    # Proposal with bad score
    bad_prop = ProposalRecord(
        parent_hash="p",
        child_sequence="WSKGE",
        child_hash="c2",
        mutations=[Mutation(ref_residue="M", position_1based=1, mut_residue="W")],
        generator_name="esm2",
        step_proposal_scores=[-4.5],
        final_variant_score=-4.5,
    )
    valid, reason = constraint.evaluate("MSKGE", bad_prop.mutations, bad_prop.child_sequence, bad_prop)
    assert valid is False
    assert "falls below cutoff" in reason


def test_constraint_pipeline_and_gfp_factory():
    pipeline = ConstraintPipeline.create_gfp_pipeline(
        disallow_cysteine=True,
        max_charge_delta=2.0,
        max_mutations=3,
        min_delta_log_p=-5.0,
    )
    parent = "A" * 100

    # 1. Valid proposal
    p_valid = ProposalRecord(
        parent_hash="p1",
        child_sequence="V" + "A" * 99,
        child_hash="c1",
        mutations=[Mutation(ref_residue="A", position_1based=1, mut_residue="V")],
        generator_name="esm2",
        step_proposal_scores=[-1.0],
        final_variant_score=-1.0,
    )

    # 2. Invalid proposal (violates chromophore pos 66)
    p_chromophore = ProposalRecord(
        parent_hash="p1",
        child_sequence="A" * 65 + "V" + "A" * 34,
        child_hash="c2",
        mutations=[Mutation(ref_residue="A", position_1based=66, mut_residue="V")],
        generator_name="esm2",
        step_proposal_scores=[-1.0],
        final_variant_score=-1.0,
    )

    # 3. Invalid proposal (introduces Cys)
    p_cys = ProposalRecord(
        parent_hash="p1",
        child_sequence="C" + "A" * 99,
        child_hash="c3",
        mutations=[Mutation(ref_residue="A", position_1based=1, mut_residue="C")],
        generator_name="esm2",
        step_proposal_scores=[-1.0],
        final_variant_score=-1.0,
    )

    accepted, rejected = pipeline.filter_proposals([p_valid, p_chromophore, p_cys], parent_sequence=parent)
    assert len(accepted) == 1
    assert accepted[0].child_sequence == p_valid.child_sequence
    assert accepted[0].is_accepted is True

    assert len(rejected) == 2
    assert rejected[0].is_accepted is False
    assert "Violated protected positions" in rejected[0].rejection_reasons[0]
    assert "Disallowed amino acid" in rejected[1].rejection_reasons[0]


def test_constraint_value_errors_and_net_charge():
    charge = compute_net_charge("RKHDE")  # 1.0 + 1.0 + 0.1 - 1.0 - 1.0 = 0.1
    assert abs(charge - 0.1) < 1e-5

    with pytest.raises(ValueError):
        MaxMutationCountConstraint(max_mutations=0)

    with pytest.raises(ValueError):
        ChargeDeltaConstraint(max_abs_delta=-0.5)

    c = MinimumLikelihoodConstraint()
    valid, reason = c.evaluate("AAA", [], "AAA", proposal=None)
    assert valid is True
    assert reason is None
