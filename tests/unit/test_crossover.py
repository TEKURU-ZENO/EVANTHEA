"""Unit tests for sequence and homologous mutation crossover operators."""

import pytest

from evanthea.data.schemas import Mutation, VariantRecord
from evanthea.evolution.crossover import (
    HomologousMutationCrossover,
    SinglePointCrossover,
)
from evanthea.evolution.individual import Individual
from evanthea.protein.sequence import apply_mutations


def _make_ind(seq: str, var_id: str, muts: list[Mutation] | None = None) -> Individual:
    v = VariantRecord(
        variant_id=var_id,
        protein_id="P",
        mutations=muts or [],
        mutated_sequence=seq,
        is_wildtype=not bool(muts),
    )
    return Individual(variant=v)


def test_single_point_crossover():
    ref = "AAAAA"
    # Parent 1: "MAAAA", mutation M1A
    m1 = Mutation(ref_residue="A", position_1based=1, mut_residue="M")
    p1 = _make_ind("MAAAA", "p1", [m1])

    # Parent 2: "AAAAV", mutation A5V
    m2 = Mutation(ref_residue="A", position_1based=5, mut_residue="V")
    p2 = _make_ind("AAAAV", "p2", [m2])

    op = SinglePointCrossover(reference_sequence=ref, seed=42)
    c1, c2 = op.crossover(p1, p2, generation=1)

    assert len(c1.sequence) == 5
    assert len(c2.sequence) == 5
    assert c1.parent_ids == ["p1", "p2"]
    assert c2.parent_ids == ["p1", "p2"]
    assert c1.generation == 1

    # Verified against reference
    assert c1.sequence == apply_mutations(ref, c1.mutations)
    assert c2.sequence == apply_mutations(ref, c2.mutations)


def test_single_point_crossover_length_mismatch():
    ref = "AAAAA"
    p1 = _make_ind("AAAAA", "p1")
    p2 = _make_ind("AAAA", "p2")
    op = SinglePointCrossover(reference_sequence=ref)
    with pytest.raises(ValueError):
        op.crossover(p1, p2)


def test_homologous_mutation_crossover():
    ref = "AAAAA"
    # Parent 1 has mutation at pos 1: M
    # Parent 2 has mutation at pos 5: V
    m1 = Mutation(ref_residue="A", position_1based=1, mut_residue="M")
    m2 = Mutation(ref_residue="A", position_1based=5, mut_residue="V")
    p1 = _make_ind("MAAAA", "p1", [m1])
    p2 = _make_ind("AAAAV", "p2", [m2])

    op = HomologousMutationCrossover(reference_sequence=ref, seed=10)
    c1, c2 = op.crossover(p1, p2, generation=2)

    assert len(c1.sequence) == 5
    assert len(c2.sequence) == 5
    assert c1.sequence == apply_mutations(ref, c1.mutations)
    assert c2.sequence == apply_mutations(ref, c2.mutations)

    # Check position conflict handling
    # Both have mutation at pos 2 with different residues (C vs D)
    p3 = _make_ind("ACAAA", "p3", [Mutation(ref_residue="A", position_1based=2, mut_residue="C")])
    p4 = _make_ind("ADAAA", "p4", [Mutation(ref_residue="A", position_1based=2, mut_residue="D")])
    c3, c4 = op.crossover(p3, p4, generation=3)
    assert c3.sequence[1] in ["C", "D"]
    assert c4.sequence[1] in ["C", "D"]
