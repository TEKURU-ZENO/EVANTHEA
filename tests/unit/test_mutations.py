"""Unit tests for mutation parsing, coordinate normalization, and variant reconstruction."""

import pytest
from evanthea.data.schemas import Mutation
from evanthea.exceptions import (
    CoordinateOutOfBoundsError,
    DuplicateMutationError,
    MutationParseError,
    SequenceMismatchError,
)
from evanthea.protein.sequence import apply_mutations, parse_mutation_string


def test_parse_mutation_string_single():
    """Parses standard single mutation notation (e.g. M1A)."""
    muts = parse_mutation_string("M1A")
    assert len(muts) == 1
    assert muts[0].ref_residue == "M"
    assert muts[0].position_1based == 1
    assert muts[0].position_0based == 0
    assert muts[0].mut_residue == "A"
    assert muts[0].to_string() == "M1A"


def test_parse_mutation_string_combinatorial():
    """Parses multi-mutants across different delimiter formats."""
    # Colon delimited (ProteinGym style)
    muts1 = parse_mutation_string("K3E:F8L:G10A")
    assert len(muts1) == 3
    assert [m.to_string() for m in muts1] == ["K3E", "F8L", "G10A"]

    # Comma and whitespace delimited
    muts2 = parse_mutation_string("K3E, F8L , G10A")
    assert len(muts2) == 3

    # Mixed ordering gets sorted by 1-based position
    muts3 = parse_mutation_string("G10A:K3E")
    assert [m.to_string() for m in muts3] == ["K3E", "G10A"]


def test_parse_mutation_string_wt():
    """Wild-type strings return empty mutation list."""
    assert parse_mutation_string("WT") == []
    assert parse_mutation_string("wt") == []
    assert parse_mutation_string("") == []
    assert parse_mutation_string("None") == []


def test_parse_mutation_string_invalid_syntax():
    """Malformed mutation syntax throws MutationParseError."""
    with pytest.raises(MutationParseError):
        parse_mutation_string("12A")  # Missing ref AA
    with pytest.raises(MutationParseError):
        parse_mutation_string("MA")   # Missing position
    with pytest.raises(MutationParseError):
        parse_mutation_string("M1")   # Missing mut AA
    with pytest.raises(MutationParseError):
        parse_mutation_string("M0A")  # Position 0 invalid in 1-based convention


def test_parse_mutation_string_duplicate_conflict():
    """Conflicting substitutions at the same position raise DuplicateMutationError."""
    with pytest.raises(DuplicateMutationError):
        parse_mutation_string("M1A:M1C")


def test_apply_mutations_successful_reconstruction():
    """Applying parsed mutations reproduces ground truth sequence."""
    wt_seq = "MSKGEELFTG"
    muts = parse_mutation_string("M1A:E5K:G10A")
    mutated = apply_mutations(wt_seq, muts)
    assert mutated == "ASKGKELFTA"


def test_apply_mutations_wildtype_mismatch():
    """Throws SequenceMismatchError if mutation specifies incorrect wild-type residue."""
    wt_seq = "MSKGEELFTG"
    # Position 1 is M, not L
    muts = [Mutation(ref_residue="L", position_1based=1, mut_residue="A")]
    with pytest.raises(SequenceMismatchError) as exc_info:
        apply_mutations(wt_seq, muts)
    assert exc_info.value.position == 1
    assert exc_info.value.expected_wt == "L"
    assert exc_info.value.actual_wt == "M"


def test_apply_mutations_coordinate_out_of_bounds():
    """Throws CoordinateOutOfBoundsError if position exceeds sequence length."""
    wt_seq = "MSKGE"
    muts = [Mutation(ref_residue="M", position_1based=10, mut_residue="A")]
    with pytest.raises(CoordinateOutOfBoundsError) as exc_info:
        apply_mutations(wt_seq, muts)
    assert exc_info.value.position == 10
    assert exc_info.value.sequence_length == 5
