"""Unit tests for sequence validation and amino acid mechanics."""

import pytest
from evanthea.exceptions import SequenceValidationError
from evanthea.protein.sequence import (
    compute_hamming_distance,
    compute_sequence_hash,
    validate_amino_acid_string,
)


def test_validate_amino_acid_string_valid():
    """Canonical sequences are correctly normalized and validated."""
    seq = "mskgEelftgvvpiliEldGdvnghkfs"
    cleaned = validate_amino_acid_string(seq)
    assert cleaned == seq.upper()
    assert len(cleaned) == 28


def test_validate_amino_acid_string_all_canonical():
    """All 20 standard IUPAC amino acids are accepted."""
    all_20 = "ACDEFGHIKLMNPQRSTVWY"
    assert validate_amino_acid_string(all_20) == all_20


def test_validate_amino_acid_string_invalid_characters():
    """Non-canonical and ambiguous residues (B, Z, X, J, U, O, numbers) are strictly rejected."""
    with pytest.raises(SequenceValidationError) as exc_info:
        validate_amino_acid_string("MSKGEELFTXGVVP")
    assert "X" in exc_info.value.invalid_characters
    assert exc_info.value.position == 10

    with pytest.raises(SequenceValidationError):
        validate_amino_acid_string("MSKGE123")


def test_validate_amino_acid_string_empty():
    """Empty or whitespace strings raise SequenceValidationError."""
    with pytest.raises(SequenceValidationError):
        validate_amino_acid_string("")
    with pytest.raises(SequenceValidationError):
        validate_amino_acid_string("   ")


def test_compute_sequence_hash():
    """Hashing is deterministic and case-insensitive."""
    h1 = compute_sequence_hash("MskgE")
    h2 = compute_sequence_hash("MSKGE")
    assert h1 == h2
    assert len(h1) == 64  # SHA-256


def test_compute_hamming_distance():
    """Hamming distance accurately counts residue substitutions."""
    assert compute_hamming_distance("MSKGE", "MSKGE") == 0
    assert compute_hamming_distance("MSKGE", "MAKGE") == 1
    assert compute_hamming_distance("MSKGE", "MSAAA") == 3
    assert compute_hamming_distance("MSKGE", "AAAAA") == 5

    with pytest.raises(ValueError):
        compute_hamming_distance("MSK", "MSKGE")
