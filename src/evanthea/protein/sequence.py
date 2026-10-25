"""Sequence validation, mutation parsing, coordinate normalization, and variant reconstruction."""

import hashlib
import re
from typing import List, Optional, Set

from evanthea.constants import STANDARD_AMINO_ACIDS
from evanthea.data.schemas import Mutation
from evanthea.exceptions import (
    CoordinateOutOfBoundsError,
    DuplicateMutationError,
    MutationParseError,
    SequenceMismatchError,
    SequenceValidationError,
)

# Regex pattern for single-point substitution notation e.g. M1A or L123F
MUTATION_PATTERN = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")


def validate_amino_acid_string(sequence: str, allow_gaps: bool = False) -> str:
    """Validates that a sequence consists strictly of canonical 20 amino acids.

    Args:
        sequence: The raw amino acid string.
        allow_gaps: If True, '-' gap characters are permitted.

    Returns:
        Cleaned, uppercase sequence string.

    Raises:
        SequenceValidationError: If invalid characters are detected.
    """
    if not sequence or not sequence.strip():
        raise SequenceValidationError("Sequence cannot be empty or whitespace.")

    cleaned = sequence.strip().upper()
    valid_chars: Set[str] = set(STANDARD_AMINO_ACIDS)
    if allow_gaps:
        valid_chars.add("-")

    invalid_chars = set()
    first_invalid_pos = None

    for idx, char in enumerate(cleaned):
        if char not in valid_chars:
            invalid_chars.add(char)
            if first_invalid_pos is None:
                first_invalid_pos = idx + 1  # 1-indexed

    if invalid_chars:
        raise SequenceValidationError(
            f"Sequence contains invalid non-canonical characters: {sorted(invalid_chars)} "
            f"at 1-indexed position {first_invalid_pos}.",
            invalid_characters=invalid_chars,
            position=first_invalid_pos,
        )

    return cleaned


def parse_mutation_string(mutation_str: str) -> List[Mutation]:
    """Parses single or combinatorial mutation strings into structured Mutation models.

    Supported formats:
        - "M1A"
        - "M1A:G2C" (colon-separated, ProteinGym standard)
        - "M1A, G2C" (comma-separated)
        - "M1A/G2C" (slash-separated)
        - "WT", "wt", or "" (returns empty list)

    Args:
        mutation_str: Raw mutation descriptor.

    Returns:
        List of Mutation objects sorted by 1-indexed position.

    Raises:
        MutationParseError: If formatting syntax is invalid.
        DuplicateMutationError: If multiple mutations target the same position.
    """
    if not mutation_str or mutation_str.strip().upper() in {"WT", "WILDTYPE", "WILD_TYPE", "NONE"}:
        return []

    # Normalize delimiters: replace colons, commas, semicolons, slashes with spaces
    tokens = [t.strip() for t in re.split(r"[:;,/\s]+", mutation_str.strip()) if t.strip()]

    parsed_mutations: List[Mutation] = []
    seen_positions: dict[int, str] = {}

    for token in tokens:
        match = MUTATION_PATTERN.match(token)
        if not match:
            raise MutationParseError(
                token,
                "Expected notation '<WT_Residue><1-based Position><Mut_Residue>' (e.g. 'M1A')."
            )

        ref_aa, pos_str, mut_aa = match.groups()
        ref_aa = ref_aa.upper()
        mut_aa = mut_aa.upper()
        pos = int(pos_str)

        if pos <= 0:
            raise MutationParseError(token, f"Position must be >= 1 (1-based), got {pos}.")

        if ref_aa not in STANDARD_AMINO_ACIDS:
            raise MutationParseError(token, f"Reference residue '{ref_aa}' is not a standard amino acid.")
        if mut_aa not in STANDARD_AMINO_ACIDS:
            raise MutationParseError(token, f"Mutant residue '{mut_aa}' is not a standard amino acid.")

        if pos in seen_positions:
            existing_mut = seen_positions[pos]
            if existing_mut != mut_aa:
                raise DuplicateMutationError(
                    pos,
                    f"Conflicting mutations at position {pos}: substituted with both '{existing_mut}' and '{mut_aa}'."
                )
            # If identical duplicate token, ignore or keep once
            continue

        seen_positions[pos] = mut_aa
        parsed_mutations.append(
            Mutation(ref_residue=ref_aa, position_1based=pos, mut_residue=mut_aa)
        )

    # Deterministic sorting by position
    parsed_mutations.sort(key=lambda m: m.position_1based)
    return parsed_mutations


def apply_mutations(reference_sequence: str, mutations: List[Mutation]) -> str:
    """Applies a list of point mutations to a reference sequence with strict validation.

    Args:
        reference_sequence: Validated wild-type sequence.
        mutations: List of Mutation models to apply.

    Returns:
        Reconstructed mutated sequence string.

    Raises:
        CoordinateOutOfBoundsError: If mutation position is outside sequence bounds.
        SequenceMismatchError: If the wild-type residue in mutation doesn't match reference.
    """
    ref_seq = validate_amino_acid_string(reference_sequence)
    seq_len = len(ref_seq)
    seq_chars = list(ref_seq)

    for mut in mutations:
        pos_1b = mut.position_1based
        if pos_1b < 1 or pos_1b > seq_len:
            raise CoordinateOutOfBoundsError(pos_1b, seq_len, mut.to_string())

        pos_0b = mut.position_0based
        actual_wt = seq_chars[pos_0b]
        if actual_wt != mut.ref_residue:
            raise SequenceMismatchError(
                position=pos_1b,
                expected_wt=mut.ref_residue,
                actual_wt=actual_wt,
                mutation_str=mut.to_string()
            )

        seq_chars[pos_0b] = mut.mut_residue

    return "".join(seq_chars)


def compute_sequence_hash(sequence: str) -> str:
    """Computes a deterministic SHA-256 hash of a normalized canonical sequence."""
    cleaned = validate_amino_acid_string(sequence)
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


def compute_hamming_distance(seq1: str, seq2: str) -> int:
    """Computes the pairwise Hamming distance (number of mismatching positions).

    Raises:
        ValueError: If sequences have different lengths.
    """
    if len(seq1) != len(seq2):
        raise ValueError(f"Sequences must have equal length (got {len(seq1)} and {len(seq2)}).")
    return sum(1 for a, b in zip(seq1, seq2) if a != b)
