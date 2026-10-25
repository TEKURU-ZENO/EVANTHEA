"""Sequence alignment, sub-sequence matching, and coordinate offset mapping."""

from typing import Optional, Tuple


def find_exact_subsequence_offset(reference_sequence: str, query_sequence: str) -> Optional[int]:
    """Finds the 0-indexed starting offset of query_sequence within reference_sequence.

    Returns:
        0-indexed offset if exact match found, else None.
    """
    pos = reference_sequence.find(query_sequence)
    return pos if pos != -1 else None


def align_construct_to_reference(
    construct_seq: str,
    uniprot_seq: str
) -> Tuple[int, float]:
    """Finds the optimal alignment offset and sequence identity between an assay construct and a reference.

    Uses a sliding window search for maximum identity if exact match fails.

    Returns:
        Tuple of (best_0based_offset, fractional_identity)
    """
    if construct_seq in uniprot_seq:
        offset = uniprot_seq.find(construct_seq)
        return offset, 1.0

    best_offset = 0
    best_matches = -1
    c_len = len(construct_seq)
    u_len = len(uniprot_seq)

    # If construct is longer or equal to reference
    if c_len >= u_len:
        for offset in range(c_len - u_len + 1):
            sub = construct_seq[offset : offset + u_len]
            matches = sum(1 for a, b in zip(sub, uniprot_seq) if a == b)
            if matches > best_matches:
                best_matches = matches
                best_offset = -offset  # negative offset
        return best_offset, best_matches / u_len

    # If construct is shorter than reference
    for offset in range(u_len - c_len + 1):
        sub = uniprot_seq[offset : offset + c_len]
        matches = sum(1 for a, b in zip(sub, construct_seq) if a == b)
        if matches > best_matches:
            best_matches = matches
            best_offset = offset

    identity = best_matches / c_len if c_len > 0 else 0.0
    return best_offset, identity
