"""Domain exceptions for the EVANTHEA framework."""

from typing import Optional


class EvantheaException(Exception):
    """Base exception for all EVANTHEA domain errors."""
    pass


class SequenceValidationError(EvantheaException):
    """Raised when an amino acid sequence contains invalid or non-canonical residues."""
    def __init__(self, message: str, invalid_characters: Optional[set[str]] = None, position: Optional[int] = None):
        super().__init__(message)
        self.invalid_characters = invalid_characters or set()
        self.position = position


class MutationParseError(EvantheaException):
    """Raised when a mutation string does not conform to expected notation (e.g., 'M1A' or 'M1A:G2C')."""
    def __init__(self, mutation_str: str, message: str):
        super().__init__(f"Failed to parse mutation '{mutation_str}': {message}")
        self.mutation_str = mutation_str


class CoordinateOutOfBoundsError(EvantheaException):
    """Raised when a mutation position exceeds the sequence boundaries or is non-positive."""
    def __init__(self, position: int, sequence_length: int, mutation_str: Optional[str] = None):
        super().__init__(
            f"Position {position} is out of bounds for sequence of length {sequence_length} "
            f"(valid range: 1 to {sequence_length})" + (f" in mutation '{mutation_str}'" if mutation_str else "")
        )
        self.position = position
        self.sequence_length = sequence_length
        self.mutation_str = mutation_str


class SequenceMismatchError(EvantheaException):
    """Raised when the wild-type residue in a mutation does not match the reference sequence."""
    def __init__(self, position: int, expected_wt: str, actual_wt: str, mutation_str: Optional[str] = None):
        super().__init__(
            f"Wild-type residue mismatch at position {position}: "
            f"mutation specifies '{expected_wt}', but reference sequence has '{actual_wt}'"
            + (f" in mutation '{mutation_str}'" if mutation_str else "")
        )
        self.position = position
        self.expected_wt = expected_wt
        self.actual_wt = actual_wt
        self.mutation_str = mutation_str


class DuplicateMutationError(EvantheaException):
    """Raised when a variant specifies multiple conflicting mutations at the same position."""
    def __init__(self, position: int, message: str):
        super().__init__(f"Duplicate or conflicting mutation at position {position}: {message}")
        self.position = position


class DataLeakageError(EvantheaException):
    """Raised when train/validation/test splits violate disjointness or leakage-prevention constraints."""
    def __init__(self, message: str, leaking_identifiers: Optional[list[str]] = None):
        super().__init__(f"Data leakage detected: {message}")
        self.leaking_identifiers = leaking_identifiers or []


class ModelInferenceError(EvantheaException):
    """Raised when a protein language model fails during tokenization, inference, or scoring."""
    pass


class CacheError(EvantheaException):
    """Raised when embedding or evaluation cache encounters serialization or integrity failures."""
    pass
