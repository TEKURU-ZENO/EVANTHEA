"""Comprehensive biological dataset validation, sequence reconciliation, and audit reporting."""

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from evanthea.data.schemas import Mutation, ProteinRecord, VariantRecord
from evanthea.exceptions import EvantheaException
from evanthea.protein.sequence import (
    apply_mutations,
    parse_mutation_string,
    validate_amino_acid_string,
)


class ValidationIssue(BaseModel):
    """Specific validation failure for a variant."""
    row_index: int
    variant_id: str
    error_type: str
    message: str
    mutation_string: Optional[str] = None


class ValidationSummary(BaseModel):
    """Statistical summary of dataset validation results."""
    total_records: int
    valid_records: int
    invalid_records: int
    wildtype_records: int
    single_mutants: int
    multi_mutants: int
    issues: List[ValidationIssue] = Field(default_factory=list)

    @property
    def validity_rate(self) -> float:
        return (self.valid_records / self.total_records) if self.total_records > 0 else 0.0


def validate_variant_against_reference(
    row_idx: int,
    variant_id: str,
    mutated_seq_raw: str,
    mutation_str_raw: Optional[str],
    reference_protein: ProteinRecord,
) -> Tuple[Optional[VariantRecord], Optional[ValidationIssue]]:
    """Validates an individual variant row against the reference protein.

    Args:
        row_idx: Original index in source table.
        variant_id: Unique variant identifier.
        mutated_seq_raw: The amino acid sequence recorded for this variant.
        mutation_str_raw: Raw mutation string (e.g. 'M1A:G2C' or None/empty).
        reference_protein: Ground truth reference protein record.

    Returns:
        Tuple of (VariantRecord if valid else None, ValidationIssue if invalid else None).
    """
    try:
        # 1. Validate candidate mutated sequence characters
        cleaned_mut_seq = validate_amino_acid_string(mutated_seq_raw)

        # 2. Parse mutations if provided
        mutations: List[Mutation] = []
        if mutation_str_raw:
            mutations = parse_mutation_string(mutation_str_raw)

        # 3. Check sequence length against reference
        if len(cleaned_mut_seq) != reference_protein.length:
            return None, ValidationIssue(
                row_index=row_idx,
                variant_id=variant_id,
                error_type="LengthMismatch",
                message=f"Variant sequence length {len(cleaned_mut_seq)} does not match reference {reference_protein.length}",
                mutation_string=mutation_str_raw,
            )

        # 4. If mutations exist, verify that applying them to reference reproduces the candidate sequence
        if mutations:
            reconstructed = apply_mutations(reference_protein.wildtype_sequence, mutations)
            if reconstructed != cleaned_mut_seq:
                return None, ValidationIssue(
                    row_index=row_idx,
                    variant_id=variant_id,
                    error_type="ReconstructionMismatch",
                    message="Applying parsed mutations to wild-type does not match the provided mutated_sequence.",
                    mutation_string=mutation_str_raw,
                )
        else:
            # If no mutations parsed, check if it matches wild-type
            if cleaned_mut_seq != reference_protein.wildtype_sequence:
                return None, ValidationIssue(
                    row_index=row_idx,
                    variant_id=variant_id,
                    error_type="UnannotatedMutations",
                    message="Variant differs from wild-type, but no mutation string was provided.",
                    mutation_string=mutation_str_raw,
                )

        is_wt = (len(mutations) == 0) and (cleaned_mut_seq == reference_protein.wildtype_sequence)

        variant = VariantRecord(
            variant_id=variant_id,
            protein_id=reference_protein.id,
            mutations=mutations,
            mutated_sequence=cleaned_mut_seq,
            is_wildtype=is_wt,
            num_mutations=len(mutations),
        )
        return variant, None

    except EvantheaException as e:
        return None, ValidationIssue(
            row_index=row_idx,
            variant_id=variant_id,
            error_type=e.__class__.__name__,
            message=str(e),
            mutation_string=mutation_str_raw,
        )
    except Exception as e:
        return None, ValidationIssue(
            row_index=row_idx,
            variant_id=variant_id,
            error_type="UnexpectedError",
            message=str(e),
            mutation_string=mutation_str_raw,
        )
