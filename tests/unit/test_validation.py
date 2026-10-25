"""Unit tests for dataset row validation and reconciliation against ProteinRecord."""

from evanthea.data.schemas import ProteinRecord
from evanthea.data.validation import validate_variant_against_reference


def test_validate_variant_against_reference_valid_wt():
    wt_seq = "MSKGEELFTG"
    protein = ProteinRecord(id="TEST_PROT", name="Test Protein", wildtype_sequence=wt_seq, length=10)

    variant, issue = validate_variant_against_reference(
        row_idx=0,
        variant_id="var_wt",
        mutated_seq_raw=wt_seq,
        mutation_str_raw="WT",
        reference_protein=protein
    )
    assert issue is None
    assert variant is not None
    assert variant.is_wildtype is True
    assert variant.num_mutations == 0
    assert variant.mutation_string == "WT"


def test_validate_variant_against_reference_valid_mutant():
    wt_seq = "MSKGEELFTG"
    mut_seq = "ASKGEELFTA"
    protein = ProteinRecord(id="TEST_PROT", name="Test Protein", wildtype_sequence=wt_seq, length=10)

    variant, issue = validate_variant_against_reference(
        row_idx=1,
        variant_id="var_mut",
        mutated_seq_raw=mut_seq,
        mutation_str_raw="M1A:G10A",
        reference_protein=protein
    )
    assert issue is None
    assert variant is not None
    assert variant.is_wildtype is False
    assert variant.num_mutations == 2
    assert variant.mutation_string == "M1A:G10A"


def test_validate_variant_against_reference_mismatch_detected():
    wt_seq = "MSKGEELFTG"
    mut_seq = "ASKGEELFTG"
    protein = ProteinRecord(id="TEST_PROT", name="Test Protein", wildtype_sequence=wt_seq, length=10)

    # Mutation string claims position 1 is L instead of M
    variant, issue = validate_variant_against_reference(
        row_idx=2,
        variant_id="var_err",
        mutated_seq_raw=mut_seq,
        mutation_str_raw="L1A",
        reference_protein=protein
    )
    assert variant is None
    assert issue is not None
    assert issue.error_type == "SequenceMismatchError"
