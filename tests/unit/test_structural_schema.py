"""Unit tests for structural data schemas (ClashRecord, StructurePredictionResult, StructuralAssessment)."""

import pytest

from evanthea.structure.schema import ClashRecord, StructuralAssessment, StructurePredictionResult


def test_clash_record_model():
    clash = ClashRecord(
        atom1_name="CA",
        atom2_name="CB",
        res1_id=10,
        res2_id=25,
        res1_name="LEU",
        res2_name="VAL",
        distance=2.1,
        overlap=0.9,
    )
    assert clash.res1_id == 10
    assert clash.overlap == 0.9


def test_structure_prediction_result():
    res = StructurePredictionResult(
        sequence_hash="abcdef1234567890" * 4,
        variant_id="VAR_1",
        sequence="ACDEF",
        pdb_text="ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C\nEND\n",
        plddt_scores=[90.0, 91.0, 89.0, 92.0, 88.0],
        mean_plddt=90.0,
        ptm_score=0.88,
    )
    assert res.mean_plddt == 90.0
    assert res.ptm_score == 0.88


def test_structural_assessment_viability():
    assess = StructuralAssessment(
        sequence_hash="1234567890abcdef" * 4,
        variant_id="LEAD_CANDIDATE",
        sequence="MSKGEELFTG",
        mean_plddt=88.5,
        plddt_per_residue=[85.0] * 10,
        disordered_residues_count=0,
        rmsd_to_reference=1.2,
        clash_count=0,
        contact_preservation_ratio=0.95,
        sasa_total=2500.0,
        sasa_hydrophobic=800.0,
        is_structurally_viable=True,
        violations=[],
    )
    assert assess.is_structurally_viable is True
    assert len(assess.violations) == 0
