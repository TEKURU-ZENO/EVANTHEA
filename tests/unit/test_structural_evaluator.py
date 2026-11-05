"""Unit tests for StructuralValidator, quality gate auditing, and persistent structure caching."""

import pytest

from evanthea.memory.storage import PersistentMemoryStore
from evanthea.structure.evaluator import StructuralValidator
from evanthea.structure.predictor import MockStructurePredictor


def test_structural_validator_passing(tmp_path):
    wt = "MSKGEELFTG"
    validator = StructuralValidator(
        predictor=MockStructurePredictor(base_plddt=88.0),
        min_plddt=70.0,
        max_rmsd=3.0,
        max_clashes=5,
        min_contact_preservation=0.80,
    )

    # Set reference from wild-type prediction
    wt_pred = validator.predictor.predict_structure(wt, variant_id="WT")
    validator.set_reference_from_text(wt_pred.pdb_text)

    # Validate identical sequence
    assessment, pdb_text = validator.validate_candidate(wt, variant_id="WT_VAL")

    assert assessment.is_structurally_viable is True
    assert len(assessment.violations) == 0
    assert assessment.mean_plddt >= 70.0
    assert assessment.rmsd_to_reference is not None
    assert assessment.rmsd_to_reference < 0.1
    assert assessment.contact_preservation_ratio == 1.0


def test_structural_validator_failing_quality_gates():
    wt = "MSKGEELFTG"
    # Unreasonably strict threshold to force failure
    validator = StructuralValidator(
        predictor=MockStructurePredictor(base_plddt=80.0),
        min_plddt=95.0,  # Strict threshold
        max_rmsd=0.001,  # Strict threshold
    )

    wt_pred = validator.predictor.predict_structure(wt, variant_id="WT")
    validator.set_reference_from_text(wt_pred.pdb_text)

    # Slightly mutated sequence
    mut_seq = "MSKGEELATG"
    assessment, _ = validator.validate_candidate(mut_seq, variant_id="MUT_VAL")

    assert assessment.is_structurally_viable is False
    assert len(assessment.violations) > 0
    assert any("pLDDT" in v for v in assessment.violations)


def test_structural_validator_caching(tmp_path):
    db_path = tmp_path / "struct_cache.db"
    store = PersistentMemoryStore(db_path=db_path)

    validator = StructuralValidator(
        predictor=MockStructurePredictor(),
        memory_store=store,
    )

    seq = "MSKGEELFTG"
    # Run 1: Cache Miss
    assess1, pdb1 = validator.validate_candidate(seq, variant_id="CAND_1")
    assert "cached" not in assess1.metadata

    # Run 2: Cache Hit
    assess2, pdb2 = validator.validate_candidate(seq, variant_id="CAND_1")
    assert assess2.metadata.get("cached") is True
    assert assess2.mean_plddt == assess1.mean_plddt
    assert pdb1 == pdb2


def test_structural_validator_set_reference_from_file_and_batch(tmp_path):
    wt = "MSKGEELFTG"
    pred = MockStructurePredictor().predict_structure(wt, variant_id="WT_FILE")
    ref_file = tmp_path / "ref_wt.pdb"
    ref_file.write_text(pred.pdb_text)

    validator = StructuralValidator(
        predictor=MockStructurePredictor(),
        min_plddt=70.0,
    )
    validator.set_reference_from_file(ref_file)

    batch = [
        (wt, "CAND_WT"),
        ("MTKGEELFTG", "CAND_MUT"),
    ]
    results = validator.batch_validate(batch)
    assert len(results) == 2
    assert results[0][0].variant_id == "CAND_WT"
    assert results[1][0].variant_id == "CAND_MUT"


def test_structure_relaxer_behavior():
    from evanthea.structure.relaxation import StructureRelaxer
    relaxer = StructureRelaxer()
    assert isinstance(relaxer.is_openmm_available, bool)

    dummy_pdb = "REMARK DUMMY PDB TEXT\n"
    res_text, was_relaxed = relaxer.relax_pdb_text(dummy_pdb)
    if not relaxer.is_openmm_available:
        assert was_relaxed is False
        assert res_text == dummy_pdb
