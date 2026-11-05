"""Integration tests for EVANTHEA Structural & Biophysical Validation Engine (Phase 7)."""

from pathlib import Path
import tempfile
import pytest

from evanthea.memory.storage import PersistentMemoryStore
from evanthea.structure.evaluator import StructuralValidator
from evanthea.structure.predictor import MockStructurePredictor
from evanthea.structure.relaxation import StructureRelaxer
from evanthea.structure.schema import StructuralAssessment


@pytest.fixture
def temp_memory_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_memory.db"
        store = PersistentMemoryStore(db_path=db_path)
        yield store
        store.close()


def test_end_to_end_structural_pipeline_on_gfp(temp_memory_store):
    """Verifies end-to-end structural prediction, relaxation fallback, metric calculation,

    quality gate assessment, and persistence caching on a representative GFP segment.
    """
    wt_sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTT"
    # Variant 1: Single point mutation (S2T)
    var1_seq = "MTKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTT"
    # Variant 2: Triple mutation (S2T, V11I, K26R)
    var2_seq = "MTKGEELFTGIVPILVELDGDVNGHRFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTT"

    predictor = MockStructurePredictor(base_plddt=85.0, seed=42)
    relaxer = StructureRelaxer()

    validator = StructuralValidator(
        predictor=predictor,
        relaxer=relaxer,
        memory_store=temp_memory_store,
        min_plddt=70.0,
        max_rmsd=2.5,
        max_clashes=5,
        min_contact_preservation=0.85,
    )

    # 1. Establish reference using WT
    wt_pred = predictor.predict_structure(wt_sequence, variant_id="WT_REF")
    validator.set_reference_from_text(wt_pred.pdb_text)

    # 2. Validate WT (should have 0.0 RMSD, 1.0 contact preservation, and pass gates)
    wt_assessment, wt_pdb = validator.validate_candidate(wt_sequence, variant_id="WT_EVAL")
    assert isinstance(wt_assessment, StructuralAssessment)
    assert wt_assessment.sequence == wt_sequence
    assert wt_assessment.rmsd_to_reference == pytest.approx(0.0, abs=1e-5)
    assert wt_assessment.contact_preservation_ratio == pytest.approx(1.0, abs=1e-5)
    assert wt_assessment.mean_plddt >= 70.0
    assert wt_assessment.is_structurally_viable is True
    assert len(wt_assessment.violations) == 0
    assert "ATOM" in wt_pdb

    # 3. Validate single mutant
    v1_assessment, v1_pdb = validator.validate_candidate(var1_seq, variant_id="VAR_S2T")
    assert isinstance(v1_assessment, StructuralAssessment)
    assert v1_assessment.sequence == var1_seq
    assert v1_assessment.rmsd_to_reference is not None
    assert v1_assessment.rmsd_to_reference >= 0.0
    assert v1_assessment.contact_preservation_ratio is not None
    assert v1_assessment.contact_preservation_ratio > 0.80

    # 4. Validate triple mutant
    v2_assessment, v2_pdb = validator.validate_candidate(var2_seq, variant_id="VAR_TRIPLE")
    assert isinstance(v2_assessment, StructuralAssessment)
    assert v2_assessment.sequence == var2_seq
    assert v2_assessment.sasa_total is not None
    assert v2_assessment.sasa_hydrophobic is not None
    assert v2_assessment.sasa_hydrophobic <= v2_assessment.sasa_total

    # 5. Verify caching: re-validating var1_seq should load from SQLite without calling predictor
    cached_assessment, cached_pdb = validator.validate_candidate(var1_seq, variant_id="VAR_S2T_RETRY")
    assert cached_assessment.sequence_hash == v1_assessment.sequence_hash
    assert cached_assessment.mean_plddt == pytest.approx(v1_assessment.mean_plddt, abs=1e-4)
    assert cached_assessment.rmsd_to_reference == pytest.approx(v1_assessment.rmsd_to_reference, abs=1e-4)
    assert cached_assessment.clash_count == v1_assessment.clash_count
    assert cached_pdb == v1_pdb
