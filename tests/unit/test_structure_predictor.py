"""Unit tests for structure prediction backends (Mock, ReferenceLoader, ESMFold wrapper)."""

import pytest

from evanthea.structure.predictor import ESMFoldPredictor, MockStructurePredictor, ReferencePDBLoader


def test_mock_structure_predictor():
    seq = "MSKGEELFTG"
    predictor = MockStructurePredictor(base_plddt=85.0)
    res = predictor.predict_structure(seq, variant_id="TEST_MOCK")

    assert res.sequence == seq
    assert len(res.plddt_scores) == len(seq)
    assert 70.0 <= res.mean_plddt <= 95.0
    assert "ATOM" in res.pdb_text
    assert "CA  MET" in res.pdb_text or "CA  GLY" in res.pdb_text
    assert "TER" in res.pdb_text
    assert "END" in res.pdb_text


def test_reference_pdb_loader():
    seq = "MSKGE"
    predictor = MockStructurePredictor()
    pred_res = predictor.predict_structure(seq, variant_id="REF_TEST")

    loader = ReferencePDBLoader()
    struct = loader.load_from_text(pred_res.pdb_text, structure_id="TEST")

    coords, parsed_seq = loader.extract_ca_coordinates(struct)
    assert coords.shape == (5, 3)
    assert parsed_seq == seq


def test_esmfold_predictor_fallback():
    # If no local model weights or network API, falls back cleanly to Mock
    predictor = ESMFoldPredictor(model_name="non_existent/model", use_api_fallback=False)
    assert not predictor.is_local_available

    res = predictor.predict_structure("ACDEF", variant_id="FALLBACK_TEST")
    assert res.sequence == "ACDEF"
    assert "ATOM" in res.pdb_text
    assert "fallback_from" in res.metadata
