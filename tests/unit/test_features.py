"""Unit tests for feature matrix construction, biophysical descriptors, and feature scaling."""

import numpy as np
import pytest
from evanthea.data.schemas import Mutation, ProteinRecord, VariantRecord
from evanthea.predictors.features import (
    FeatureConfig,
    FeatureMatrixBuilder,
    FeatureScaler,
    compute_biophysical_deltas,
)
from evanthea.protein.sequence import compute_sequence_hash


def _create_mock_variant(vid: str, wt_seq: str, mut_tuples: list[tuple[str, int, str]]) -> VariantRecord:
    mutations = [Mutation(ref_residue=r, position_1based=p, mut_residue=m) for r, p, m in mut_tuples]
    seq_chars = list(wt_seq)
    for m in mutations:
        seq_chars[m.position_0based] = m.mut_residue
    return VariantRecord(
        variant_id=vid,
        protein_id="GFP",
        mutations=mutations,
        mutated_sequence="".join(seq_chars),
        is_wildtype=len(mutations) == 0,
        num_mutations=len(mutations),
    )


def test_compute_biophysical_deltas():
    """Biophysical deltas accurately calculate charge, hydropathy, and volume changes."""
    # Wild-type has 0 delta
    d_charge, d_hydro, d_vol = compute_biophysical_deltas([])
    assert d_charge == 0.0
    assert d_hydro == 0.0
    assert d_vol == 0.0

    # K (+1.0) to E (-1.0) -> charge change is -2.0
    mut_ke = [Mutation(ref_residue="K", position_1based=3, mut_residue="E")]
    d_charge, d_hydro, d_vol = compute_biophysical_deltas(mut_ke)
    assert pytest.approx(d_charge) == -2.0
    # K hydropathy -3.9, E hydropathy -3.5 -> delta is (-3.5) - (-3.9) = +0.4
    assert pytest.approx(d_hydro) == 0.4
    # K vol 168.6, E vol 138.4 -> delta is 138.4 - 168.6 = -30.2
    assert pytest.approx(d_vol) == -30.2


def test_feature_matrix_builder_configurations():
    """Validates output shapes and exact arithmetic across Configs A, B, C, D, E."""
    wt_seq = "MSKGEELFTG"
    protein = ProteinRecord(id="GFP", name="GFP", wildtype_sequence=wt_seq, length=10)

    # 3 variants: WT, single mutant, double mutant
    v_wt = _create_mock_variant("v_wt", wt_seq, [])
    v_s = _create_mock_variant("v_s", wt_seq, [("M", 1, "A")])
    v_d = _create_mock_variant("v_d", wt_seq, [("M", 1, "A"), ("G", 4, "A")])
    variants = [v_wt, v_s, v_d]

    # Mock embeddings of dimension D = 16
    D = 16
    wt_hash = compute_sequence_hash(wt_seq)
    s_hash = compute_sequence_hash(v_s.mutated_sequence)
    d_hash = compute_sequence_hash(v_d.mutated_sequence)

    wt_emb = np.ones(D, dtype=np.float32) * 1.0
    s_emb = np.ones(D, dtype=np.float32) * 2.0
    d_emb = np.ones(D, dtype=np.float32) * 3.0

    mock_emb_dict = {wt_hash: wt_emb, s_hash: s_emb, d_hash: d_emb}

    builder = FeatureMatrixBuilder(reference_protein=protein, wt_embedding=wt_emb)

    # 1. Config A: Sequence Embeddings (D)
    X_a, ids_a = builder.build_matrix(variants, config=FeatureConfig.SEQ_EMB, provided_embeddings=mock_emb_dict)
    assert X_a.shape == (3, D)
    assert ids_a == ["v_wt", "v_s", "v_d"]
    np.testing.assert_allclose(X_a[0], wt_emb)

    # 2. Config B: Embedding Differences (D)
    X_b, _ = builder.build_matrix(variants, config=FeatureConfig.EMB_DIFF, provided_embeddings=mock_emb_dict)
    assert X_b.shape == (3, D)
    np.testing.assert_allclose(X_b[0], np.zeros(D))       # WT - WT = 0
    np.testing.assert_allclose(X_b[1], np.ones(D) * 1.0)   # s_emb (2.0) - wt (1.0) = 1.0
    np.testing.assert_allclose(X_b[2], np.ones(D) * 2.0)   # d_emb (3.0) - wt (1.0) = 2.0

    # 3. Config C: Concatenated Embeddings (2D)
    X_c, _ = builder.build_matrix(variants, config=FeatureConfig.EMB_CONCAT, provided_embeddings=mock_emb_dict)
    assert X_c.shape == (3, 2 * D)
    np.testing.assert_allclose(X_c[1, :D], s_emb)
    np.testing.assert_allclose(X_c[1, D:], s_emb - wt_emb)

    # 4. Config D: Mutation Features (6)
    X_d, _ = builder.build_matrix(variants, config=FeatureConfig.MUTATION_FEATURES, provided_embeddings=mock_emb_dict)
    assert X_d.shape == (3, 6)
    # Check num_mutations column (index 0)
    assert list(X_d[:, 0]) == [0.0, 1.0, 2.0]

    # 5. Config E: Hybrid (D + 6)
    X_e, _ = builder.build_matrix(variants, config=FeatureConfig.HYBRID, provided_embeddings=mock_emb_dict)
    assert X_e.shape == (3, D + 6)
    np.testing.assert_allclose(X_e[1, :D], s_emb - wt_emb)
    np.testing.assert_allclose(X_e[1, D:], X_d[1])


def test_feature_scaler():
    """FeatureScaler standardizes training matrix and applies learned scaling to test."""
    X_train = np.array([[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]], dtype=np.float32)
    X_test = np.array([[2.0, 15.0]], dtype=np.float32)

    scaler = FeatureScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    assert pytest.approx(np.mean(X_train_scaled, axis=0)) == [0.0, 0.0]
    assert pytest.approx(np.std(X_train_scaled, axis=0)) == [1.0, 1.0]

    X_test_scaled = scaler.transform(X_test)
    assert X_test_scaled.shape == (1, 2)
