"""Unit tests for ExactNumpyVectorIndex and FaissVectorIndex."""

from pathlib import Path
import pytest
import numpy as np

from evanthea.memory.vector_index import ExactNumpyVectorIndex, FaissVectorIndex


def test_exact_numpy_vector_index_add_and_count():
    idx = ExactNumpyVectorIndex(dimension=4)
    assert idx.count() == 0

    hashes = ["h1", "h2", "h3"]
    vecs = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)
    meta = [
        {"variant_id": "v1", "fitness_values": {"fit": 1.0}},
        {"variant_id": "v2", "fitness_values": {"fit": 2.0}},
        {"variant_id": "v3", "fitness_values": {"fit": 3.0}},
    ]

    idx.add(hashes, vecs, meta)
    assert idx.count() == 3
    assert idx.has_hash("h1") is True
    assert idx.has_hash("h2") is True
    assert idx.has_hash("h4") is False


def test_exact_numpy_vector_index_cosine_search():
    idx = ExactNumpyVectorIndex(dimension=3)
    hashes = ["h_x", "h_y", "h_neg_x"]
    vecs = np.array([
        [1.0, 0.0, 0.0],   # identical to query
        [0.0, 1.0, 0.0],   # orthogonal
        [-1.0, 0.0, 0.0],  # opposite
    ], dtype=np.float32)

    idx.add(hashes, vecs)

    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    res = idx.search(query, top_k=3, metric="cosine")

    assert len(res) == 3
    # Top match must be h_x with cosine similarity = 1.0 and distance = 0.0
    assert res[0].sequence_hash == "h_x"
    assert res[0].similarity_score == pytest.approx(1.0, abs=1e-5)
    assert res[0].distance == pytest.approx(0.0, abs=1e-5)

    # Second match is orthogonal (sim = 0.0, dist = 1.0)
    assert res[1].sequence_hash == "h_y"
    assert res[1].similarity_score == pytest.approx(0.0, abs=1e-5)

    # Third match is opposite (sim = -1.0, dist = 2.0)
    assert res[2].sequence_hash == "h_neg_x"
    assert res[2].similarity_score == pytest.approx(-1.0, abs=1e-5)


def test_exact_numpy_vector_index_l2_search():
    idx = ExactNumpyVectorIndex(dimension=2)
    hashes = ["p0", "p1", "p2"]
    vecs = np.array([
        [0.0, 0.0],
        [1.0, 1.0],  # dist to [0,0] is sqrt(2) ≈ 1.414
        [3.0, 4.0],  # dist to [0,0] is 5.0
    ], dtype=np.float32)

    idx.add(hashes, vecs)

    query = np.array([0.0, 0.0], dtype=np.float32)
    res = idx.search(query, top_k=3, metric="l2")

    assert len(res) == 3
    assert res[0].sequence_hash == "p0"
    assert res[0].distance == pytest.approx(0.0, abs=1e-5)

    assert res[1].sequence_hash == "p1"
    assert res[1].distance == pytest.approx(np.sqrt(2.0), rel=1e-4)

    assert res[2].sequence_hash == "p2"
    assert res[2].distance == pytest.approx(5.0, rel=1e-4)


def test_exact_numpy_vector_index_persistence(tmp_path: Path):
    idx = ExactNumpyVectorIndex(dimension=3)
    hashes = ["h1", "h2"]
    vecs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    meta = [{"variant_id": "v1"}, {"variant_id": "v2"}]
    idx.add(hashes, vecs, meta)

    file_path = tmp_path / "index.npz"
    idx.save(file_path)
    assert file_path.exists()

    idx_loaded = ExactNumpyVectorIndex(dimension=3)
    idx_loaded.load(file_path)

    assert idx_loaded.count() == 2
    assert idx_loaded.has_hash("h1") is True
    assert idx_loaded.has_hash("h2") is True

    # Search should yield same result
    q = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    res = idx_loaded.search(q, top_k=1, metric="cosine")
    assert res[0].sequence_hash == "h1"
    assert res[0].variant_id == "v1"


def test_faiss_vector_index_fallback_behavior():
    idx = FaissVectorIndex(dimension=3)
    hashes = ["h1", "h2"]
    vecs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    meta = [{"variant_id": "v1"}, {"variant_id": "v2"}]

    idx.add(hashes, vecs, meta)
    assert idx.count() == 2
    assert idx.has_hash("h1") is True

    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    res = idx.search(q, top_k=2, metric="cosine")
    assert len(res) == 2
    assert res[0].sequence_hash == "h1"

    res_l2 = idx.search(q, top_k=2, metric="l2")
    assert len(res_l2) == 2
    assert res_l2[0].sequence_hash == "h1"


def test_vector_index_validation_errors():
    with pytest.raises(ValueError):
        ExactNumpyVectorIndex(dimension=0)

    idx = ExactNumpyVectorIndex(dimension=3)
    with pytest.raises(ValueError):
        idx.add(["h1"], np.array([[1.0, 2.0]], dtype=np.float32))

    with pytest.raises(ValueError):
        idx.add(["h1", "h2"], np.array([[1.0, 2.0, 3.0]], dtype=np.float32))

    with pytest.raises(ValueError):
        idx.search(np.array([1.0, 2.0]), top_k=1)
