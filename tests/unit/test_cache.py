"""Unit tests for the SQLite + HDF5 two-tier persistent cache."""

import shutil
from pathlib import Path
import numpy as np
import pytest

from evanthea.data.cache import EmbeddingCache

TEST_CACHE_DIR = Path("data/interim/test_cache")


@pytest.fixture
def cache():
    if TEST_CACHE_DIR.exists():
        shutil.rmtree(TEST_CACHE_DIR)
    c = EmbeddingCache(TEST_CACHE_DIR)
    yield c
    if TEST_CACHE_DIR.exists():
        shutil.rmtree(TEST_CACHE_DIR)


def test_embedding_cache_roundtrip(cache: EmbeddingCache):
    """Embeddings and metadata are correctly persisted and retrieved without loss."""
    seq_hash = "abc123hash"
    dummy_emb = np.random.randn(640).astype(np.float32)

    assert not cache.has_sequence(seq_hash)

    cache.put_embedding(
        seq_hash=seq_hash,
        embedding=dummy_emb,
        length=238,
        model_name="esm2_150m",
        masked_marginal_score=-1.25,
        pseudo_perplexity=4.82,
        extra_metadata={"mutation": "M1A"},
    )

    assert cache.has_sequence(seq_hash)

    meta = cache.get_metadata(seq_hash)
    assert meta is not None
    assert meta["length"] == 238
    assert meta["model_name"] == "esm2_150m"
    assert pytest.approx(meta["masked_marginal_score"], rel=1e-3) == -1.25
    assert pytest.approx(meta["pseudo_perplexity"], rel=1e-3) == 4.82
    assert meta["metadata"]["mutation"] == "M1A"

    retrieved_emb = cache.get_embedding(seq_hash)
    assert retrieved_emb is not None
    # float16 precision check
    np.testing.assert_allclose(retrieved_emb, dummy_emb, atol=1e-2)
