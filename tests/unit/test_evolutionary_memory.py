"""Unit tests for EvolutionaryMemory central coordinator."""

from pathlib import Path
import pytest
import numpy as np

from evanthea.data.schemas import Mutation, VariantRecord
from evanthea.evolution.individual import Individual
from evanthea.memory.manager import EvolutionaryMemory
from evanthea.protein.language_model import MockProteinLanguageModel


def _make_individual(seq: str, var_id: str, fit: float = 1.0, objs: list[float] | None = None) -> Individual:
    v = VariantRecord(
        variant_id=var_id,
        protein_id="GFP",
        mutations=[],
        mutated_sequence=seq,
        is_wildtype=True,
    )
    return Individual(
        variant=v,
        fitness_values={"score": fit},
        objectives_vector=objs or [fit],
        is_evaluated=True,
    )


def test_evolutionary_memory_registration_and_lookup():
    memory = EvolutionaryMemory(db_path=":memory:", embedding_dim=4, use_faiss=False)

    ind = _make_individual("MSKGE", "v1", fit=2.5, objs=[2.5])
    fake_emb = [1.0, 0.0, 0.0, 0.0]

    rec = memory.register_candidate(ind, embedding=fake_emb)
    assert rec.variant_id == "v1"
    assert rec.sequence == "MSKGE"
    assert memory.has_sequence("MSKGE") is True

    # First lookup: hit
    cached = memory.lookup_exact("MSKGE")
    assert cached is not None
    assert cached.variant_id == "v1"
    assert cached.fitness_values["score"] == 2.5

    # Lookup of absent sequence: miss
    miss = memory.lookup_exact("AAAAA")
    assert miss is None

    stats = memory.stats
    assert stats.total_proposals == 2
    assert stats.cache_hits == 1
    assert stats.evaluations_saved == 1
    assert stats.cache_hit_rate == 0.5
    assert stats.unique_sequences == 1

    memory.close()


def test_evolutionary_memory_query_similar():
    mock_plm = MockProteinLanguageModel(hidden_dim=4)
    memory = EvolutionaryMemory(db_path=":memory:", embedding_dim=4, language_model=mock_plm, use_faiss=False)

    # Register two candidates with specific embeddings
    ind1 = _make_individual("MSKGE", "v1", fit=1.0)
    ind2 = _make_individual("ASKGE", "v2", fit=2.0)

    memory.register_candidate(ind1, embedding=[1.0, 0.0, 0.0, 0.0])
    memory.register_candidate(ind2, embedding=[0.0, 1.0, 0.0, 0.0])

    # Query with vector close to ind1
    results = memory.query_similar([0.9, 0.1, 0.0, 0.0], top_k=2, metric="cosine")
    assert len(results) == 2
    assert results[0].variant_id == "v1"
    assert results[0].similarity_score > 0.8


def test_evolutionary_memory_global_pareto_archive():
    memory = EvolutionaryMemory(db_path=":memory:", embedding_dim=4, use_faiss=False)

    # 3 candidates:
    # c1: [5.0, 1.0] -> non-dominated
    # c2: [1.0, 5.0] -> non-dominated
    # c3: [2.0, 0.5] -> strictly dominated by c1 ([5.0 >= 2.0, 1.0 >= 0.5])
    c1 = _make_individual("A", "c1", objs=[5.0, 1.0])
    c2 = _make_individual("C", "c2", objs=[1.0, 5.0])
    c3 = _make_individual("D", "c3", objs=[2.0, 0.5])

    memory.register_candidate(c1)
    memory.register_candidate(c2)
    memory.register_candidate(c3)

    archive = memory.get_global_pareto_archive()
    assert len(archive) == 2
    arch_ids = {r.variant_id for r in archive}
    assert arch_ids == {"c1", "c2"}
