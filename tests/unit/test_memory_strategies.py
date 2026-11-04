"""Unit tests for memory strategies (EvaluationBypasser, ArchiveElitismInjector) and schema properties."""

import numpy as np
import pytest

from evanthea.data.schemas import Mutation, VariantRecord
from evanthea.evolution.individual import Individual
from evanthea.evolution.objectives import MultiObjectiveEvaluator, Objective, ObjectiveDirection
from evanthea.memory.manager import EvolutionaryMemory
from evanthea.memory.schema import MemoryRecord, RedundancyStats
from evanthea.memory.storage import PersistentMemoryStore
from evanthea.memory.strategies import ArchiveElitismInjector, EvaluationBypasser
from evanthea.protein.sequence import compute_sequence_hash


def test_evaluation_bypasser_partition_and_hydrate(tmp_path):
    store = PersistentMemoryStore(db_path=tmp_path / "strat_test.db")
    memory = EvolutionaryMemory(store=store)
    bypasser = EvaluationBypasser(memory=memory)

    # 1. Pre-register a candidate in memory
    rec = MemoryRecord(
        sequence_hash=compute_sequence_hash("MKW"),
        variant_id="HISTORICAL_1",
        sequence="MKW",
        mutations=[Mutation(ref_residue="M", position_1based=1, mut_residue="M")],
        fitness_values={"fit": 9.5},
        objectives_vector=[9.5],
        is_feasible=True,
        generation=1,
    )
    store.save_record(rec)

    # 2. Form test batch
    # ind1: already evaluated
    v1 = VariantRecord(variant_id="V1", protein_id="P", mutations=[], mutated_sequence="AAA")
    ind1 = Individual(variant=v1, fitness_values={"fit": 1.0}, is_evaluated=True)

    # ind2: cached in memory
    v2 = VariantRecord(variant_id="V2", protein_id="P", mutations=[], mutated_sequence="MKW")
    ind2 = Individual(variant=v2)

    # ind3: novel sequence
    v3 = VariantRecord(variant_id="V3", protein_id="P", mutations=[], mutated_sequence="CCC")
    ind3 = Individual(variant=v3)

    # ind4: duplicate of ind3 in same batch
    v4 = VariantRecord(variant_id="V4", protein_id="P", mutations=[], mutated_sequence="CCC")
    ind4 = Individual(variant=v4)

    bypassed, novel_reps = bypasser.partition_and_hydrate([ind1, ind2, ind3, ind4])

    assert len(bypassed) == 3  # ind1 (already eval), ind2 (cached), ind4 (intra-batch dup)
    assert len(novel_reps) == 1  # ind3 (unique novel)
    assert ind2.is_evaluated is True
    assert ind2.fitness_values["fit"] == 9.5
    assert novel_reps[0].sequence == "CCC"


def test_archive_elitism_injector(tmp_path):
    store = PersistentMemoryStore(db_path=tmp_path / "elite_test.db")
    memory = EvolutionaryMemory(store=store)
    injector = ArchiveElitismInjector(memory=memory, max_injection=2)

    # Save 2 Pareto optimal historical candidates
    r1 = MemoryRecord(
        sequence_hash=compute_sequence_hash("AAA"),
        variant_id="ELITE_1",
        sequence="AAA",
        mutations=[],
        fitness_values={"obj1": 10.0, "obj2": 2.0},
        objectives_vector=[10.0, 2.0],
        is_feasible=True,
    )
    r2 = MemoryRecord(
        sequence_hash=compute_sequence_hash("CCC"),
        variant_id="ELITE_2",
        sequence="CCC",
        mutations=[],
        fitness_values={"obj1": 2.0, "obj2": 10.0},
        objectives_vector=[2.0, 10.0],
        is_feasible=True,
    )
    store.save_record(r1)
    store.save_record(r2)

    # Current population has 1 individual, target is 3
    v_curr = VariantRecord(variant_id="CURR_1", protein_id="P", mutations=[], mutated_sequence="GGG")
    ind_curr = Individual(variant=v_curr, is_evaluated=True)

    pop_list = [ind_curr]
    injected = injector.inject_archive_elites(pop_list, target_size=3, generation=5)

    assert len(injected) == 3
    injected_seqs = {ind.sequence for ind in injected}
    assert "GGG" in injected_seqs
    assert "AAA" in injected_seqs
    assert "CCC" in injected_seqs

    # When population is already at target size, no change
    no_change = injector.inject_archive_elites(injected, target_size=3, generation=5)
    assert len(no_change) == 3


def test_redundancy_stats_properties():
    stats = RedundancyStats()
    assert stats.cache_hit_rate == 0.0
    assert stats.effective_exploration_ratio == 0.0

    stats.total_proposals = 100
    stats.cache_hits = 25
    stats.unique_sequences = 75
    assert stats.cache_hit_rate == 0.25
    assert stats.effective_exploration_ratio == 0.75


def test_memory_context_manager_and_errors(tmp_path):
    db_path = tmp_path / "ctx_test.db"
    with EvolutionaryMemory(db_path=db_path) as mem:
        assert mem.get_novelty_score(np.zeros(128, dtype=np.float32)) == 0.0
        # Querying with string sequence when language_model is None raises ValueError
        with pytest.raises(ValueError, match="requires an attached ProteinLanguageModel"):
            mem.query_similar("ACDEF")
