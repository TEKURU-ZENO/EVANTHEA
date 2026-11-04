"""Integration test verifying end-to-end evolutionary memory pipeline, persistent multi-session caching, and vector indexing."""

import numpy as np
import pytest

from evanthea.data.schemas import VariantRecord
from evanthea.evolution.individual import Individual
from evanthea.evolution.objectives import MultiObjectiveEvaluator, Objective, ObjectiveDirection
from evanthea.memory.algorithms import MemoryAwareNSGA2
from evanthea.memory.manager import EvolutionaryMemory
from evanthea.memory.storage import PersistentMemoryStore
from evanthea.memory.vector_index import ExactNumpyVectorIndex
from evanthea.mutation.point_mutations import UniformRandomMutationGenerator
from evanthea.mutation.proposals import CandidateProposalAPI


def test_end_to_end_memory_assisted_evolution_and_multi_session_caching(tmp_path):
    """Verifies that evolutionary memory persists across multiple independent optimization runs."""
    db_file = tmp_path / "integrated_evo_memory.db"
    wt = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDAT"  # avGFP N-terminal fragment

    gen = UniformRandomMutationGenerator(seed=101)
    api = CandidateProposalAPI(operator=gen, protein_id="avGFP_FRAG")

    eval_calls_session1 = 0

    def obj_a(ind: Individual) -> float:
        nonlocal eval_calls_session1
        eval_calls_session1 += 1
        return float(ind.sequence.count("E") + ind.sequence.count("D"))

    def obj_b(ind: Individual) -> float:
        return float(ind.sequence.count("G") - ind.sequence.count("V"))

    evaluator1 = MultiObjectiveEvaluator(
        objectives=[
            Objective(name="charge_residues", direction=ObjectiveDirection.MAXIMIZE),
            Objective(name="glycine_balance", direction=ObjectiveDirection.MAXIMIZE),
        ],
        evaluators={
            "charge_residues": obj_a,
            "glycine_balance": obj_b,
        },
    )

    # SESSION 1
    memory1 = EvolutionaryMemory(db_path=db_file)
    opt1 = MemoryAwareNSGA2(
        evaluator=evaluator1,
        proposal_api=api,
        wildtype_sequence=wt,
        memory=memory1,
        population_size=12,
        max_generations=3,
        seed=101,
    )

    final_pop1, history1 = opt1.run()
    assert len(final_pop1) == 12
    assert eval_calls_session1 == opt1.evaluations_count
    n_variants_session1 = memory1.store.count_variants()
    assert n_variants_session1 > 0
    memory1.store.close()

    # SESSION 2: Launch a new optimizer on the same database file
    eval_calls_session2 = 0

    def obj_a_session2(ind: Individual) -> float:
        nonlocal eval_calls_session2
        eval_calls_session2 += 1
        return float(ind.sequence.count("E") + ind.sequence.count("D"))

    evaluator2 = MultiObjectiveEvaluator(
        objectives=[
            Objective(name="charge_residues", direction=ObjectiveDirection.MAXIMIZE),
            Objective(name="glycine_balance", direction=ObjectiveDirection.MAXIMIZE),
        ],
        evaluators={
            "charge_residues": obj_a_session2,
            "glycine_balance": obj_b,
        },
    )

    # Use same seed to generate identical initial proposals
    gen2 = UniformRandomMutationGenerator(seed=101)
    api2 = CandidateProposalAPI(operator=gen2, protein_id="avGFP_FRAG")

    memory2 = EvolutionaryMemory(db_path=db_file)
    opt2 = MemoryAwareNSGA2(
        evaluator=evaluator2,
        proposal_api=api2,
        wildtype_sequence=wt,
        memory=memory2,
        population_size=12,
        max_generations=2,
        seed=101,
    )

    final_pop2, history2 = opt2.run()

    # Session 2 should bypass initial proposals already evaluated in session 1!
    stats2 = memory2.stats
    assert stats2.cache_hits > 0
    assert stats2.evaluations_saved > 0
    assert eval_calls_session2 < opt2.evaluations_count + stats2.evaluations_saved
    assert memory2.store.count_variants() >= n_variants_session1
    memory2.store.close()


def test_vector_similarity_and_novelty_querying(tmp_path):
    """Verifies dense vector similarity search and novelty scoring integrated with EvolutionaryMemory."""
    db_file = tmp_path / "vector_test.db"
    index = ExactNumpyVectorIndex(dimension=8, default_metric="cosine")
    memory = EvolutionaryMemory(db_path=db_file, embedding_dim=8, vector_index=index)

    # Register 5 synthetic candidates with distinct embeddings
    wt_seq = "ACDEFGHIKL"
    for i in range(5):
        seq = wt_seq[:i] + "M" + wt_seq[i+1:]
        emb = np.zeros(8, dtype=np.float32)
        emb[i] = 1.0  # One-hot orthogonal vectors
        var = VariantRecord(
            variant_id=f"MUT_{i}",
            protein_id="SYNTH",
            mutations=[],
            mutated_sequence=seq,
            raw_score=float(i * 1.5),
        )
        memory.register_candidate(var, embedding=emb)

    assert memory.store.count_variants() == 5
    assert len(index) == 5

    # Query with target close to vector 2
    query_emb = np.zeros(8, dtype=np.float32)
    query_emb[2] = 0.95
    query_emb[3] = 0.05

    results = memory.query_similar(query_emb, top_k=2)
    assert len(results) == 2
    assert results[0].variant_id == "MUT_2"
    assert results[0].similarity_score > 0.9

    # Novelty score against distant region
    distant_emb = np.zeros(8, dtype=np.float32)
    distant_emb[7] = 1.0
    novelty = memory.get_novelty_score(distant_emb, k_nearest=2)
    assert novelty >= 0.0

    memory.store.close()
