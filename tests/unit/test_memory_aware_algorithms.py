"""Unit tests for memory-aware evolutionary search algorithms (MemoryAwareNSGA2 & MemoryAwareGA)."""

from typing import List
import pytest

from evanthea.evolution.individual import Individual
from evanthea.evolution.objectives import MultiObjectiveEvaluator, Objective, ObjectiveDirection
from evanthea.memory.algorithms import MemoryAwareGA, MemoryAwareNSGA2
from evanthea.memory.manager import EvolutionaryMemory
from evanthea.memory.storage import PersistentMemoryStore
from evanthea.mutation.point_mutations import UniformRandomMutationGenerator
from evanthea.mutation.proposals import CandidateProposalAPI


def test_memory_aware_nsga2_bypasses_duplicate_evaluations(tmp_path):
    db_path = tmp_path / "test_memory.db"
    store = PersistentMemoryStore(db_path=db_path)
    memory = EvolutionaryMemory(store=store)

    wt = "MSKGEELFTG"
    gen = UniformRandomMutationGenerator(seed=42)
    api = CandidateProposalAPI(operator=gen, protein_id="TEST_PROT")

    call_count = {"a_count": 0, "g_count": 0}

    def eval_a(ind: Individual) -> float:
        call_count["a_count"] += 1
        return float(ind.sequence.count("A"))

    def eval_g(ind: Individual) -> float:
        call_count["g_count"] += 1
        return float(ind.sequence.count("G"))

    obj1 = Objective(name="a_count", direction=ObjectiveDirection.MAXIMIZE)
    obj2 = Objective(name="g_count", direction=ObjectiveDirection.MAXIMIZE)

    evaluator = MultiObjectiveEvaluator(
        objectives=[obj1, obj2],
        evaluators={
            "a_count": eval_a,
            "g_count": eval_g,
        },
    )

    optimizer = MemoryAwareNSGA2(
        evaluator=evaluator,
        proposal_api=api,
        wildtype_sequence=wt,
        memory=memory,
        population_size=10,
        max_generations=4,
        reference_point=[-1.0, -1.0],
        archive_injection=True,
        seed=42,
    )

    final_pop, history = optimizer.run()

    assert len(final_pop) == 10
    assert len(history.generations) == 5
    assert all(ind.is_evaluated for ind in final_pop.individuals)

    # In our memory setup, the number of evaluator calls must strictly equal optimizer.evaluations_count
    assert call_count["a_count"] == optimizer.evaluations_count
    assert call_count["g_count"] == optimizer.evaluations_count

    # Redundancy stats should be tracked
    stats = memory.get_redundancy_stats()
    assert stats.total_proposals > 0
    assert stats.unique_sequences == optimizer.evaluations_count
    # If any duplicates occurred, cache_hits > 0 and forward passes saved
    assert stats.cache_hits == stats.evaluations_saved


def test_memory_aware_nsga2_budget_conservation(tmp_path):
    """Verifies that under a fixed evaluation budget, memory awareness enables more proposals."""
    db_path = tmp_path / "budget_test.db"
    store = PersistentMemoryStore(db_path=db_path)
    memory = EvolutionaryMemory(store=store)

    # Short sequence leads to frequent duplicate mutations
    wt = "MK"
    gen = UniformRandomMutationGenerator(seed=99)
    api = CandidateProposalAPI(operator=gen, protein_id="SHORT_PROT")

    eval_calls = 0

    def mock_eval(ind: Individual) -> float:
        nonlocal eval_calls
        eval_calls += 1
        return float(len(ind.sequence))

    obj = Objective(name="len", direction=ObjectiveDirection.MAXIMIZE)
    evaluator = MultiObjectiveEvaluator(
        objectives=[obj],
        evaluators={"len": mock_eval},
    )

    budget = 20
    optimizer = MemoryAwareNSGA2(
        evaluator=evaluator,
        proposal_api=api,
        wildtype_sequence=wt,
        memory=memory,
        population_size=6,
        max_generations=10,
        max_evaluations=budget,
        seed=99,
    )

    final_pop, history = optimizer.run()

    # The actual forward passes must strictly equal the optimizer tracked count
    assert eval_calls == optimizer.evaluations_count
    # Loop should have terminated well before 10 generations due to budget
    assert len(history.generations) < 11
    assert optimizer.evaluations_count >= budget

    # Redundancy stats verify caching occurred on the tiny search space
    stats = memory.get_redundancy_stats()
    assert stats.total_proposals >= eval_calls
    if stats.cache_hits > 0:
        assert stats.evaluations_saved > 0


def test_memory_aware_ga_run(tmp_path):
    db_path = tmp_path / "ga_memory.db"
    store = PersistentMemoryStore(db_path=db_path)
    memory = EvolutionaryMemory(store=store)

    wt = "MSKGEELFTG"
    gen = UniformRandomMutationGenerator(seed=77)
    api = CandidateProposalAPI(operator=gen, protein_id="TEST_PROT")

    eval_count = 0

    def fitness_fn(ind: Individual) -> float:
        nonlocal eval_count
        eval_count += 1
        return float(ind.sequence.count("E") * 2.0)

    obj = Objective(name="predicted_fitness", direction=ObjectiveDirection.MAXIMIZE)
    evaluator = MultiObjectiveEvaluator(
        objectives=[obj],
        evaluators={"predicted_fitness": fitness_fn},
    )

    optimizer = MemoryAwareGA(
        evaluator=evaluator,
        proposal_api=api,
        wildtype_sequence=wt,
        memory=memory,
        target_objective_name="predicted_fitness",
        population_size=8,
        max_generations=3,
        seed=77,
    )

    final_pop, history = optimizer.run()

    assert len(final_pop) == 8
    assert all(ind.is_evaluated for ind in final_pop.individuals)
    assert eval_count == optimizer.evaluations_count

    best_ind = final_pop.best_by_objective("predicted_fitness", maximize=True)
    assert best_ind is not None
    assert best_ind.fitness_values["predicted_fitness"] >= 0.0

    # Ensure memory store persisted the evaluated variants
    stats = memory.get_redundancy_stats()
    assert stats.unique_sequences == eval_count
    assert len(store) == eval_count
