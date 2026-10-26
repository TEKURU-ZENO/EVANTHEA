"""Unit tests for SingleObjectiveGA algorithm."""

import pytest

from evanthea.evolution.algorithms.genetic_algorithm import SingleObjectiveGA
from evanthea.evolution.objectives import MultiObjectiveEvaluator, Objective, ObjectiveDirection
from evanthea.mutation.point_mutations import UniformRandomMutationGenerator
from evanthea.mutation.proposals import CandidateProposalAPI


def test_single_objective_ga_run():
    wt = "MSKGEELFTG"
    gen = UniformRandomMutationGenerator(seed=42)
    api = CandidateProposalAPI(operator=gen, protein_id="TEST_PROT")

    # Objective: favor sequence with more 'A' residues
    obj = Objective(name="a_count", direction=ObjectiveDirection.MAXIMIZE)
    evaluator = MultiObjectiveEvaluator(
        objectives=[obj],
        evaluators={"a_count": lambda ind: float(ind.sequence.count("A"))},
    )

    ga = SingleObjectiveGA(
        evaluator=evaluator,
        proposal_api=api,
        wildtype_sequence=wt,
        target_objective_name="a_count",
        population_size=10,
        max_generations=3,
        elitism_ratio=0.1,
        seed=42,
    )

    final_pop, history = ga.run()

    assert len(final_pop) == 10
    assert len(history.generations) == 4  # Gen 0 + 3 steps
    assert history.total_evaluations > 0

    # Best individual in final population should have non-negative fitness
    best_ind = final_pop.best_by_objective("a_count", maximize=True)
    assert best_ind.fitness_values["a_count"] >= 0.0

    # Trajectory should have values for all 4 generations
    traj = history.get_trajectory("a_count", statistic="max")
    assert len(traj) == 4
