"""Unit tests for baseline optimization algorithms (Random Search, Greedy Hill Climbing, ESM-Guided)."""

import pytest

from evanthea.evolution.algorithms.baselines import (
    ESMGuidedSearch,
    GreedyHillClimber,
    RandomSearchOptimizer,
)
from evanthea.evolution.objectives import MultiObjectiveEvaluator, Objective, ObjectiveDirection
from evanthea.mutation.point_mutations import UniformRandomMutationGenerator
from evanthea.mutation.proposals import CandidateProposalAPI


def test_random_search_optimizer():
    wt = "MSKGEELFTG"
    gen = UniformRandomMutationGenerator(seed=42)
    api = CandidateProposalAPI(operator=gen, protein_id="TEST")

    evaluator = MultiObjectiveEvaluator(
        objectives=[Objective(name="score", direction=ObjectiveDirection.MAXIMIZE)],
        evaluators={"score": lambda ind: float(len(ind.mutations))},
    )

    opt = RandomSearchOptimizer(
        evaluator=evaluator,
        proposal_api=api,
        wildtype_sequence=wt,
        population_size=8,
        max_generations=2,
        seed=42,
    )

    pop, history = opt.run()
    assert len(pop) == 8
    assert len(history.generations) == 3


def test_greedy_hill_climber():
    wt = "MSKGEELFTG"
    gen = UniformRandomMutationGenerator(seed=99)
    api = CandidateProposalAPI(operator=gen, protein_id="TEST")

    # Score increases with mutation count
    evaluator = MultiObjectiveEvaluator(
        objectives=[Objective(name="mut_count", direction=ObjectiveDirection.MAXIMIZE)],
        evaluators={"mut_count": lambda ind: float(len(ind.mutations))},
    )

    opt = GreedyHillClimber(
        evaluator=evaluator,
        proposal_api=api,
        wildtype_sequence=wt,
        target_objective_name="mut_count",
        population_size=6,
        max_generations=2,
        seed=99,
    )

    pop, history = opt.run()
    assert len(pop) == 6
    champ = pop.best_by_objective("mut_count", maximize=True)
    assert champ.fitness_values["mut_count"] >= 0.0


def test_esm_guided_search():
    wt = "MSKGEELFTG"
    gen = UniformRandomMutationGenerator(seed=12)
    api = CandidateProposalAPI(operator=gen, protein_id="TEST")

    evaluator = MultiObjectiveEvaluator(
        objectives=[Objective(name="esm2_plausibility", direction=ObjectiveDirection.MAXIMIZE)],
        evaluators={"esm2_plausibility": lambda ind: -float(len(ind.mutations))},
    )

    opt = ESMGuidedSearch(
        evaluator=evaluator,
        proposal_api=api,
        wildtype_sequence=wt,
        population_size=6,
        max_generations=2,
        seed=12,
    )

    pop, history = opt.run()
    assert len(pop) == 6
