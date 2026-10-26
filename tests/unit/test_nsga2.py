"""Unit tests for NSGA2Optimizer multi-objective evolution algorithm."""

import pytest

from evanthea.evolution.algorithms.nsga2 import NSGA2Optimizer
from evanthea.evolution.objectives import MultiObjectiveEvaluator, Objective, ObjectiveDirection
from evanthea.mutation.point_mutations import UniformRandomMutationGenerator
from evanthea.mutation.proposals import CandidateProposalAPI


def test_nsga2_optimizer_run():
    wt = "MSKGEELFTG"
    gen = UniformRandomMutationGenerator(seed=123)
    api = CandidateProposalAPI(operator=gen, protein_id="TEST_PROT")

    # Competing objectives:
    # 1. 'a_count': count of 'A'
    # 2. 'g_count': count of 'G'
    obj1 = Objective(name="a_count", direction=ObjectiveDirection.MAXIMIZE)
    obj2 = Objective(name="g_count", direction=ObjectiveDirection.MAXIMIZE)

    evaluator = MultiObjectiveEvaluator(
        objectives=[obj1, obj2],
        evaluators={
            "a_count": lambda ind: float(ind.sequence.count("A")),
            "g_count": lambda ind: float(ind.sequence.count("G")),
        },
    )

    optimizer = NSGA2Optimizer(
        evaluator=evaluator,
        proposal_api=api,
        wildtype_sequence=wt,
        population_size=12,
        max_generations=3,
        reference_point=[-1.0, -1.0],
        seed=123,
    )

    final_pop, history = optimizer.run()

    assert len(final_pop) == 12
    assert len(history.generations) == 4
    assert history.total_evaluations > 0

    # Pareto front
    pareto_front = final_pop.get_pareto_front()
    assert len(pareto_front) >= 1
    for ind in pareto_front:
        assert ind.rank == 0
        assert ind.crowding_distance >= 0.0

    # Hypervolume trajectory
    hv_traj = history.get_hypervolume_trajectory()
    assert len(hv_traj) == 4
    assert all(h >= 0.0 for h in hv_traj)


def test_nsga2_reproducibility():
    wt = "MSKGEELFTG"
    gen = UniformRandomMutationGenerator(seed=77)
    api = CandidateProposalAPI(operator=gen, protein_id="TEST_PROT")

    evaluator = MultiObjectiveEvaluator(
        objectives=[Objective(name="len", direction=ObjectiveDirection.MAXIMIZE)],
        evaluators={"len": lambda ind: float(len(ind.sequence))},
    )

    opt1 = NSGA2Optimizer(evaluator, api, wt, population_size=6, max_generations=2, seed=77)
    opt2 = NSGA2Optimizer(evaluator, api, wt, population_size=6, max_generations=2, seed=77)

    pop1, _ = opt1.run()
    pop2, _ = opt2.run()

    assert [ind.sequence for ind in pop1] == [ind.sequence for ind in pop2]
