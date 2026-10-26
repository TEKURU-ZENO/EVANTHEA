"""Non-Dominated Sorting Genetic Algorithm II (NSGA-II) for multi-objective protein evolution."""

from typing import List, Optional, Sequence
import numpy as np

from evanthea.evolution.algorithms.base import EvolutionaryAlgorithm
from evanthea.evolution.crossover import CrossoverOperator, SinglePointCrossover
from evanthea.evolution.individual import Individual
from evanthea.evolution.objectives import MultiObjectiveEvaluator
from evanthea.evolution.pareto import (
    calculate_crowding_distance,
    fast_non_dominated_sort,
)
from evanthea.evolution.population import Population
from evanthea.evolution.selection import ParentSelector, TournamentSelector
from evanthea.mutation.proposals import CandidateProposalAPI


class NSGA2Optimizer(EvolutionaryAlgorithm):
    """Full implementation of NSGA-II with (μ + λ) elitism, non-dominated sorting, and crowding distance."""

    def __init__(
        self,
        evaluator: MultiObjectiveEvaluator,
        proposal_api: CandidateProposalAPI,
        wildtype_sequence: str,
        population_size: int = 50,
        max_generations: int = 20,
        max_evaluations: Optional[int] = None,
        crossover_rate: float = 0.5,
        crossover_operator: Optional[CrossoverOperator] = None,
        selector: Optional[ParentSelector] = None,
        reference_point: Optional[Sequence[float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            name="NSGA-II",
            evaluator=evaluator,
            proposal_api=proposal_api,
            wildtype_sequence=wildtype_sequence,
            population_size=population_size,
            max_generations=max_generations,
            max_evaluations=max_evaluations,
            reference_point=reference_point,
            seed=seed,
        )
        self.crossover_rate = crossover_rate
        self.crossover_operator = crossover_operator or SinglePointCrossover(
            reference_sequence=wildtype_sequence,
            seed=seed,
        )
        self.selector = selector or TournamentSelector(
            tournament_size=2,
            use_crowding=True,
            seed=seed,
        )

    def step(self, population: Population) -> Population:
        """Executes one generation of NSGA-II:
        1. Ensure P_t is sorted and assigned ranks/crowding distances.
        2. Generate offspring Q_t of size N via tournament selection, crossover, and mutation.
        3. Evaluate Q_t.
        4. Combine R_t = P_t ∪ Q_t (size 2N).
        5. Fast non-dominated sort on R_t.
        6. Fill P_{t+1} front by front, using crowding distance for boundary truncation.
        """
        n = self.population_size

        # 1. Ensure current population has ranks and crowding distances computed
        fronts = fast_non_dominated_sort(population.individuals)
        for front in fronts:
            calculate_crowding_distance(front)

        # 2. Generate offspring population Q_t of size N
        offspring_list: List[Individual] = []
        gen_idx = population.generation + 1

        while len(offspring_list) < n:
            if self.rng.random() < self.crossover_rate:
                parents = self.selector.select(population, n_select=2)
                c1, c2 = self.crossover_operator.crossover(parents[0], parents[1], generation=gen_idx)
                for child in [c1, c2]:
                    if len(offspring_list) < n:
                        mut_vars = self.proposal_api.propose_candidates(
                            parent=child.sequence,
                            n_candidates=1,
                        )
                        if mut_vars:
                            ind = Individual(variant=mut_vars[0], generation=gen_idx, parent_ids=[child.id])
                            offspring_list.append(ind)
            else:
                parent = self.selector.select(population, n_select=1)[0]
                mut_vars = self.proposal_api.propose_candidates(
                    parent=parent.sequence,
                    n_candidates=1,
                )
                if mut_vars:
                    ind = Individual(variant=mut_vars[0], generation=gen_idx, parent_ids=[parent.id])
                    offspring_list.append(ind)

        # 3. Evaluate offspring population Q_t
        q_pop = Population(individuals=offspring_list, generation=gen_idx)
        self.evaluate_population(q_pop)

        # 4. Combine R_t = P_t ∪ Q_t (size 2N)
        r_individuals = population.individuals + offspring_list

        # 5. Non-dominated sort on combined pool
        r_fronts = fast_non_dominated_sort(r_individuals)

        # 6. Elitist survivor selection
        next_individuals: List[Individual] = []
        for front in r_fronts:
            calculate_crowding_distance(front)
            needed = n - len(next_individuals)

            if len(front) <= needed:
                next_individuals.extend(front)
            else:
                # Sort front descending by crowding distance to break ties
                sorted_front = sorted(
                    front,
                    key=lambda ind: ind.crowding_distance,
                    reverse=True,
                )
                next_individuals.extend(sorted_front[:needed])
                break

        return Population(individuals=next_individuals, generation=gen_idx)
