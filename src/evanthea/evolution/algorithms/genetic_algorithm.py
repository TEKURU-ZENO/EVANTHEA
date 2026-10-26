"""Single-objective Genetic Algorithm (GA) with elitism, tournament selection, and mutation."""

from typing import List, Optional, Sequence
import numpy as np

from evanthea.evolution.algorithms.base import EvolutionaryAlgorithm
from evanthea.evolution.crossover import CrossoverOperator, SinglePointCrossover
from evanthea.evolution.individual import Individual
from evanthea.evolution.objectives import MultiObjectiveEvaluator
from evanthea.evolution.population import Population
from evanthea.evolution.selection import ParentSelector, TournamentSelector
from evanthea.mutation.proposals import CandidateProposalAPI


class SingleObjectiveGA(EvolutionaryAlgorithm):
    """Classical single-objective Genetic Algorithm with elitist preservation and tournament selection."""

    def __init__(
        self,
        evaluator: MultiObjectiveEvaluator,
        proposal_api: CandidateProposalAPI,
        wildtype_sequence: str,
        target_objective_name: str = "predicted_fitness",
        population_size: int = 50,
        max_generations: int = 20,
        max_evaluations: Optional[int] = None,
        elitism_ratio: float = 0.05,
        crossover_rate: float = 0.2,
        crossover_operator: Optional[CrossoverOperator] = None,
        selector: Optional[ParentSelector] = None,
        reference_point: Optional[Sequence[float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            name="SingleObjectiveGA",
            evaluator=evaluator,
            proposal_api=proposal_api,
            wildtype_sequence=wildtype_sequence,
            population_size=population_size,
            max_generations=max_generations,
            max_evaluations=max_evaluations,
            reference_point=reference_point,
            seed=seed,
        )
        self.target_objective_name = target_objective_name
        self.elitism_ratio = elitism_ratio
        self.crossover_rate = crossover_rate
        self.crossover_operator = crossover_operator or SinglePointCrossover(
            reference_sequence=wildtype_sequence,
            seed=seed,
        )
        self.selector = selector or TournamentSelector(
            tournament_size=2,
            use_crowding=False,
            single_objective_name=target_objective_name,
            seed=seed,
        )

    def step(self, population: Population) -> Population:
        """Executes one generation of selection, crossover, mutation, and elitist replacement."""
        n = self.population_size

        # 1. Elitism: preserve top k individuals directly
        k_elites = max(1, int(np.ceil(self.elitism_ratio * n)))
        sorted_inds = sorted(
            population.individuals,
            key=lambda ind: ind.fitness_values.get(self.target_objective_name, -np.inf),
            reverse=True,
        )
        survivors: List[Individual] = sorted_inds[:k_elites]

        # 2. Fill remaining slots with offspring
        while len(survivors) < n:
            if self.rng.random() < self.crossover_rate:
                # Recombination
                parents = self.selector.select(population, n_select=2)
                c1, c2 = self.crossover_operator.crossover(
                    parents[0],
                    parents[1],
                    generation=population.generation + 1,
                )
                for child in [c1, c2]:
                    if len(survivors) < n:
                        # Apply mutation to recombinant child
                        mut_vars = self.proposal_api.propose_candidates(
                            parent=child.sequence,
                            n_candidates=1,
                        )
                        if mut_vars:
                            offspring = Individual(
                                variant=mut_vars[0],
                                generation=population.generation + 1,
                                parent_ids=[child.id],
                            )
                            survivors.append(offspring)
            else:
                # Direct mutation
                parent = self.selector.select(population, n_select=1)[0]
                mut_vars = self.proposal_api.propose_candidates(
                    parent=parent.sequence,
                    n_candidates=1,
                )
                if mut_vars:
                    offspring = Individual(
                        variant=mut_vars[0],
                        generation=population.generation + 1,
                        parent_ids=[parent.id],
                    )
                    survivors.append(offspring)

        return Population(individuals=survivors, generation=population.generation + 1)
