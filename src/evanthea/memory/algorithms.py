"""Memory-aware evolutionary algorithms integrating zero-cost evaluation bypassing and archive elitism."""

from typing import List, Optional, Sequence
import numpy as np

from evanthea.evolution.algorithms.genetic_algorithm import SingleObjectiveGA
from evanthea.evolution.algorithms.nsga2 import NSGA2Optimizer
from evanthea.evolution.crossover import CrossoverOperator
from evanthea.evolution.individual import Individual
from evanthea.evolution.objectives import MultiObjectiveEvaluator
from evanthea.evolution.pareto import calculate_crowding_distance, fast_non_dominated_sort
from evanthea.evolution.population import Population
from evanthea.evolution.selection import ParentSelector
from evanthea.memory.manager import EvolutionaryMemory
from evanthea.memory.strategies import ArchiveElitismInjector, EvaluationBypasser
from evanthea.mutation.proposals import CandidateProposalAPI


class MemoryAwareNSGA2(NSGA2Optimizer):
    """NSGA-II optimizer integrating persistent evolutionary memory for redundant evaluation elimination."""

    def __init__(
        self,
        evaluator: MultiObjectiveEvaluator,
        proposal_api: CandidateProposalAPI,
        wildtype_sequence: str,
        memory: EvolutionaryMemory,
        population_size: int = 50,
        max_generations: int = 20,
        max_evaluations: Optional[int] = None,
        crossover_rate: float = 0.5,
        crossover_operator: Optional[CrossoverOperator] = None,
        selector: Optional[ParentSelector] = None,
        reference_point: Optional[Sequence[float]] = None,
        archive_injection: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            evaluator=evaluator,
            proposal_api=proposal_api,
            wildtype_sequence=wildtype_sequence,
            population_size=population_size,
            max_generations=max_generations,
            max_evaluations=max_evaluations,
            crossover_rate=crossover_rate,
            crossover_operator=crossover_operator,
            selector=selector,
            reference_point=reference_point,
            seed=seed,
        )
        self.name = "MemoryAwareNSGA-II"
        self.memory = memory
        self.bypasser = EvaluationBypasser(memory=memory)
        self.archive_injector = ArchiveElitismInjector(memory=memory) if archive_injection else None

    def evaluate_population(self, population: Population) -> None:
        """Evaluates population through memory bypasser, only charging budget for novel sequences."""
        unevaluated = [ind for ind in population.individuals if not ind.is_evaluated]
        if unevaluated:
            new_evals = self.bypasser.evaluate_with_memory(unevaluated, self.evaluator)
            self.evaluations_count += new_evals
            for ind in unevaluated:
                self.lineage_tracker.record_individual(ind)

    def step(self, population: Population) -> Population:
        """Executes one generation of Memory-Aware NSGA-II."""
        n = self.population_size

        # 1. Non-dominated sort current population
        fronts = fast_non_dominated_sort(population.individuals)
        for front in fronts:
            calculate_crowding_distance(front)

        # 2. Generate offspring Q_t of size N
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

        # 3. Evaluate offspring through memory bypasser
        q_pop = Population(individuals=offspring_list, generation=gen_idx)
        self.evaluate_population(q_pop)

        # 4. Optional global archive injection
        if self.archive_injector is not None and gen_idx > 2:
            offspring_list = self.archive_injector.inject_archive_elites(
                population_individuals=offspring_list,
                target_size=n,
                generation=gen_idx,
            )

        # 5. Combine R_t = P_t ∪ Q_t (size 2N)
        r_individuals = population.individuals + offspring_list
        r_fronts = fast_non_dominated_sort(r_individuals)

        # 6. Elitist survivor selection with crowding distance
        next_individuals: List[Individual] = []
        for front in r_fronts:
            calculate_crowding_distance(front)
            needed = n - len(next_individuals)

            if len(front) <= needed:
                next_individuals.extend(front)
            else:
                sorted_front = sorted(
                    front,
                    key=lambda ind: ind.crowding_distance,
                    reverse=True,
                )
                next_individuals.extend(sorted_front[:needed])
                break

        return Population(individuals=next_individuals, generation=gen_idx)


class MemoryAwareGA(SingleObjectiveGA):
    """Single-objective Genetic Algorithm with memory bypassing for redundant evaluation prevention."""

    def __init__(
        self,
        evaluator: MultiObjectiveEvaluator,
        proposal_api: CandidateProposalAPI,
        wildtype_sequence: str,
        memory: EvolutionaryMemory,
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
            evaluator=evaluator,
            proposal_api=proposal_api,
            wildtype_sequence=wildtype_sequence,
            target_objective_name=target_objective_name,
            population_size=population_size,
            max_generations=max_generations,
            max_evaluations=max_evaluations,
            elitism_ratio=elitism_ratio,
            crossover_rate=crossover_rate,
            crossover_operator=crossover_operator,
            selector=selector,
            reference_point=reference_point,
            seed=seed,
        )
        self.name = "MemoryAwareGA"
        self.memory = memory
        self.bypasser = EvaluationBypasser(memory=memory)

    def evaluate_population(self, population: Population) -> None:
        """Evaluates population through memory bypasser, only charging budget for novel sequences."""
        unevaluated = [ind for ind in population.individuals if not ind.is_evaluated]
        if unevaluated:
            new_evals = self.bypasser.evaluate_with_memory(unevaluated, self.evaluator)
            self.evaluations_count += new_evals
            for ind in unevaluated:
                self.lineage_tracker.record_individual(ind)
