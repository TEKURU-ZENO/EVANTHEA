"""Base abstract framework for evolutionary search algorithms with budget tracking and logging."""

from abc import ABC, abstractmethod
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from evanthea.data.schemas import VariantRecord
from evanthea.evolution.history import EvolutionHistory, GenerationRecord
from evanthea.evolution.individual import Individual
from evanthea.evolution.lineage import LineageTracker
from evanthea.evolution.objectives import MultiObjectiveEvaluator
from evanthea.evolution.pareto import (
    calculate_crowding_distance,
    compute_hypervolume,
    compute_spacing_metric,
    fast_non_dominated_sort,
)
from evanthea.evolution.population import Population
from evanthea.mutation.proposals import CandidateProposalAPI


class EvolutionaryAlgorithm(ABC):
    """Abstract base class orchestrating population evolution, budget enforcement, and provenance tracking."""

    def __init__(
        self,
        name: str,
        evaluator: MultiObjectiveEvaluator,
        proposal_api: CandidateProposalAPI,
        wildtype_sequence: str,
        population_size: int = 50,
        max_generations: int = 20,
        max_evaluations: Optional[int] = None,
        reference_point: Optional[Sequence[float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        if population_size < 2:
            raise ValueError(f"population_size must be >= 2, got {population_size}")
        if max_generations < 1:
            raise ValueError(f"max_generations must be >= 1, got {max_generations}")

        self.name = name
        self.evaluator = evaluator
        self.proposal_api = proposal_api
        self.wildtype_sequence = wildtype_sequence
        self.population_size = population_size
        self.max_generations = max_generations
        self.max_evaluations = max_evaluations or (population_size * (max_generations + 1))
        self.reference_point = reference_point or [-5.0] * len(evaluator.objectives)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        if seed is not None and hasattr(self.proposal_api.operator, "set_seed"):
            self.proposal_api.operator.set_seed(seed)

        self.lineage_tracker = LineageTracker()
        self.history = EvolutionHistory(algorithm_name=name)
        self.evaluations_count = 0

    def initialize_population(
        self,
        seed_sequences: Optional[List[str]] = None,
    ) -> Population:
        """Initializes population P_0 from wild-type and optional seed variants."""
        pop = Population(generation=0)

        # 1. Always include wild-type ancestor
        wt_variant = VariantRecord(
            variant_id="WT_ANCESTOR",
            protein_id=self.proposal_api.protein_id,
            mutations=[],
            mutated_sequence=self.wildtype_sequence,
            is_wildtype=True,
            metadata={"origin": "wildtype_ancestor"},
        )
        wt_ind = Individual(variant=wt_variant, generation=0, parent_ids=[])
        pop.add(wt_ind)
        self.lineage_tracker.record_individual(wt_ind, operator="wildtype_seed")

        # 2. Add any user-provided seed variants
        if seed_sequences:
            for i, s in enumerate(seed_sequences):
                if len(pop) >= self.population_size:
                    break
                v = VariantRecord(
                    variant_id=f"SEED_{i+1}",
                    protein_id=self.proposal_api.protein_id,
                    mutations=[],
                    mutated_sequence=s,
                    is_wildtype=False,
                    metadata={"origin": "user_seed"},
                )
                ind = Individual(variant=v, generation=0, parent_ids=["WT_ANCESTOR"])
                pop.add(ind)
                self.lineage_tracker.record_individual(ind, operator="user_seed")

        # 3. Fill remaining slots with single-point proposals from wild-type
        needed = self.population_size - len(pop)
        if needed > 0:
            initial_variants = self.proposal_api.propose_candidates(
                parent=self.wildtype_sequence,
                n_candidates=needed,
            )
            for v in initial_variants:
                ind = Individual(variant=v, generation=0, parent_ids=["WT_ANCESTOR"])
                pop.add(ind)
                self.lineage_tracker.record_individual(ind, operator="initial_proposal")

        # Evaluate initial population
        self.evaluate_population(pop)
        return pop

    def evaluate_population(self, population: Population) -> None:
        """Evaluates all un-evaluated individuals in the population and updates evaluation counter."""
        unevaluated = [ind for ind in population.individuals if not ind.is_evaluated]
        if unevaluated:
            self.evaluator.evaluate_population(unevaluated)
            self.evaluations_count += len(unevaluated)
            for ind in unevaluated:
                self.lineage_tracker.record_individual(ind)

    def log_generation_metrics(
        self,
        population: Population,
        generation: int,
        elapsed_s: float,
    ) -> GenerationRecord:
        """Calculates Pareto fronts, hypervolume, and logs generation summary."""
        # Non-dominated sort
        fronts = fast_non_dominated_sort(population.individuals)
        if fronts and fronts[0]:
            calculate_crowding_distance(fronts[0])
            hv = compute_hypervolume(fronts[0], reference_point=self.reference_point)
            spacing = compute_spacing_metric(fronts[0])
        else:
            hv = 0.0
            spacing = 0.0

        stats = population.summary_statistics()

        best_dict: Dict[str, float] = {}
        for obj in self.evaluator.objectives:
            try:
                best_ind = population.best_by_objective(obj.name, maximize=(obj.direction.value == "maximize"))
                best_dict[obj.name] = best_ind.fitness_values[obj.name]
            except Exception:
                pass

        record = GenerationRecord(
            generation=generation,
            population_size=len(population),
            unique_sequences=stats["unique_sequences"],
            pareto_front_size=len(fronts[0]) if fronts else 0,
            pairwise_diversity=stats["pairwise_hamming_diversity"],
            hypervolume=hv,
            spacing_metric=spacing,
            objective_metrics=stats["objectives"],
            best_by_objective=best_dict,
            elapsed_time_s=elapsed_s,
        )
        self.history.add_generation(record)
        return record

    @abstractmethod
    def step(self, population: Population) -> Population:
        """Executes a single generational evolution step, returning the offspring/survivor population."""
        raise NotImplementedError

    def run(
        self,
        initial_population: Optional[Population] = None,
    ) -> Tuple[Population, EvolutionHistory]:
        """Runs the evolutionary search until max_generations or max_evaluations budget is reached."""
        start_total = time.time()

        if self.seed is not None:
            self.rng = np.random.default_rng(self.seed)
            if hasattr(self.proposal_api.operator, "set_seed"):
                self.proposal_api.operator.set_seed(self.seed)
            if hasattr(self, "selector") and hasattr(self.selector, "set_seed"):
                self.selector.set_seed(self.seed)
            if hasattr(self, "crossover_operator") and hasattr(self.crossover_operator, "set_seed"):
                self.crossover_operator.set_seed(self.seed)

        pop = initial_population or self.initialize_population()
        self.evaluate_population(pop)

        gen = 0
        self.log_generation_metrics(pop, generation=gen, elapsed_s=time.time() - start_total)

        while gen < self.max_generations and self.evaluations_count < self.max_evaluations:
            gen_start = time.time()
            pop = self.step(pop)
            self.evaluate_population(pop)
            gen += 1
            pop.generation = gen
            self.log_generation_metrics(pop, generation=gen, elapsed_s=time.time() - gen_start)

        # Finalize history
        self.history.total_evaluations = self.evaluations_count
        self.history.total_runtime_s = time.time() - start_total
        self.history.final_pareto_front = pop.get_pareto_front()

        return pop, self.history
