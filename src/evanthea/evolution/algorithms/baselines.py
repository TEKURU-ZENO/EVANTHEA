"""Baseline optimization algorithms for benchmark comparisons: Random Search, Greedy Hill Climbing, ESM-Guided."""

from typing import List, Optional, Sequence
import numpy as np

from evanthea.evolution.algorithms.base import EvolutionaryAlgorithm
from evanthea.evolution.individual import Individual
from evanthea.evolution.objectives import MultiObjectiveEvaluator
from evanthea.evolution.population import Population
from evanthea.mutation.proposals import CandidateProposalAPI


class RandomSearchOptimizer(EvolutionaryAlgorithm):
    """Unguided random mutation search baseline."""

    def __init__(
        self,
        evaluator: MultiObjectiveEvaluator,
        proposal_api: CandidateProposalAPI,
        wildtype_sequence: str,
        population_size: int = 50,
        max_generations: int = 20,
        max_evaluations: Optional[int] = None,
        reference_point: Optional[Sequence[float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            name="RandomSearch",
            evaluator=evaluator,
            proposal_api=proposal_api,
            wildtype_sequence=wildtype_sequence,
            population_size=population_size,
            max_generations=max_generations,
            max_evaluations=max_evaluations,
            reference_point=reference_point,
            seed=seed,
        )

    def step(self, population: Population) -> Population:
        """Samples random variants directly from wild-type or random members."""
        gen_idx = population.generation + 1
        new_inds: List[Individual] = []

        # Sample uniform mutations from random parents
        for _ in range(self.population_size):
            parent = self.rng.choice(population.individuals)
            cand_vars = self.proposal_api.propose_candidates(parent.sequence, n_candidates=1)
            if cand_vars:
                ind = Individual(variant=cand_vars[0], generation=gen_idx, parent_ids=[parent.id])
                new_inds.append(ind)

        return Population(individuals=new_inds, generation=gen_idx)


class GreedyHillClimber(EvolutionaryAlgorithm):
    """Greedy local search maintaining the best candidate and moving to improving neighbors."""

    def __init__(
        self,
        evaluator: MultiObjectiveEvaluator,
        proposal_api: CandidateProposalAPI,
        wildtype_sequence: str,
        target_objective_name: str = "predicted_fitness",
        population_size: int = 20,
        max_generations: int = 20,
        max_evaluations: Optional[int] = None,
        reference_point: Optional[Sequence[float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            name="GreedyHillClimbing",
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

    def step(self, population: Population) -> Population:
        """Generates neighborhood variants around the current champion and transitions if improved."""
        gen_idx = population.generation + 1
        champion = population.best_by_objective(self.target_objective_name, maximize=True)
        champ_score = champion.fitness_values.get(self.target_objective_name, -np.inf)

        # Propose neighborhood mutations
        cand_vars = self.proposal_api.propose_candidates(
            parent=champion.sequence,
            n_candidates=self.population_size - 1,
        )
        new_inds: List[Individual] = [champion]

        for v in cand_vars:
            ind = Individual(variant=v, generation=gen_idx, parent_ids=[champion.id])
            new_inds.append(ind)

        # Evaluate offspring
        q_pop = Population(individuals=new_inds, generation=gen_idx)
        self.evaluate_population(q_pop)

        # Pick best
        best_candidate = q_pop.best_by_objective(self.target_objective_name, maximize=True)
        best_score = best_candidate.fitness_values.get(self.target_objective_name, -np.inf)

        if best_score > champ_score:
            # Transition to new local peak
            return Population(individuals=[best_candidate] + new_inds[1:], generation=gen_idx)

        # No improvement: keep champion and exploration pool
        return q_pop


class ESMGuidedSearch(EvolutionaryAlgorithm):
    """Evolutionary walk driven strictly by ESM-2 masked marginal plausibility (Δ log p)."""

    def __init__(
        self,
        evaluator: MultiObjectiveEvaluator,
        proposal_api: CandidateProposalAPI,
        wildtype_sequence: str,
        population_size: int = 50,
        max_generations: int = 20,
        max_evaluations: Optional[int] = None,
        reference_point: Optional[Sequence[float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            name="ESMGuidedSearch",
            evaluator=evaluator,
            proposal_api=proposal_api,
            wildtype_sequence=wildtype_sequence,
            population_size=population_size,
            max_generations=max_generations,
            max_evaluations=max_evaluations,
            reference_point=reference_point,
            seed=seed,
        )

    def step(self, population: Population) -> Population:
        """Filters candidates prioritizing ESM-2 likelihood."""
        gen_idx = population.generation + 1
        best_plm = population.best_by_objective("esm2_plausibility", maximize=True)

        cand_vars = self.proposal_api.propose_candidates(
            parent=best_plm.sequence,
            n_candidates=self.population_size,
        )
        new_inds: List[Individual] = []
        for v in cand_vars:
            ind = Individual(variant=v, generation=gen_idx, parent_ids=[best_plm.id])
            new_inds.append(ind)

        return Population(individuals=new_inds, generation=gen_idx)
