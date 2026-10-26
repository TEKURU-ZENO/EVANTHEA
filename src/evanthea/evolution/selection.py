"""Parent selection operators: Tournament, Roulette-wheel, Rank, and Crowded Comparison."""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from evanthea.evolution.individual import Individual
from evanthea.evolution.pareto import crowded_comparison
from evanthea.evolution.population import Population


class ParentSelector(ABC):
    """Abstract base class for parent selection mechanisms."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def set_seed(self, seed: Optional[int]) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def select(self, population: Population, n_select: int = 2) -> List[Individual]:
        """Selects n_select individuals from the population."""
        raise NotImplementedError


class TournamentSelector(ParentSelector):
    """Tournament selection supporting both NSGA-II crowded comparison and single-objective criteria."""

    def __init__(
        self,
        tournament_size: int = 2,
        use_crowding: bool = True,
        single_objective_name: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(seed=seed)
        if tournament_size < 1:
            raise ValueError(f"tournament_size must be >= 1, got {tournament_size}")
        self.tournament_size = tournament_size
        self.use_crowding = use_crowding
        self.single_objective_name = single_objective_name

    def select(self, population: Population, n_select: int = 2) -> List[Individual]:
        if len(population) == 0:
            raise ValueError("Cannot select from an empty population.")

        pop_inds = population.individuals
        selected: List[Individual] = []

        for _ in range(n_select):
            # Sample tournament candidates
            sample_size = min(self.tournament_size, len(pop_inds))
            candidate_indices = self.rng.choice(len(pop_inds), size=sample_size, replace=False)
            candidates = [pop_inds[idx] for idx in candidate_indices]

            # Determine tournament winner
            winner = candidates[0]
            for challenger in candidates[1:]:
                if self.use_crowding:
                    # NSGA-II crowded comparison operator
                    cmp = crowded_comparison(challenger, winner)
                    if cmp < 0:  # challenger is strictly better
                        winner = challenger
                elif self.single_objective_name is not None:
                    # Single-objective maximization
                    w_score = winner.fitness_values.get(self.single_objective_name, -np.inf)
                    c_score = challenger.fitness_values.get(self.single_objective_name, -np.inf)
                    if c_score > w_score:
                        winner = challenger
                else:
                    # Default: first objective in objectives_vector
                    if challenger.objectives_vector and winner.objectives_vector:
                        if challenger.objectives_vector[0] > winner.objectives_vector[0]:
                            winner = challenger

            selected.append(winner)

        return selected


class RouletteWheelSelector(ParentSelector):
    """Fitness-proportionate roulette wheel selection for single-objective optimization."""

    def __init__(
        self,
        objective_name: str,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(seed=seed)
        self.objective_name = objective_name

    def select(self, population: Population, n_select: int = 2) -> List[Individual]:
        if len(population) == 0:
            raise ValueError("Cannot select from an empty population.")

        pop_inds = population.individuals
        raw_scores = np.array([
            ind.fitness_values.get(self.objective_name, 0.0)
            for ind in pop_inds
        ], dtype=np.float64)

        # Shift to strictly positive probabilities
        min_score = np.min(raw_scores)
        shifted = (raw_scores - min_score) + 1e-4
        probs = shifted / np.sum(shifted)

        chosen_indices = self.rng.choice(len(pop_inds), size=n_select, replace=True, p=probs)
        return [pop_inds[idx] for idx in chosen_indices]


class RandomSelector(ParentSelector):
    """Uniform random selection baseline."""

    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__(seed=seed)

    def select(self, population: Population, n_select: int = 2) -> List[Individual]:
        if len(population) == 0:
            raise ValueError("Cannot select from an empty population.")
        chosen_indices = self.rng.choice(len(population), size=n_select, replace=True)
        return [population[idx] for idx in chosen_indices]
