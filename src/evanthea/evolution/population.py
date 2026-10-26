"""Population container and diversity tracking for evolutionary protein engineering."""

import itertools
from typing import Any, Dict, Iterator, List, Optional
import numpy as np
from pydantic import BaseModel, Field

from evanthea.evolution.individual import Individual
from evanthea.protein.sequence import compute_hamming_distance


class Population(BaseModel):
    """Collection of candidate protein individuals in an evolutionary generation."""

    individuals: List[Individual] = Field(default_factory=list, description="List of individuals")
    generation: int = Field(0, ge=0, description="Generation index")

    def __len__(self) -> int:
        return len(self.individuals)

    def __iter__(self) -> Iterator[Individual]:
        return iter(self.individuals)

    def __getitem__(self, index: int) -> Individual:
        return self.individuals[index]

    @property
    def size(self) -> int:
        return len(self.individuals)

    def add(self, individual: Individual) -> None:
        """Appends an individual to the population."""
        self.individuals.append(individual)

    def extend(self, individuals: List[Individual]) -> None:
        """Extends population with a list of individuals."""
        self.individuals.extend(individuals)

    def unique_sequences(self) -> List[str]:
        """Returns list of unique amino acid sequences present in the population."""
        return list({ind.sequence for ind in self.individuals})

    def unique_individuals(self) -> List[Individual]:
        """Returns deduplicated individuals (keeping first occurrence of each unique sequence)."""
        seen: set[str] = set()
        deduped: List[Individual] = []
        for ind in self.individuals:
            if ind.sequence not in seen:
                seen.add(ind.sequence)
                deduped.append(ind)
        return deduped

    def compute_pairwise_hamming_diversity(self, max_samples: int = 150) -> float:
        """Computes mean pairwise Hamming distance across population individuals."""
        unique_seqs = self.unique_sequences()
        if len(unique_seqs) <= 1:
            return 0.0

        if len(unique_seqs) > max_samples:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(unique_seqs), size=max_samples, replace=False)
            eval_seqs = [unique_seqs[i] for i in indices]
        else:
            eval_seqs = unique_seqs

        dists = [
            compute_hamming_distance(s1, s2)
            for s1, s2 in itertools.combinations(eval_seqs, 2)
        ]
        return float(np.mean(dists)) if dists else 0.0

    def best_by_objective(self, objective_name: str, maximize: bool = True) -> Individual:
        """Finds the top-performing individual for a specific named objective."""
        if not self.individuals:
            raise ValueError("Cannot select best individual from an empty population.")

        evaluated = [ind for ind in self.individuals if ind.is_evaluated and objective_name in ind.fitness_values]
        if not evaluated:
            raise ValueError(f"No individuals evaluated for objective '{objective_name}'.")

        if maximize:
            return max(evaluated, key=lambda ind: ind.fitness_values[objective_name])
        return min(evaluated, key=lambda ind: ind.fitness_values[objective_name])

    def get_pareto_front(self) -> List[Individual]:
        """Returns all individuals assigned to Rank 0 (the non-dominated Pareto front)."""
        return [ind for ind in self.individuals if ind.rank == 0]

    def summary_statistics(self) -> Dict[str, Any]:
        """Computes summary statistics for evaluated objectives and population diversity."""
        stats: Dict[str, Any] = {
            "generation": self.generation,
            "population_size": len(self.individuals),
            "unique_sequences": len(self.unique_sequences()),
            "pairwise_hamming_diversity": self.compute_pairwise_hamming_diversity(),
            "objectives": {},
        }

        if not self.individuals or not self.individuals[0].is_evaluated:
            return stats

        for obj_name in self.individuals[0].fitness_values.keys():
            vals = [ind.fitness_values[obj_name] for ind in self.individuals if obj_name in ind.fitness_values]
            if vals:
                arr = np.array(vals, dtype=np.float64)
                stats["objectives"][obj_name] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "median": float(np.median(arr)),
                }

        pareto_front = self.get_pareto_front()
        stats["pareto_front_size"] = len(pareto_front)
        return stats
