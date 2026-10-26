"""Evolutionary history tracking, generation logs, and performance trajectories."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from evanthea.evolution.individual import Individual


class GenerationRecord(BaseModel):
    """Snapshot of population metrics at the end of a single evolutionary generation."""
    generation: int
    population_size: int
    unique_sequences: int
    pareto_front_size: int
    pairwise_diversity: float
    hypervolume: Optional[float] = None
    spacing_metric: Optional[float] = None
    objective_metrics: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    best_by_objective: Dict[str, float] = Field(default_factory=dict)
    elapsed_time_s: float = 0.0


class EvolutionHistory(BaseModel):
    """Container tracking the complete history, metrics, and Pareto snapshots across generations."""
    algorithm_name: str
    generations: List[GenerationRecord] = Field(default_factory=list)
    total_evaluations: int = 0
    total_runtime_s: float = 0.0
    final_pareto_front: List[Individual] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_generation(self, record: GenerationRecord) -> None:
        """Appends a generation record to history."""
        self.generations.append(record)

    def get_trajectory(self, objective_name: str, statistic: str = "max") -> List[float]:
        """Extracts numerical trajectory of a specific objective metric over generations."""
        vals: List[float] = []
        for g in self.generations:
            if objective_name in g.objective_metrics and statistic in g.objective_metrics[objective_name]:
                vals.append(g.objective_metrics[objective_name][statistic])
            elif statistic == "best" and objective_name in g.best_by_objective:
                vals.append(g.best_by_objective[objective_name])
        return vals

    def get_hypervolume_trajectory(self) -> List[float]:
        """Returns list of hypervolume values per generation."""
        return [g.hypervolume for g in self.generations if g.hypervolume is not None]

    def get_diversity_trajectory(self) -> List[float]:
        """Returns list of pairwise diversity values per generation."""
        return [g.pairwise_diversity for g in self.generations]

    def save_json(self, path: Path | str) -> None:
        """Saves history to disk in JSON format."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
