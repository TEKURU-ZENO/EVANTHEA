"""Individual entity data model wrapping protein variants with multi-objective fitness and Pareto metrics."""

from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from evanthea.data.schemas import Mutation, VariantRecord


class Individual(BaseModel):
    """Represents a single candidate in an evolutionary population with multi-objective scores."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    variant: VariantRecord = Field(..., description="Underlying protein variant record")
    fitness_values: Dict[str, float] = Field(
        default_factory=dict,
        description="Raw evaluated objective scores mapping objective_name -> score"
    )
    objectives_vector: Optional[List[float]] = Field(
        None,
        description="Normalized objective vector aligned for maximization"
    )
    rank: Optional[int] = Field(
        None,
        description="Pareto non-dominated front rank (0 = first front / Pareto optimal)"
    )
    crowding_distance: float = Field(
        0.0,
        description="Crowding distance metric in objective space"
    )
    generation: int = Field(
        0,
        ge=0,
        description="Generation in which this candidate was created"
    )
    parent_ids: List[str] = Field(
        default_factory=list,
        description="Variant IDs of parent individual(s)"
    )
    is_evaluated: bool = Field(
        False,
        description="Whether fitness values have been computed"
    )
    is_feasible: bool = Field(
        True,
        description="Whether individual satisfies all biological and feasibility constraints"
    )
    feasibility_violations: List[str] = Field(
        default_factory=list,
        description="Detailed list of violated constraints if infeasible"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary evolutionary metadata"
    )

    @property
    def id(self) -> str:
        """Unique variant identifier."""
        return self.variant.variant_id

    @property
    def sequence(self) -> str:
        """Full amino acid sequence."""
        return self.variant.mutated_sequence

    @property
    def mutations(self) -> List[Mutation]:
        """Ordered list of substitutions relative to wild-type."""
        return self.variant.mutations

    @property
    def mutation_string(self) -> str:
        """Standard mutation string notation e.g. 'M1A:G2C' or 'WT'."""
        return self.variant.mutation_string

    @property
    def num_mutations(self) -> int:
        """Count of substitutions."""
        return len(self.variant.mutations)

    def dominates(self, other: "Individual") -> bool:
        """Returns True if this individual Pareto-dominates the other individual.

        Requires:
        - For all objectives m: self.objectives_vector[m] >= other.objectives_vector[m]
        - For at least one objective m: self.objectives_vector[m] > other.objectives_vector[m]
        """
        if self.objectives_vector is None or other.objectives_vector is None:
            raise ValueError("Cannot evaluate Pareto dominance before computing objectives_vector.")

        v1 = np.asarray(self.objectives_vector, dtype=np.float64)
        v2 = np.asarray(other.objectives_vector, dtype=np.float64)

        all_greater_or_equal = np.all(v1 >= v2 - 1e-9)
        at_least_one_strictly_greater = np.any(v1 > v2 + 1e-9)

        return bool(all_greater_or_equal and at_least_one_strictly_greater)

    def get_objective(self, name: str) -> float:
        """Returns raw score for a named objective."""
        if name not in self.fitness_values:
            raise KeyError(f"Objective '{name}' not found in evaluated fitness values: {list(self.fitness_values.keys())}")
        return self.fitness_values[name]
