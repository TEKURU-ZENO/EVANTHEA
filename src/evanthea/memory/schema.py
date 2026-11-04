"""Pydantic V2 schemas representing memory records, similarity search results, and redundancy analytics."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from evanthea.data.schemas import Mutation


class MemoryRecord(BaseModel):
    """Immutable record of an evaluated protein variant stored in evolutionary memory."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sequence_hash: str = Field(..., description="Deterministic SHA-256 hash of the amino acid sequence")
    variant_id: str = Field(..., description="Unique variant identifier")
    sequence: str = Field(..., min_length=1, description="Full canonical amino acid sequence")
    mutations: List[Mutation] = Field(default_factory=list, description="Ordered list of point substitutions")
    fitness_values: Dict[str, float] = Field(
        default_factory=dict,
        description="Evaluated objective scores mapping objective_name -> raw score"
    )
    objectives_vector: Optional[List[float]] = Field(
        None,
        description="Normalized objective vector aligned for maximization"
    )
    is_feasible: bool = Field(True, description="Whether variant passes all biological constraints")
    feasibility_violations: List[str] = Field(
        default_factory=list,
        description="Detailed list of violated constraints if infeasible"
    )
    generation: int = Field(0, ge=0, description="Generation in which this variant was first observed")
    parent_ids: List[str] = Field(default_factory=list, description="IDs of parent variants")
    embedding: Optional[List[float]] = Field(
        None,
        description="ESM-2 sequence-level embedding vector for similarity retrieval"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC timestamp of variant registration"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary provenance, assay, and search metadata"
    )

    @property
    def mutation_string(self) -> str:
        """Returns standard colon-delimited notation e.g. 'M1A:G2C' or 'WT'."""
        if not self.mutations:
            return "WT"
        return ":".join(m.to_string() for m in self.mutations)

    @property
    def num_mutations(self) -> int:
        return len(self.mutations)


class SimilaritySearchResult(BaseModel):
    """Result container for dense vector similarity queries in protein representation space."""
    sequence_hash: str = Field(..., description="Sequence SHA-256 hash of matching candidate")
    variant_id: str = Field(..., description="Variant identifier of matching candidate")
    similarity_score: float = Field(..., description="Similarity metric (e.g. cosine similarity in [-1, 1])")
    distance: float = Field(..., description="Metric distance (e.g. Euclidean L2 distance in [0, inf))")
    fitness_values: Dict[str, float] = Field(
        default_factory=dict,
        description="Cached objective values of the retrieved candidate"
    )
    mutations: List[Mutation] = Field(default_factory=list, description="Substitutions of the matching candidate")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata of retrieved candidate")


class RedundancyStats(BaseModel):
    """Auditing metrics tracking duplicate proposal prevention and compute savings."""
    total_proposals: int = Field(0, ge=0, description="Total proposals generated across the search")
    unique_sequences: int = Field(0, ge=0, description="Distinct unique sequences evaluated")
    cache_hits: int = Field(0, ge=0, description="Number of duplicate proposals intercepted")
    evaluations_saved: int = Field(0, ge=0, description="Number of expensive model forward passes prevented")

    @property
    def cache_hit_rate(self) -> float:
        """Proportion of generated proposals that were duplicates intercepted by memory."""
        if self.total_proposals == 0:
            return 0.0
        return float(self.cache_hits / self.total_proposals)

    @property
    def effective_exploration_ratio(self) -> float:
        """Ratio of unique variants to total proposals."""
        if self.total_proposals == 0:
            return 0.0
        return float(self.unique_sequences / self.total_proposals)
