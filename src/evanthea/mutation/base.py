"""Base interfaces and data schemas for protein mutation generation and provenance tracking."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from evanthea.data.schemas import Mutation, VariantRecord
from evanthea.protein.sequence import (
    apply_mutations,
    compute_sequence_hash,
    validate_amino_acid_string,
)


class ProposalRecord(BaseModel):
    """Immutable audit record detailing the origin, sampling settings, and validation of a proposed variant."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    parent_hash: str = Field(..., description="Deterministic SHA-256 hash of parent amino acid sequence")
    child_sequence: str = Field(..., min_length=1, description="Reconstructed variant sequence")
    child_hash: str = Field(..., description="Deterministic SHA-256 hash of child amino acid sequence")
    mutations: List[Mutation] = Field(default_factory=list, description="Ordered list of applied point mutations")
    generator_name: str = Field(..., description="Identifier of generator (e.g. uniform_random, blosum62, esm2)")
    temperature: Optional[float] = Field(None, description="Sampling temperature parameter T")
    top_k: Optional[int] = Field(None, description="Top-k token truncation parameter")
    top_p: Optional[float] = Field(None, description="Nucleus top-p probability threshold")
    position_strategy: str = Field("uniform", description="Position selection strategy (uniform, entropy, conservation)")
    seed: Optional[int] = Field(None, description="RNG seed used for deterministic proposal generation")
    step_proposal_scores: List[float] = Field(
        default_factory=list,
        description="Plausibility / log-odds scores computed at each sampling step"
    )
    final_variant_score: Optional[float] = Field(
        None,
        description="Holistic plausibility score computed over the complete candidate sequence"
    )
    is_accepted: bool = Field(True, description="Whether the candidate passed all biological constraints")
    rejection_reasons: List[str] = Field(
        default_factory=list,
        description="Explicit biological explanations if candidate was rejected"
    )
    generation_time_ms: Optional[float] = Field(None, description="Elapsed latency in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary generator-specific metadata")

    @property
    def mutation_string(self) -> str:
        """Standard HGVS/ProteinGym format e.g. 'M1A:G2C' or 'WT'."""
        if not self.mutations:
            return "WT"
        return ":".join(m.to_string() for m in self.mutations)

    def to_variant_record(
        self,
        protein_id: str,
        variant_id: Optional[str] = None,
        raw_score: Optional[float] = None,
        normalized_score: Optional[float] = None,
    ) -> VariantRecord:
        """Converts proposal into standard EVANTHEA VariantRecord with embedded provenance."""
        v_id = variant_id if variant_id is not None else self.child_hash[:16]
        meta = dict(self.metadata)
        meta["proposal"] = self.model_dump(mode="json")

        return VariantRecord(
            variant_id=v_id,
            protein_id=protein_id,
            mutations=self.mutations,
            mutated_sequence=self.child_sequence,
            raw_score=raw_score,
            normalized_score=normalized_score,
            is_wildtype=len(self.mutations) == 0,
            num_mutations=len(self.mutations),
            metadata=meta,
        )


class MutationOperator(ABC):
    """Abstract base class for all single- and multi-point mutation generators."""

    def __init__(self, name: str, seed: Optional[int] = None) -> None:
        self.name = name
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def set_seed(self, seed: Optional[int]) -> None:
        """Re-seeds the operator's random number generator for reproducibility."""
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def propose(
        self,
        parent_sequence: str,
        n_proposals: int = 1,
        allowed_positions: Optional[Set[int]] = None,
        **kwargs: Any,
    ) -> List[ProposalRecord]:
        """Proposes one or more candidate variants from a parent sequence.

        Args:
            parent_sequence: Canonical parent amino acid sequence.
            n_proposals: Number of candidate variants to propose.
            allowed_positions: Optional set of 1-based positions eligible for mutation.
            **kwargs: Additional generator-specific parameters.

        Returns:
            List of ProposalRecord instances detailing the proposed mutations and scores.
        """
        raise NotImplementedError
