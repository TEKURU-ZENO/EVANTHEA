"""Unified proposal generation API integrating operators, constraints, and lineage tracking."""

from typing import Any, Dict, List, Optional, Set, Tuple, Union

from evanthea.data.schemas import VariantRecord
from evanthea.mutation.base import MutationOperator, ProposalRecord
from evanthea.mutation.constraints import ConstraintPipeline
from evanthea.protein.sequence import validate_amino_acid_string


class CandidateProposalAPI:
    """High-level facade orchestrating variant proposal generation, constraint filtering, and provenance tracking.

    This interface is directly called by downstream evolutionary search algorithms (Phase 4).
    """

    def __init__(
        self,
        operator: MutationOperator,
        protein_id: str,
        constraint_pipeline: Optional[ConstraintPipeline] = None,
        max_resample_factor: int = 5,
    ) -> None:
        self.operator = operator
        self.protein_id = protein_id
        self.constraint_pipeline = constraint_pipeline or ConstraintPipeline()
        self.max_resample_factor = max_resample_factor

    def propose_with_audit(
        self,
        parent: Union[VariantRecord, str],
        n_proposals: int = 1,
        allowed_positions: Optional[Set[int]] = None,
        **kwargs: Any,
    ) -> Tuple[List[ProposalRecord], List[ProposalRecord]]:
        """Generates proposals and applies constraint pipeline, returning accepted and rejected subsets."""
        parent_seq = parent.mutated_sequence if isinstance(parent, VariantRecord) else parent
        parent_seq = validate_amino_acid_string(parent_seq)

        raw_props = self.operator.propose(
            parent_sequence=parent_seq,
            n_proposals=n_proposals,
            allowed_positions=allowed_positions,
            **kwargs,
        )

        return self.constraint_pipeline.filter_proposals(raw_props, parent_sequence=parent_seq)

    def propose_candidates(
        self,
        parent: Union[VariantRecord, str],
        n_candidates: int = 1,
        allowed_positions: Optional[Set[int]] = None,
        **kwargs: Any,
    ) -> List[VariantRecord]:
        """Proposes strictly valid, constraint-passing VariantRecords for downstream evolutionary search.

        If constraints reject proposals, automatically resamples up to max_resample_factor * n_candidates attempts.
        """
        parent_seq = parent.mutated_sequence if isinstance(parent, VariantRecord) else parent
        parent_seq = validate_amino_acid_string(parent_seq)

        accepted_variants: List[VariantRecord] = []
        max_total_proposals = self.max_resample_factor * n_candidates
        proposals_generated = 0

        while len(accepted_variants) < n_candidates and proposals_generated < max_total_proposals:
            batch_size = min(n_candidates - len(accepted_variants), 50)
            proposals_generated += batch_size

            accepted, _ = self.propose_with_audit(
                parent=parent_seq,
                n_proposals=batch_size,
                allowed_positions=allowed_positions,
                **kwargs,
            )

            for prop in accepted:
                v_rec = prop.to_variant_record(protein_id=self.protein_id)
                accepted_variants.append(v_rec)
                if len(accepted_variants) == n_candidates:
                    break

        return accepted_variants
