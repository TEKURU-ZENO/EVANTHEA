"""Combinatorial multi-mutant generation engine supporting independent and sequential sampling."""

import time
from typing import Any, Dict, List, Literal, Optional, Set
import numpy as np

from evanthea.constants import STANDARD_AMINO_ACIDS
from evanthea.data.schemas import Mutation
from evanthea.exceptions import CoordinateOutOfBoundsError
from evanthea.mutation.base import MutationOperator, ProposalRecord
from evanthea.mutation.point_mutations import (
    BLOSUM62MutationGenerator,
    ESM2PointMutationGenerator,
    UniformRandomMutationGenerator,
    _validate_allowed_positions,
)
from evanthea.protein.language_model import ProteinLanguageModel
from evanthea.protein.sequence import (
    apply_mutations,
    compute_sequence_hash,
    validate_amino_acid_string,
)


class CombinatorialMutationEngine(MutationOperator):
    """Generates multi-point mutant candidates via independent or sequential context-aware sampling.

    Features:
    - Degree sampling: 'fixed', 'uniform', or 'poisson'
    - Proposal dynamics:
      - 'independent': samples k positions simultaneously from parent context
      - 'sequential': context-aware walk updating sequence between substitutions (epistatic shifts)
    - Strict position collision resolution: ensures all k positions are mutually distinct
    - Lineage scoring: tracks individual step proposal scores and final holistic variant score
    """

    def __init__(
        self,
        point_generator: MutationOperator,
        degree_mode: Literal["fixed", "uniform", "poisson"] = "uniform",
        min_degree: int = 1,
        max_degree: int = 3,
        poisson_lambda: float = 2.0,
        proposal_mode: Literal["independent", "sequential"] = "sequential",
        language_model: Optional[ProteinLanguageModel] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(name=f"combinatorial_{point_generator.name}", seed=seed)
        if min_degree < 1:
            raise ValueError(f"min_degree must be >= 1, got {min_degree}")
        if max_degree < min_degree:
            raise ValueError(f"max_degree ({max_degree}) must be >= min_degree ({min_degree})")
        if poisson_lambda <= 0:
            raise ValueError(f"poisson_lambda must be > 0, got {poisson_lambda}")

        self.point_generator = point_generator
        self.degree_mode = degree_mode
        self.min_degree = min_degree
        self.max_degree = max_degree
        self.poisson_lambda = poisson_lambda
        self.proposal_mode = proposal_mode
        self.language_model = language_model

        # Synchronize RNG seed across components
        if seed is not None:
            self.point_generator.set_seed(seed)

    def _sample_degree(self) -> int:
        """Samples number of simultaneous substitutions k."""
        if self.degree_mode == "fixed":
            return self.max_degree
        elif self.degree_mode == "uniform":
            return int(self.rng.integers(self.min_degree, self.max_degree + 1))
        elif self.degree_mode == "poisson":
            sampled = int(self.rng.poisson(self.poisson_lambda))
            return int(np.clip(sampled, self.min_degree, self.max_degree))
        return self.min_degree

    def propose(
        self,
        parent_sequence: str,
        n_proposals: int = 1,
        allowed_positions: Optional[Set[int]] = None,
        **kwargs: Any,
    ) -> List[ProposalRecord]:
        ref_seq = validate_amino_acid_string(parent_sequence)
        seq_len = len(ref_seq)
        pos_list = _validate_allowed_positions(seq_len, allowed_positions)
        parent_hash = compute_sequence_hash(ref_seq)

        if len(pos_list) < self.min_degree:
            raise ValueError(
                f"Number of allowed positions ({len(pos_list)}) is less than minimum mutation degree ({self.min_degree})."
            )

        records: List[ProposalRecord] = []
        for _ in range(n_proposals):
            start_t = time.perf_counter()

            # Determine mutation count k
            k = self._sample_degree()
            k = min(k, len(pos_list))  # Cannot exceed total allowed positions

            if self.proposal_mode == "independent":
                # 1. Independent proposal: select k distinct positions
                chosen_positions = set(self.rng.choice(pos_list, size=k, replace=False).tolist())
                mutations: List[Mutation] = []
                step_scores: List[float] = []

                # Propose mutation for each chosen position independently against parent
                for pos_1b in chosen_positions:
                    sub_prop = self.point_generator.propose(
                        parent_sequence=ref_seq,
                        n_proposals=1,
                        allowed_positions={pos_1b},
                    )[0]
                    mut = sub_prop.mutations[0]
                    mutations.append(mut)
                    if sub_prop.step_proposal_scores:
                        step_scores.append(sub_prop.step_proposal_scores[0])

            else:
                # 2. Sequential context-aware proposal
                remaining_positions = set(pos_list)
                current_seq = ref_seq
                mutations = []
                step_scores = []

                for _ in range(k):
                    if not remaining_positions:
                        break

                    # Propose point mutation in the updated sequence context
                    sub_prop = self.point_generator.propose(
                        parent_sequence=current_seq,
                        n_proposals=1,
                        allowed_positions=remaining_positions,
                    )[0]

                    sub_mut = sub_prop.mutations[0]
                    pos_1b = sub_mut.position_1based
                    mut_aa = sub_mut.mut_residue
                    wt_original_aa = ref_seq[pos_1b - 1]

                    # Record mutation relative to original parent
                    orig_mut = Mutation(
                        ref_residue=wt_original_aa,
                        position_1based=pos_1b,
                        mut_residue=mut_aa,
                    )
                    mutations.append(orig_mut)

                    if sub_prop.step_proposal_scores:
                        step_scores.append(sub_prop.step_proposal_scores[0])

                    # Update context for next iteration and remove position
                    remaining_positions.remove(pos_1b)
                    current_seq = sub_prop.child_sequence

            # Sort mutations deterministically by sequence position
            mutations.sort(key=lambda m: m.position_1based)
            child_seq = apply_mutations(ref_seq, mutations)
            child_hash = compute_sequence_hash(child_seq)

            # Compute final variant score
            if self.language_model is not None and len(mutations) > 0:
                try:
                    comb_res = self.language_model.score_combinatorial_variant(
                        ref_seq,
                        mutations,
                        method="additive",
                    )
                    final_score = float(comb_res.score)
                except Exception:
                    final_score = float(sum(step_scores)) if step_scores else None
            else:
                final_score = float(sum(step_scores)) if step_scores else None

            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            records.append(
                ProposalRecord(
                    parent_hash=parent_hash,
                    child_sequence=child_seq,
                    child_hash=child_hash,
                    mutations=mutations,
                    generator_name=self.name,
                    position_strategy=f"{self.proposal_mode}_{self.degree_mode}",
                    seed=self.seed,
                    step_proposal_scores=step_scores,
                    final_variant_score=final_score,
                    is_accepted=True,
                    rejection_reasons=[],
                    generation_time_ms=elapsed_ms,
                    metadata={
                        "degree_k": len(mutations),
                        "proposal_mode": self.proposal_mode,
                        "degree_mode": self.degree_mode,
                    },
                )
            )

        return records
