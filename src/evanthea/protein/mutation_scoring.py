"""High-level mutation scoring, masked marginal calculation, and plausibility estimation."""

from typing import List, Literal, Optional
from evanthea.data.schemas import Mutation, VariantRecord
from evanthea.protein.language_model import (
    CombinatorialScoreResult,
    MutationScoreResult,
    ProteinLanguageModel,
)


class MutationScorer:
    """Evaluates sequence substitutions using a ProteinLanguageModel backend."""

    def __init__(self, plm: ProteinLanguageModel):
        self.plm = plm

    def score_single_mutations(
        self,
        reference_sequence: str,
        mutations: List[Mutation],
        method: Literal["masked_marginal", "wildtype_marginal"] = "masked_marginal",
    ) -> List[MutationScoreResult]:
        """Calculates zero-shot fitness log-odds for a collection of point mutations."""
        return self.plm.score_mutations(reference_sequence, mutations, method=method)

    def score_variant(
        self,
        reference_sequence: str,
        variant: VariantRecord,
        method: Literal["additive", "joint_masked"] = "additive",
    ) -> CombinatorialScoreResult:
        """Scores a variant record containing one or more mutations."""
        return self.plm.score_combinatorial_variant(
            reference_sequence, variant.mutations, method=method
        )

    def compute_pseudo_perplexity(self, sequence: str) -> float:
        """Calculates pseudo-perplexity PLL of a full sequence."""
        return self.plm.score_sequence_plausibility(sequence)
