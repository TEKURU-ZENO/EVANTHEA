"""Unit tests for ProteinLanguageModel interface, data structures, and validation rules."""

import pytest
import torch
from evanthea.data.schemas import Mutation
from evanthea.exceptions import CoordinateOutOfBoundsError, SequenceMismatchError
from evanthea.protein.language_model import (
    CombinatorialScoreResult,
    MutationScoreResult,
    ProteinLanguageModel,
    SequenceEmbeddingResult,
)


class MockProteinLanguageModel(ProteinLanguageModel):
    """Deterministic mock PLM for fast unit testing without downloading 600MB weights."""

    def __init__(self, hidden_dim: int = 128):
        self.hidden_dim = hidden_dim

    def embed(self, sequences: list[str], batch_size: int = 16) -> SequenceEmbeddingResult:
        b = len(sequences)
        pooled = torch.zeros(b, self.hidden_dim)
        res_list = [torch.zeros(len(s), self.hidden_dim) for s in sequences]
        return SequenceEmbeddingResult(
            sequence_embeddings=pooled,
            residue_embeddings=res_list,
            hidden_dim=self.hidden_dim,
            model_name="mock_plm",
        )

    def score_mutations(self, sequence: str, mutations: list[Mutation], method: str = "masked_marginal") -> list[MutationScoreResult]:
        seq_len = len(sequence)
        results = []
        for mut in mutations:
            if mut.position_1based < 1 or mut.position_1based > seq_len:
                raise CoordinateOutOfBoundsError(mut.position_1based, seq_len, mut.to_string())
            actual_wt = sequence[mut.position_0based]
            if actual_wt != mut.ref_residue:
                raise SequenceMismatchError(mut.position_1based, mut.ref_residue, actual_wt, mut.to_string())

            results.append(
                MutationScoreResult(
                    mutation=mut,
                    log_odds_score=-1.5,
                    wt_log_prob=-0.5,
                    mut_log_prob=-2.0,
                    scoring_method=method,
                )
            )
        return results

    def score_combinatorial_variant(self, sequence: str, mutations: list[Mutation], method: str = "additive") -> CombinatorialScoreResult:
        single_scores = self.score_mutations(sequence, mutations)
        total = sum(s.log_odds_score for s in single_scores)
        return CombinatorialScoreResult(
            mutations=mutations,
            variant_sequence=sequence,
            score=total,
            additive_score=total,
            scoring_method=method,
        )

    def score_sequence_plausibility(self, sequence: str) -> float:
        return 4.2


def test_mock_plm_embedding_dimensions():
    plm = MockProteinLanguageModel(hidden_dim=64)
    res = plm.embed(["MSKGE", "MSKGEELFTG"])
    assert res.sequence_embeddings.shape == (2, 64)
    assert len(res.residue_embeddings) == 2
    assert res.residue_embeddings[0].shape == (5, 64)
    assert res.residue_embeddings[1].shape == (10, 64)


def test_mock_plm_scoring_valid():
    plm = MockProteinLanguageModel()
    muts = [Mutation(ref_residue="M", position_1based=1, mut_residue="A")]
    scores = plm.score_mutations("MSKGE", muts)
    assert len(scores) == 1
    assert scores[0].log_odds_score == -1.5
    assert scores[0].mutation.to_string() == "M1A"


def test_mock_plm_scoring_wildtype_mismatch():
    plm = MockProteinLanguageModel()
    muts = [Mutation(ref_residue="L", position_1based=1, mut_residue="A")]
    with pytest.raises(SequenceMismatchError):
        plm.score_mutations("MSKGE", muts)


def test_mock_plm_scoring_out_of_bounds():
    plm = MockProteinLanguageModel()
    muts = [Mutation(ref_residue="M", position_1based=10, mut_residue="A")]
    with pytest.raises(CoordinateOutOfBoundsError):
        plm.score_mutations("MSKGE", muts)
