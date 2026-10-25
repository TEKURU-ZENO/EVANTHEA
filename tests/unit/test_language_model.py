"""Unit tests for ProteinLanguageModel interface, data structures, and validation rules."""

import pytest
import torch
from evanthea.data.schemas import Mutation
from evanthea.exceptions import CoordinateOutOfBoundsError, SequenceMismatchError
from evanthea.protein.language_model import (
    CombinatorialScoreResult,
    MockProteinLanguageModel,
    MutationScoreResult,
    ProteinLanguageModel,
    SequenceEmbeddingResult,
)


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
