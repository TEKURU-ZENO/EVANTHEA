"""Batch embedding extraction, sequence bucketing, and representation management."""

from typing import List, Optional
import torch
from evanthea.protein.language_model import ProteinLanguageModel, SequenceEmbeddingResult


class EmbeddingExtractor:
    """Manages batch extraction of sequence and residue embeddings with dynamic bucketing."""

    def __init__(self, plm: ProteinLanguageModel, default_batch_size: int = 16):
        self.plm = plm
        self.default_batch_size = default_batch_size

    def extract(self, sequences: List[str], batch_size: Optional[int] = None) -> SequenceEmbeddingResult:
        """Extracts embeddings for a list of sequences."""
        bs = batch_size or self.default_batch_size
        return self.plm.embed(sequences, batch_size=bs)

    @staticmethod
    def bucket_by_length(sequences: List[str]) -> List[List[str]]:
        """Buckets sequences by length to minimize padding during batch inference."""
        sorted_indices = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
        # Return sorted sequences
        return [sequences[i] for i in sorted_indices]
