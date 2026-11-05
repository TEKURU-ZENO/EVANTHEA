"""Unified Evolutionary Memory manager coordinating SQLite persistence, vector similarity indexing, and redundancy analytics."""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union
import numpy as np

from evanthea.data.schemas import VariantRecord
from evanthea.evolution.individual import Individual
from evanthea.memory.schema import MemoryRecord, RedundancyStats, SimilaritySearchResult
from evanthea.memory.storage import PersistentMemoryStore
from evanthea.memory.vector_index import ExactNumpyVectorIndex, FaissVectorIndex, VectorIndex
from evanthea.protein.language_model import ProteinLanguageModel
from evanthea.protein.sequence import compute_sequence_hash, validate_amino_acid_string


class EvolutionaryMemory:
    """Central evolutionary memory controller managing persistent storage, similarity indexing, and search auditing."""

    def __init__(
        self,
        db_path: Path | str = ":memory:",
        store: Optional[PersistentMemoryStore] = None,
        embedding_dim: int = 128,
        vector_index: Optional[VectorIndex] = None,
        language_model: Optional[ProteinLanguageModel] = None,
        use_faiss: bool = True,
    ) -> None:
        if store is not None:
            self.store = store
        else:
            self.store = PersistentMemoryStore(db_path=db_path)
        self.embedding_dim = embedding_dim
        self.language_model = language_model

        if vector_index is not None:
            self.vector_index = vector_index
        elif use_faiss:
            self.vector_index = FaissVectorIndex(dimension=embedding_dim)
        else:
            self.vector_index = ExactNumpyVectorIndex(dimension=embedding_dim)

        self._stats = RedundancyStats()

    @property
    def stats(self) -> RedundancyStats:
        """Returns current live redundancy statistics."""
        self._stats.unique_sequences = self.store.count_variants()
        return self._stats

    def get_redundancy_stats(self) -> RedundancyStats:
        """Returns current live redundancy statistics (alias for stats property)."""
        return self.stats

    def has_sequence(self, sequence_or_hash: str) -> bool:
        """Checks if a sequence or sequence hash is already stored in memory."""
        if len(sequence_or_hash) == 64 and all(c in "0123456789abcdefABCDEF" for c in sequence_or_hash):
            seq_hash = sequence_or_hash.lower()
        else:
            seq_hash = compute_sequence_hash(sequence_or_hash)
        return self.store.has_sequence(seq_hash)

    def lookup_exact(self, sequence_or_hash: str) -> Optional[MemoryRecord]:
        """Performs O(1) exact lookup in memory. Increments redundancy counters upon cache hit."""
        self._stats.total_proposals += 1

        if len(sequence_or_hash) == 64 and all(c in "0123456789abcdefABCDEF" for c in sequence_or_hash):
            seq_hash = sequence_or_hash.lower()
        else:
            seq_hash = compute_sequence_hash(sequence_or_hash)

        record = self.store.get_record(seq_hash)
        if record is not None and record.fitness_values:
            self._stats.cache_hits += 1
            self._stats.evaluations_saved += 1
            return record

        return None

    def register_candidate(
        self,
        candidate: Union[Individual, VariantRecord, MemoryRecord],
        embedding: Optional[Sequence[float] | np.ndarray] = None,
    ) -> MemoryRecord:
        """Stores a candidate and its evaluated properties in persistent storage and vector index."""
        if isinstance(candidate, MemoryRecord):
            rec = candidate
            emb_vec = embedding if embedding is not None else candidate.embedding
        elif isinstance(candidate, Individual):
            seq_hash = compute_sequence_hash(candidate.sequence)
            rec = MemoryRecord(
                sequence_hash=seq_hash,
                variant_id=candidate.id,
                sequence=candidate.sequence,
                mutations=candidate.mutations,
                fitness_values=dict(candidate.fitness_values),
                objectives_vector=list(candidate.objectives_vector) if candidate.objectives_vector else None,
                is_feasible=candidate.is_feasible,
                feasibility_violations=candidate.feasibility_violations,
                generation=candidate.generation,
                parent_ids=candidate.parent_ids,
                metadata=candidate.metadata,
            )
            emb_vec = embedding
        elif isinstance(candidate, VariantRecord):
            seq_hash = compute_sequence_hash(candidate.mutated_sequence)
            rec = MemoryRecord(
                sequence_hash=seq_hash,
                variant_id=candidate.variant_id,
                sequence=candidate.mutated_sequence,
                mutations=candidate.mutations,
                fitness_values={"raw_score": candidate.raw_score} if candidate.raw_score is not None else {},
                is_feasible=True,
                generation=0,
                metadata=candidate.metadata,
            )
            emb_vec = embedding
        else:
            raise TypeError(f"Unsupported candidate type: {type(candidate)}")

        # If embedding is not provided but PLM is attached, extract embedding
        if emb_vec is None and self.language_model is not None:
            try:
                emb_res = self.language_model.embed([rec.sequence])
                emb_vec = emb_res.sequence_embeddings[0].cpu().numpy().tolist()
            except Exception:
                emb_vec = None

        if emb_vec is not None:
            arr_emb = np.asarray(emb_vec, dtype=np.float32).flatten()
            if len(arr_emb) == self.embedding_dim:
                rec.embedding = arr_emb.tolist()
                self.vector_index.add(
                    hashes=[rec.sequence_hash],
                    vectors=arr_emb,
                    metadata=[{
                        "variant_id": rec.variant_id,
                        "fitness_values": rec.fitness_values,
                        "mutations": [m.model_dump() for m in rec.mutations],
                    }],
                )

        self.store.save_record(rec)
        return rec

    def query_similar(
        self,
        query: Union[str, np.ndarray, Sequence[float]],
        top_k: int = 5,
        metric: Literal["cosine", "l2"] = "cosine",
    ) -> List[SimilaritySearchResult]:
        """Retrieves top-k structurally / representationally similar historical variants."""
        if isinstance(query, str):
            if self.language_model is None:
                raise ValueError("Querying by amino acid sequence string requires an attached ProteinLanguageModel.")
            cleaned = validate_amino_acid_string(query)
            emb_res = self.language_model.embed([cleaned])
            q_vec = emb_res.sequence_embeddings[0].cpu().numpy()
        else:
            q_vec = np.asarray(query, dtype=np.float32).flatten()

        return self.vector_index.search(q_vec, top_k=top_k, metric=metric)

    def get_novelty_score(
        self,
        query: Union[str, np.ndarray, Sequence[float]],
        k_nearest: int = 5,
        metric: Literal["cosine", "l2"] = "cosine",
    ) -> float:
        """Computes representational novelty score as the mean distance to k-nearest neighbors in memory.

        Args:
            query: Query sequence string or embedding vector.
            k_nearest: Number of nearest neighbors to average over.
            metric: Distance metric to use ('cosine' or 'l2').

        Returns:
            Mean distance to the k-nearest neighbors (higher values indicate greater novelty).
        """
        try:
            results = self.query_similar(query, top_k=k_nearest, metric=metric)
        except ValueError:
            return 0.0

        if not results:
            return 0.0

        distances = [r.distance for r in results]
        return float(np.mean(distances))

    def get_global_pareto_archive(self) -> List[MemoryRecord]:
        """Computes the global non-dominated Pareto front across all evaluated variants in memory."""
        all_recs = self.store.get_all_records()
        evaluated = [r for r in all_recs if r.objectives_vector is not None]
        if not evaluated:
            return []

        # Find non-dominated subset
        pareto_recs: List[MemoryRecord] = []
        for i, p in enumerate(evaluated):
            p_vec = np.asarray(p.objectives_vector, dtype=np.float64)
            is_dominated = False

            for j, q in enumerate(evaluated):
                if i == j:
                    continue
                q_vec = np.asarray(q.objectives_vector, dtype=np.float64)
                # Check if q dominates p
                if np.all(q_vec >= p_vec - 1e-9) and np.any(q_vec > p_vec + 1e-9):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_recs.append(p)

        return pareto_recs

    def close(self) -> None:
        """Closes memory store."""
        self.store.close()

    def __enter__(self) -> "EvolutionaryMemory":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
