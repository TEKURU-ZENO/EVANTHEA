"""Decoupled vector indexing abstractions for dense protein representation similarity search."""

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
import numpy as np

from evanthea.memory.schema import SimilaritySearchResult


class VectorIndex(ABC):
    """Abstract base class for protein sequence embedding index and similarity retrieval."""

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError(f"VectorIndex dimension must be > 0, got {dimension}")
        self.dimension = dimension

    @abstractmethod
    def add(
        self,
        hashes: Sequence[str],
        vectors: np.ndarray,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """Adds dense embedding vectors aligned with sequence hashes to the index.

        Args:
            hashes: List of unique sequence hashes of length N.
            vectors: Float array of shape (N, D).
            metadata: Optional list of N metadata dictionaries.
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        metric: Literal["cosine", "l2"] = "cosine",
    ) -> List[SimilaritySearchResult]:
        """Searches index for nearest neighbors to a query vector.

        Args:
            query_vector: Query embedding vector of shape (D,) or (1, D).
            top_k: Maximum number of nearest candidates to return.
            metric: Distance metric ('cosine' for normalized angular similarity, 'l2' for Euclidean).

        Returns:
            List of SimilaritySearchResult objects sorted by nearest proximity.
        """
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """Returns the total number of indexed vectors."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Enables len(index) to return total indexed vectors."""
        return self.count()

    @abstractmethod
    def has_hash(self, seq_hash: str) -> bool:
        """Checks if a sequence hash is present in the index."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path | str) -> None:
        """Persists index to disk."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path | str) -> None:
        """Loads index from disk."""
        raise NotImplementedError


class ExactNumpyVectorIndex(VectorIndex):
    """Pure NumPy vector index supporting exact Cosine and Euclidean (L2) similarity search.

    Features:
    - 100% exact recall (no quantization or approximation error)
    - Zero external dependencies (guaranteed to work across all platforms and architectures)
    - Matrix-vectorized dot product and broadcasting for high throughput
    """

    def __init__(self, dimension: int, default_metric: Literal["cosine", "l2"] = "cosine", **kwargs: Any) -> None:
        super().__init__(dimension=dimension)
        self.default_metric = default_metric
        self.hashes: List[str] = []
        self.hash_to_idx: Dict[str, int] = {}
        self.vectors: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        self.metadata_list: List[Dict[str, Any]] = []

    def add(
        self,
        hashes: Sequence[str],
        vectors: np.ndarray,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)

        n, d = vecs.shape
        if d != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {d}")
        if len(hashes) != n:
            raise ValueError(f"Length mismatch: got {len(hashes)} hashes for {n} vectors")

        meta = list(metadata) if metadata is not None else [{} for _ in range(n)]

        new_hashes: List[str] = []
        new_vecs: List[np.ndarray] = []
        new_meta: List[Dict[str, Any]] = []

        for h, v, m in zip(hashes, vecs, meta):
            if h in self.hash_to_idx:
                # Update existing vector and metadata
                idx = self.hash_to_idx[h]
                self.vectors[idx] = v
                self.metadata_list[idx] = m
            else:
                idx = len(self.hashes) + len(new_hashes)
                self.hash_to_idx[h] = idx
                new_hashes.append(h)
                new_vecs.append(v)
                new_meta.append(m)

        if new_vecs:
            stacked_new = np.stack(new_vecs, axis=0)
            if self.vectors.shape[0] == 0:
                self.vectors = stacked_new
            else:
                self.vectors = np.concatenate([self.vectors, stacked_new], axis=0)
            self.hashes.extend(new_hashes)
            self.metadata_list.extend(new_meta)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        metric: Literal["cosine", "l2"] = "cosine",
    ) -> List[SimilaritySearchResult]:
        q = np.asarray(query_vector, dtype=np.float32).flatten()
        if len(q) != self.dimension:
            raise ValueError(f"Query vector dimension mismatch: expected {self.dimension}, got {len(q)}")

        if len(self.hashes) == 0:
            return []

        k = min(top_k, len(self.hashes))
        if k <= 0:
            return []

        if metric == "cosine":
            # Normalized cosine similarity: q · v / (||q|| * ||v||)
            q_norm = np.linalg.norm(q)
            if q_norm < 1e-12:
                q_normalized = q
            else:
                q_normalized = q / q_norm

            v_norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            v_norms = np.where(v_norms < 1e-12, 1.0, v_norms)
            v_normalized = self.vectors / v_norms

            # Cosine similarities in [-1, 1]
            sims = np.dot(v_normalized, q_normalized)
            # Distance in [0, 2]
            dists = 1.0 - sims

            # Sort descending by similarity
            top_indices = np.argsort(sims)[::-1][:k]

        elif metric == "l2":
            # Euclidean distance: ||q - v||_2
            diffs = self.vectors - q
            dists = np.linalg.norm(diffs, axis=1)
            # Convert distance to bounded similarity: 1 / (1 + dist)
            sims = 1.0 / (1.0 + dists)

            # Sort ascending by distance
            top_indices = np.argsort(dists)[:k]

        else:
            raise ValueError(f"Unsupported metric '{metric}'. Must be 'cosine' or 'l2'.")

        results: List[SimilaritySearchResult] = []
        for idx in top_indices:
            h = self.hashes[idx]
            m = self.metadata_list[idx]
            results.append(
                SimilaritySearchResult(
                    sequence_hash=h,
                    variant_id=m.get("variant_id", h[:16]),
                    similarity_score=float(sims[idx]),
                    distance=float(dists[idx]),
                    fitness_values=m.get("fitness_values", {}),
                    mutations=m.get("mutations", []),
                    metadata=m,
                )
            )

        return results

    def count(self) -> int:
        return len(self.hashes)

    def has_hash(self, seq_hash: str) -> bool:
        return seq_hash in self.hash_to_idx

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        meta_json = [json.dumps(m) for m in self.metadata_list]
        np.savez_compressed(
            p,
            vectors=self.vectors,
            hashes=np.array(self.hashes, dtype=object),
            metadata=np.array(meta_json, dtype=object),
            dimension=np.array([self.dimension]),
        )

    def load(self, path: Path | str) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Vector index file not found: {p}")

        data = np.load(p, allow_pickle=True)
        self.dimension = int(data["dimension"][0])
        self.vectors = np.asarray(data["vectors"], dtype=np.float32)
        self.hashes = [str(h) for h in data["hashes"]]
        self.hash_to_idx = {h: i for i, h in enumerate(self.hashes)}

        meta_raw = data["metadata"]
        self.metadata_list = [json.loads(str(m)) for m in meta_raw]


class FaissVectorIndex(VectorIndex):
    """FAISS-powered vector index with automated fallback to ExactNumpyVectorIndex.

    If faiss is installed, utilizes native C++ vector acceleration.
    If faiss is absent, transparently wraps ExactNumpyVectorIndex without API change.
    """

    def __init__(self, dimension: int, default_metric: Literal["cosine", "l2"] = "cosine", **kwargs: Any) -> None:
        super().__init__(dimension=dimension)
        self.default_metric = default_metric
        self._fallback_index = ExactNumpyVectorIndex(dimension=dimension, default_metric=default_metric)
        self._faiss_available = False

        try:
            import faiss
            self._faiss = faiss
            self._faiss_available = True
            # Inner-product index for normalized cosine similarity
            self._index_ip = faiss.IndexFlatIP(dimension)
            # L2 index for Euclidean distance
            self._index_l2 = faiss.IndexFlatL2(dimension)
            self._hashes: List[str] = []
            self._hash_to_idx: Dict[str, int] = {}
            self._metadata: List[Dict[str, Any]] = []
        except ImportError:
            self._faiss = None
            self._faiss_available = False

    @property
    def is_faiss_native(self) -> bool:
        """Returns True if native FAISS C++ library is active."""
        return self._faiss_available

    def add(
        self,
        hashes: Sequence[str],
        vectors: np.ndarray,
        metadata: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        # Always maintain the fallback index for consistency and simple serialization
        self._fallback_index.add(hashes, vectors, metadata)

        if not self._faiss_available:
            return

        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)

        meta = list(metadata) if metadata is not None else [{} for _ in range(len(hashes))]

        # Re-build faiss indices from scratch when updating
        self._index_ip.reset()
        self._index_l2.reset()
        all_vecs = self._fallback_index.vectors

        if len(all_vecs) > 0:
            # L2 normalized for inner product (cosine)
            norms = np.linalg.norm(all_vecs, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            normed = all_vecs / norms
            self._index_ip.add(normed.astype(np.float32))
            self._index_l2.add(all_vecs.astype(np.float32))

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        metric: Literal["cosine", "l2"] = "cosine",
    ) -> List[SimilaritySearchResult]:
        if not self._faiss_available:
            return self._fallback_index.search(query_vector, top_k=top_k, metric=metric)

        n_total = self._fallback_index.count()
        if n_total == 0:
            return []

        k = min(top_k, n_total)
        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)

        if metric == "cosine":
            q_norm = np.linalg.norm(q)
            q_normed = q if q_norm < 1e-12 else q / q_norm
            scores, indices = self._index_ip.search(q_normed.astype(np.float32), k)

            results: List[SimilaritySearchResult] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                h = self._fallback_index.hashes[idx]
                m = self._fallback_index.metadata_list[idx]
                sim = float(score)
                dist = float(1.0 - sim)
                results.append(
                    SimilaritySearchResult(
                        sequence_hash=h,
                        variant_id=m.get("variant_id", h[:16]),
                        similarity_score=sim,
                        distance=dist,
                        fitness_values=m.get("fitness_values", {}),
                        mutations=m.get("mutations", []),
                        metadata=m,
                    )
                )
            return results

        elif metric == "l2":
            sq_dists, indices = self._index_l2.search(q.astype(np.float32), k)

            results = []
            for sq_d, idx in zip(sq_dists[0], indices[0]):
                if idx < 0:
                    continue
                h = self._fallback_index.hashes[idx]
                m = self._fallback_index.metadata_list[idx]
                dist = float(np.sqrt(max(0.0, sq_d)))
                sim = float(1.0 / (1.0 + dist))
                results.append(
                    SimilaritySearchResult(
                        sequence_hash=h,
                        variant_id=m.get("variant_id", h[:16]),
                        similarity_score=sim,
                        distance=dist,
                        fitness_values=m.get("fitness_values", {}),
                        mutations=m.get("mutations", []),
                        metadata=m,
                    )
                )
            return results

        raise ValueError(f"Unsupported metric: {metric}")

    def count(self) -> int:
        return self._fallback_index.count()

    def has_hash(self, seq_hash: str) -> bool:
        return self._fallback_index.has_hash(seq_hash)

    def save(self, path: Path | str) -> None:
        self._fallback_index.save(path)

    def load(self, path: Path | str) -> None:
        self._fallback_index.load(path)
        if self._faiss_available and self._fallback_index.count() > 0:
            self._index_ip.reset()
            self._index_l2.reset()
            all_vecs = self._fallback_index.vectors
            norms = np.linalg.norm(all_vecs, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            normed = all_vecs / norms
            self._index_ip.add(normed.astype(np.float32))
            self._index_l2.add(all_vecs.astype(np.float32))
