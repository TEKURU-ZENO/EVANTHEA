"""Two-tier persistent caching system: in-memory LRU + SQLite metadata + HDF5 dense embeddings."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from evanthea.exceptions import CacheError


class EmbeddingCache:
    """Manages high-throughput storage and retrieval of sequence embeddings and PLM scores."""

    def __init__(self, cache_dir: Union[Path, str]):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.cache_dir / "cache_metadata.sqlite"
        self.h5_path = self.cache_dir / "embeddings.h5"

        self._init_sqlite()

    def _init_sqlite(self) -> None:
        """Initializes SQLite tables and activates WAL mode for concurrent read/write stability."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA journal_mode = WAL;")
                conn.execute("PRAGMA synchronous = NORMAL;")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sequence_cache (
                        seq_hash TEXT PRIMARY KEY,
                        length INTEGER NOT NULL,
                        masked_marginal_score REAL,
                        pseudo_perplexity REAL,
                        model_name TEXT,
                        created_at TEXT NOT NULL,
                        metadata_json TEXT
                    );
                    """
                )
                conn.commit()
        except sqlite3.Error as e:
            raise CacheError(f"Failed to initialize SQLite cache at {self.db_path}: {e}") from e

    def has_sequence(self, seq_hash: str) -> bool:
        """Checks if a sequence hash exists in the cache."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM sequence_cache WHERE seq_hash = ? LIMIT 1;", (seq_hash,))
            return cursor.fetchone() is not None

    def get_metadata(self, seq_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieves cached metadata for a sequence hash."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT seq_hash, length, masked_marginal_score, pseudo_perplexity, model_name, created_at, metadata_json
                FROM sequence_cache WHERE seq_hash = ?;
                """,
                (seq_hash,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            return {
                "seq_hash": row[0],
                "length": row[1],
                "masked_marginal_score": row[2],
                "pseudo_perplexity": row[3],
                "model_name": row[4],
                "created_at": row[5],
                "metadata": json.loads(row[6]) if row[6] else {},
            }

    def put_embedding(
        self,
        seq_hash: str,
        embedding: np.ndarray,
        length: int,
        model_name: str,
        masked_marginal_score: Optional[float] = None,
        pseudo_perplexity: Optional[float] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Stores dense embedding in HDF5 and associated scores in SQLite."""
        import h5py

        # 1. Store dense array in HDF5
        try:
            with h5py.File(self.h5_path, "a") as h5f:
                if seq_hash in h5f:
                    del h5f[seq_hash]
                h5f.create_dataset(
                    seq_hash,
                    data=embedding.astype(np.float16),
                    compression="gzip",
                    compression_opts=4,
                )
        except Exception as e:
            raise CacheError(f"Failed to write embedding to HDF5 store: {e}") from e

        # 2. Store metadata in SQLite
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO sequence_cache
                    (seq_hash, length, masked_marginal_score, pseudo_perplexity, model_name, created_at, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        seq_hash,
                        length,
                        masked_marginal_score,
                        pseudo_perplexity,
                        model_name,
                        datetime.now(timezone.utc).isoformat(),
                        json.dumps(extra_metadata or {}),
                    ),
                )
                conn.commit()
        except sqlite3.Error as e:
            raise CacheError(f"Failed to write metadata to SQLite cache: {e}") from e

    def get_embedding(self, seq_hash: str) -> Optional[np.ndarray]:
        """Retrieves dense embedding array from HDF5 store."""
        import h5py

        if not self.h5_path.is_file():
            return None

        try:
            with h5py.File(self.h5_path, "r") as h5f:
                if seq_hash not in h5f:
                    return None
                return np.array(h5f[seq_hash], dtype=np.float32)
        except Exception as e:
            raise CacheError(f"Failed to read embedding from HDF5 store: {e}") from e
