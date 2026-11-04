"""Evolutionary Memory module for EVANTHEA.

Provides persistent caching of evaluated protein variants in SQLite,
dense vector similarity indexing for novelty/diversity analysis,
and memory-aware evolutionary search algorithms that eliminate redundant forward passes.
"""

from evanthea.memory.schema import (
    MemoryRecord,
    RedundancyStats,
    SimilaritySearchResult,
)
from evanthea.memory.vector_index import (
    ExactNumpyVectorIndex,
    FaissVectorIndex,
    VectorIndex,
)
from evanthea.memory.storage import PersistentMemoryStore
from evanthea.memory.manager import EvolutionaryMemory
from evanthea.memory.strategies import (
    ArchiveElitismInjector,
    EvaluationBypasser,
)
from evanthea.memory.algorithms import (
    MemoryAwareGA,
    MemoryAwareNSGA2,
)

__all__ = [
    "MemoryRecord",
    "RedundancyStats",
    "SimilaritySearchResult",
    "VectorIndex",
    "ExactNumpyVectorIndex",
    "FaissVectorIndex",
    "PersistentMemoryStore",
    "EvolutionaryMemory",
    "EvaluationBypasser",
    "ArchiveElitismInjector",
    "MemoryAwareNSGA2",
    "MemoryAwareGA",
]
