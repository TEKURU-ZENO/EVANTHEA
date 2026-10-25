"""Feature construction, biophysical descriptors, and embedding transformations for fitness prediction."""

from enum import Enum
from typing import Any, Dict, Final, List, Mapping, Optional, Tuple, Union
import numpy as np

from evanthea.data.cache import EmbeddingCache
from evanthea.data.schemas import Mutation, ProteinRecord, VariantRecord
from evanthea.protein.sequence import compute_sequence_hash

# ---------------------------------------------------------------------------
# Standard Biophysical Property Scales
# ---------------------------------------------------------------------------

# Kyte-Doolittle Hydropathy Index (Kyte & Doolittle, J. Mol. Biol. 1982)
KYTE_DOOLITTLE_HYDROPATHY: Final[Mapping[str, float]] = {
    "A": 1.8,  "C": 2.5,  "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5,  "K": -3.9, "L": 3.8,
    "M": 1.9,  "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2,  "W": -0.9, "Y": -1.3,
}

# Net formal residue charge at physiological pH (approx. 7.4)
PHYSIOLOGICAL_CHARGE: Final[Mapping[str, float]] = {
    "R": 1.0,  "K": 1.0,  "H": 0.1,  # basic residues (His partially protonated)
    "D": -1.0, "E": -1.0,             # acidic residues
    "A": 0.0,  "C": 0.0,  "F": 0.0,  "G": 0.0,  "I": 0.0,
    "L": 0.0,  "M": 0.0,  "N": 0.0,  "P": 0.0,  "Q": 0.0,
    "S": 0.0,  "T": 0.0,  "V": 0.0,  "W": 0.0,  "Y": 0.0,
}

# Van der Waals residue volume in Angstroms^3 (Zamyatnin, Prog. Biophys. Mol. Biol. 1972)
RESIDUE_VDW_VOLUME: Final[Mapping[str, float]] = {
    "G": 60.1,  "A": 88.6,  "S": 89.0,  "C": 108.5, "D": 111.1,
    "P": 112.7, "N": 114.1, "T": 116.1, "E": 138.4, "V": 140.0,
    "Q": 143.8, "H": 153.2, "M": 162.9, "I": 166.7, "L": 166.7,
    "K": 168.6, "R": 173.4, "F": 189.9, "Y": 193.6, "W": 227.8,
}


class FeatureConfig(str, Enum):
    """The five standardized feature configurations evaluated in EVANTHEA Phase 2."""
    SEQ_EMB = "seq_emb"                     # Config A: Mutant embedding h(x_v)
    EMB_DIFF = "emb_diff"                   # Config B: Difference vector Δh = h(x_v) - h(x_WT)
    EMB_CONCAT = "emb_concat"               # Config C: [h(x_v); Δh]
    MUTATION_FEATURES = "mutation_features" # Config D: Interpretable biophysical + PLM descriptors
    HYBRID = "hybrid"                       # Config E: [Δh; mutation_features]


def compute_biophysical_deltas(mutations: List[Mutation]) -> Tuple[float, float, float]:
    """Calculates additive changes in charge, hydropathy, and volume for a variant's mutations.

    Aggregation rule:
        ΔProperty = Sum_{m in mutations} (Property(a_mut) - Property(a_ref))
        For wild-type (empty mutations list), returns (0.0, 0.0, 0.0).

    Returns:
        Tuple of (delta_charge, delta_hydropathy, delta_volume)
    """
    if not mutations:
        return 0.0, 0.0, 0.0

    delta_charge = 0.0
    delta_hydropathy = 0.0
    delta_volume = 0.0

    for m in mutations:
        delta_charge += PHYSIOLOGICAL_CHARGE.get(m.mut_residue, 0.0) - PHYSIOLOGICAL_CHARGE.get(m.ref_residue, 0.0)
        delta_hydropathy += KYTE_DOOLITTLE_HYDROPATHY.get(m.mut_residue, 0.0) - KYTE_DOOLITTLE_HYDROPATHY.get(m.ref_residue, 0.0)
        delta_volume += RESIDUE_VDW_VOLUME.get(m.mut_residue, 0.0) - RESIDUE_VDW_VOLUME.get(m.ref_residue, 0.0)

    return float(delta_charge), float(delta_hydropathy), float(delta_volume)


class FeatureMatrixBuilder:
    """Constructs aligned feature matrices from VariantRecords and persistent EmbeddingCache."""

    def __init__(
        self,
        reference_protein: ProteinRecord,
        cache: Optional[EmbeddingCache] = None,
        wt_embedding: Optional[np.ndarray] = None,
    ):
        self.reference_protein = reference_protein
        self.cache = cache
        self.wt_hash = compute_sequence_hash(reference_protein.wildtype_sequence)

        # Retrieve or assign WT embedding
        if wt_embedding is not None:
            self.wt_embedding = np.asarray(wt_embedding, dtype=np.float32)
        elif self.cache is not None and self.cache.has_sequence(self.wt_hash):
            cached_wt = self.cache.get_embedding(self.wt_hash)
            if cached_wt is None:
                raise ValueError(f"Wild-type embedding exists in cache index but returned None for hash {self.wt_hash}.")
            self.wt_embedding = cached_wt
        else:
            self.wt_embedding = None

    def get_variant_embedding(
        self,
        variant: VariantRecord,
        provided_embeddings: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """Retrieves embedding array for a variant from in-memory dict, cache, or raises error."""
        seq_hash = compute_sequence_hash(variant.mutated_sequence)

        if provided_embeddings and seq_hash in provided_embeddings:
            return np.asarray(provided_embeddings[seq_hash], dtype=np.float32)

        if self.cache and self.cache.has_sequence(seq_hash):
            emb = self.cache.get_embedding(seq_hash)
            if emb is not None:
                return emb

        raise KeyError(
            f"No embedding found for variant '{variant.variant_id}' (sequence hash: {seq_hash[:12]}...). "
            f"Ensure embeddings are extracted and cached prior to building feature matrices."
        )

    def extract_mutation_descriptor_vector(
        self,
        variant: VariantRecord,
    ) -> np.ndarray:
        """Constructs 6-dimensional interpretable descriptor vector:

        [num_mutations, masked_marginal_score, pseudo_perplexity, delta_charge, delta_hydropathy, delta_volume]
        """
        seq_hash = compute_sequence_hash(variant.mutated_sequence)
        meta = self.cache.get_metadata(seq_hash) if self.cache else None

        # Masked marginal zero-shot score
        mm_score = 0.0
        pll = 0.0
        if meta:
            if meta.get("masked_marginal_score") is not None:
                mm_score = float(meta["masked_marginal_score"])
            if meta.get("pseudo_perplexity") is not None:
                pll = float(meta["pseudo_perplexity"])

        d_charge, d_hydro, d_vol = compute_biophysical_deltas(variant.mutations)

        return np.array(
            [
                float(variant.num_mutations),
                float(mm_score),
                float(pll),
                d_charge,
                d_hydro,
                d_vol,
            ],
            dtype=np.float32,
        )

    def build_features_for_variant(
        self,
        variant: VariantRecord,
        config: Union[FeatureConfig, str],
        provided_embeddings: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """Builds a single feature vector for a variant under the requested configuration."""
        cfg = FeatureConfig(config)
        requires_emb = cfg in {
            FeatureConfig.SEQ_EMB,
            FeatureConfig.EMB_DIFF,
            FeatureConfig.EMB_CONCAT,
            FeatureConfig.HYBRID,
        }

        h_v = self.get_variant_embedding(variant, provided_embeddings) if requires_emb else None

        if cfg == FeatureConfig.SEQ_EMB:
            assert h_v is not None
            return h_v

        if cfg == FeatureConfig.EMB_DIFF:
            assert h_v is not None
            if self.wt_embedding is None:
                raise ValueError("Wild-type embedding is required to compute embedding differences (Config B).")
            return h_v - self.wt_embedding

        if cfg == FeatureConfig.EMB_CONCAT:
            assert h_v is not None
            if self.wt_embedding is None:
                raise ValueError("Wild-type embedding is required to compute concatenated embeddings (Config C).")
            delta_h = h_v - self.wt_embedding
            return np.concatenate([h_v, delta_h], axis=0)

        if cfg == FeatureConfig.MUTATION_FEATURES:
            return self.extract_mutation_descriptor_vector(variant)

        if cfg == FeatureConfig.HYBRID:
            assert h_v is not None
            if self.wt_embedding is None:
                raise ValueError("Wild-type embedding is required to compute hybrid representations (Config E).")
            delta_h = h_v - self.wt_embedding
            desc = self.extract_mutation_descriptor_vector(variant)
            return np.concatenate([delta_h, desc], axis=0)

        raise ValueError(f"Unhandled FeatureConfig: {cfg}")

    def build_matrix(
        self,
        variants: List[VariantRecord],
        config: Union[FeatureConfig, str],
        provided_embeddings: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Constructs the full feature matrix X (N x D) aligned with variant identifiers.

        Args:
            variants: Ordered list of VariantRecords.
            config: Feature configuration to build.
            provided_embeddings: Optional in-memory mapping from sequence hash to embedding.

        Returns:
            Tuple of (X feature matrix as float32 np.ndarray, list of variant_ids).
        """
        if not variants:
            raise ValueError("Cannot build feature matrix from empty variants list.")

        feature_vectors: List[np.ndarray] = []
        variant_ids: List[str] = []

        for v in variants:
            vec = self.build_features_for_variant(v, config, provided_embeddings)
            feature_vectors.append(vec)
            variant_ids.append(v.variant_id)

        X = np.stack(feature_vectors, axis=0).astype(np.float32)

        # Strict validation
        if not np.all(np.isfinite(X)):
            raise ValueError(f"Constructed feature matrix contains non-finite values (NaN/Inf) under config '{config}'.")

        return X, variant_ids


class FeatureScaler:
    """Standardizes input feature matrix X strictly using training statistics (zero test leakage)."""

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.is_fitted = False
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X_train: np.ndarray) -> "FeatureScaler":
        arr = np.asarray(X_train, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape {arr.shape}.")

        if self.with_mean:
            self.mean_ = np.mean(arr, axis=0)
        else:
            self.mean_ = np.zeros(arr.shape[1], dtype=np.float32)

        if self.with_std:
            std = np.std(arr, axis=0, ddof=0)
            # Guard against zero variance columns
            std[std < 1e-12] = 1.0
            self.scale_ = std
        else:
            self.scale_ = np.ones(arr.shape[1], dtype=np.float32)

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("FeatureScaler must be fitted on training data before calling transform().")
        arr = np.asarray(X, dtype=np.float32)
        assert self.mean_ is not None and self.scale_ is not None
        return (arr - self.mean_) / self.scale_

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)
