"""Single-point mutation operators: Uniform Random, BLOSUM62-informed, and ESM-2-guided."""

import time
from typing import Any, Dict, List, Literal, Optional, Set, Tuple
import numpy as np

from evanthea.constants import STANDARD_AMINO_ACIDS
from evanthea.data.schemas import Mutation
from evanthea.exceptions import CoordinateOutOfBoundsError
from evanthea.mutation.base import MutationOperator, ProposalRecord
from evanthea.protein.language_model import ProteinLanguageModel
from evanthea.protein.sequence import (
    apply_mutations,
    compute_sequence_hash,
    validate_amino_acid_string,
)

# Canonical 20 amino acids sorted alphabetically
SORTED_AMINO_ACIDS = sorted(STANDARD_AMINO_ACIDS)


def _load_blosum62_matrix() -> Dict[Tuple[str, str], float]:
    """Loads and returns the standard BLOSUM62 substitution matrix as a dictionary."""
    from Bio.Align.substitution_matrices import load
    matrix = load("BLOSUM62")
    blosum_dict: Dict[Tuple[str, str], float] = {}
    for aa1 in SORTED_AMINO_ACIDS:
        for aa2 in SORTED_AMINO_ACIDS:
            blosum_dict[(aa1, aa2)] = float(matrix[aa1, aa2])
    return blosum_dict


# Cached in module
BLOSUM62_MATRIX = _load_blosum62_matrix()


def _validate_allowed_positions(
    seq_len: int,
    allowed_positions: Optional[Set[int]] = None,
) -> List[int]:
    """Validates and returns sorted list of 1-based allowed positions."""
    if allowed_positions is None:
        return list(range(1, seq_len + 1))

    for p in allowed_positions:
        if p < 1 or p > seq_len:
            raise CoordinateOutOfBoundsError(p, seq_len, f"allowed_position={p}")

    sorted_pos = sorted(allowed_positions)
    if not sorted_pos:
        raise ValueError("Allowed positions set cannot be empty.")
    return sorted_pos


class UniformRandomMutationGenerator(MutationOperator):
    """Proposes single-point mutations via uniform sampling over positions and non-wildtype residues."""

    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__(name="uniform_random", seed=seed)

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

        records: List[ProposalRecord] = []
        for _ in range(n_proposals):
            start_t = time.perf_counter()

            # 1. Select position uniformly
            pos_1b = int(self.rng.choice(pos_list))
            wt_aa = ref_seq[pos_1b - 1]

            # 2. Select replacement uniformly from remaining 19 canonical amino acids
            candidate_aas = [aa for aa in SORTED_AMINO_ACIDS if aa != wt_aa]
            mut_aa = str(self.rng.choice(candidate_aas))

            # 3. Create mutation and child sequence
            mut = Mutation(ref_residue=wt_aa, position_1based=pos_1b, mut_residue=mut_aa)
            child_seq = apply_mutations(ref_seq, [mut])
            child_hash = compute_sequence_hash(child_seq)

            # Log-odds score for uniform selection is log(1/19)
            uniform_log_prob = float(np.log(1.0 / 19.0))
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            records.append(
                ProposalRecord(
                    parent_hash=parent_hash,
                    child_sequence=child_seq,
                    child_hash=child_hash,
                    mutations=[mut],
                    generator_name=self.name,
                    position_strategy="uniform",
                    seed=self.seed,
                    step_proposal_scores=[uniform_log_prob],
                    final_variant_score=uniform_log_prob,
                    is_accepted=True,
                    rejection_reasons=[],
                    generation_time_ms=elapsed_ms,
                )
            )

        return records


class BLOSUM62MutationGenerator(MutationOperator):
    r"""Proposes single-point mutations sampled from temperature-scaled BLOSUM62 substitution probabilities.

    Sampling formula:
        P(a | x_i) ∝ exp( B(x_i, a) / T_B )   for a in A \ {x_i}
        P(x_i | x_i) = 0 (wild-type excluded)
    """

    def __init__(self, temperature: float = 1.0, seed: Optional[int] = None) -> None:
        super().__init__(name="blosum62", seed=seed)
        if temperature <= 0:
            raise ValueError(f"Temperature must be > 0, got {temperature}")
        self.temperature = temperature

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

        records: List[ProposalRecord] = []
        for _ in range(n_proposals):
            start_t = time.perf_counter()

            # 1. Select position uniformly from allowed positions
            pos_1b = int(self.rng.choice(pos_list))
            wt_aa = ref_seq[pos_1b - 1]

            # 2. Extract BLOSUM62 scores for all 19 non-wildtype residues
            candidate_aas = [aa for aa in SORTED_AMINO_ACIDS if aa != wt_aa]
            scores = np.array([BLOSUM62_MATRIX[(wt_aa, aa)] for aa in candidate_aas], dtype=np.float64)

            # 3. Apply temperature scaling and compute softmax probabilities
            scaled = scores / self.temperature
            max_s = np.max(scaled)
            exp_s = np.exp(scaled - max_s)
            probs = exp_s / np.sum(exp_s)

            # 4. Sample mutant residue
            chosen_idx = int(self.rng.choice(len(candidate_aas), p=probs))
            mut_aa = candidate_aas[chosen_idx]
            chosen_prob = float(probs[chosen_idx])
            blosum_raw_score = float(scores[chosen_idx])

            mut = Mutation(ref_residue=wt_aa, position_1based=pos_1b, mut_residue=mut_aa)
            child_seq = apply_mutations(ref_seq, [mut])
            child_hash = compute_sequence_hash(child_seq)
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            records.append(
                ProposalRecord(
                    parent_hash=parent_hash,
                    child_sequence=child_seq,
                    child_hash=child_hash,
                    mutations=[mut],
                    generator_name=self.name,
                    temperature=self.temperature,
                    position_strategy="uniform",
                    seed=self.seed,
                    step_proposal_scores=[float(np.log(chosen_prob))],
                    final_variant_score=blosum_raw_score,
                    is_accepted=True,
                    rejection_reasons=[],
                    generation_time_ms=elapsed_ms,
                    metadata={"blosum62_score": blosum_raw_score, "selection_prob": chosen_prob},
                )
            )

        return records


class ESM2PointMutationGenerator(MutationOperator):
    r"""Proposes single-point mutations using masked marginal logit distributions from a Protein Language Model.

    Features:
    - Temperature scaling (T)
    - Top-k filtering (retaining top k candidates)
    - Nucleus top-p truncation (cumulative probability mass threshold)
    - Flexible position selection strategies: 'uniform', 'entropy_weighted', 'conservation_weighted'
    - Mathematically exact plausibility score:
        Δ log p_i(a) = log p(a | x_\i) - log p(x_i | x_\i)
    """

    def __init__(
        self,
        language_model: ProteinLanguageModel,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        position_strategy: Literal["uniform", "entropy_weighted", "conservation_weighted"] = "uniform",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(name="esm2_point", seed=seed)
        if temperature <= 0:
            raise ValueError(f"Temperature must be > 0, got {temperature}")
        if top_k is not None and (top_k < 1 or top_k > 19):
            raise ValueError(f"Top-k must be in range [1, 19], got {top_k}")
        if top_p is not None and not (0.0 < top_p <= 1.0):
            raise ValueError(f"Top-p must be in range (0.0, 1.0], got {top_p}")

        self.language_model = language_model
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.position_strategy = position_strategy

    def _compute_position_probabilities(
        self,
        parent_sequence: str,
        pos_list: List[int],
    ) -> np.ndarray:
        """Computes sampling probabilities across allowed positions based on strategy."""
        n_pos = len(pos_list)
        if self.position_strategy == "uniform" or n_pos == 1:
            return np.ones(n_pos, dtype=np.float64) / n_pos

        # Single-pass wild-type marginal logits across sequence
        all_logits = self.language_model.get_wildtype_marginal_logits(parent_sequence)

        # Compute Shannon entropy H(i) for each allowed position
        entropies = []
        for pos_1b in pos_list:
            logits_dict = all_logits[pos_1b]
            logits_arr = np.array([logits_dict[aa] for aa in SORTED_AMINO_ACIDS], dtype=np.float64)
            # Softmax
            logits_arr -= np.max(logits_arr)
            probs = np.exp(logits_arr) / np.sum(np.exp(logits_arr))
            # H = - sum(p * log p)
            entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
            entropies.append(entropy)

        entropies_arr = np.array(entropies, dtype=np.float64)

        if self.position_strategy == "entropy_weighted":
            # Higher entropy -> more variable -> higher sampling probability
            weights = np.maximum(entropies_arr, 1e-6)
            total = np.sum(weights)
            return weights / total if total > 0 else np.ones(n_pos) / n_pos

        elif self.position_strategy == "conservation_weighted":
            # Lower entropy -> more conserved -> higher sampling probability
            # Use negative exponential of entropy
            weights = np.exp(-entropies_arr)
            total = np.sum(weights)
            return weights / total if total > 0 else np.ones(n_pos) / n_pos

        return np.ones(n_pos, dtype=np.float64) / n_pos

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

        # Position sampling distribution
        pos_probs = self._compute_position_probabilities(ref_seq, pos_list)

        records: List[ProposalRecord] = []
        for _ in range(n_proposals):
            start_t = time.perf_counter()

            # 1. Select position according to strategy
            pos_idx = int(self.rng.choice(len(pos_list), p=pos_probs))
            pos_1b = pos_list[pos_idx]
            wt_aa = ref_seq[pos_1b - 1]

            # 2. Get masked marginal logits at selected position
            raw_logits_dict = self.language_model.get_masked_marginal_logits(ref_seq, pos_1b)
            full_logits = np.array([raw_logits_dict[aa] for aa in SORTED_AMINO_ACIDS], dtype=np.float64)

            # Full unmasked probabilities for scientific delta-log-p calculation
            full_shifted = full_logits - np.max(full_logits)
            full_probs = np.exp(full_shifted) / np.sum(np.exp(full_shifted))
            wt_index = SORTED_AMINO_ACIDS.index(wt_aa)
            wt_log_prob = float(np.log(full_probs[wt_index] + 1e-12))

            # 3. For candidate sampling, exclude wild-type residue
            candidate_indices = [idx for idx, aa in enumerate(SORTED_AMINO_ACIDS) if aa != wt_aa]
            candidate_aas = [SORTED_AMINO_ACIDS[idx] for idx in candidate_indices]
            cand_logits = full_logits[candidate_indices].copy()

            # Apply temperature scaling
            cand_logits = cand_logits / self.temperature

            # 4. Apply Top-k truncation if configured
            if self.top_k is not None and self.top_k < len(cand_logits):
                sorted_cand_indices = np.argsort(cand_logits)[::-1]
                pruned_indices = sorted_cand_indices[self.top_k :]
                cand_logits[pruned_indices] = -np.inf

            # 5. Apply Nucleus (Top-p) truncation if configured
            if self.top_p is not None and self.top_p < 1.0:
                shifted = cand_logits - np.max(cand_logits)
                p_unnorm = np.exp(shifted)
                p_norm = p_unnorm / np.sum(p_unnorm)

                sorted_order = np.argsort(p_norm)[::-1]
                cum_mass = np.cumsum(p_norm[sorted_order])

                # Retain items up to cumulative mass >= top_p
                cutoff_idx = int(np.searchsorted(cum_mass, self.top_p))
                keep_indices = set(sorted_order[: cutoff_idx + 1])

                for idx in range(len(cand_logits)):
                    if idx not in keep_indices:
                        cand_logits[idx] = -np.inf

            # 6. Normalize over remaining candidates
            cand_shifted = cand_logits - np.max(cand_logits)
            cand_exp = np.exp(cand_shifted)
            cand_probs = cand_exp / np.sum(cand_exp)

            # 7. Sample candidate residue
            sampled_idx = int(self.rng.choice(len(candidate_aas), p=cand_probs))
            mut_aa = candidate_aas[sampled_idx]

            # 8. Compute exact Δ log p_i(a) = log p(a | x_\i) - log p(x_i | x_\i)
            mut_global_idx = SORTED_AMINO_ACIDS.index(mut_aa)
            mut_log_prob = float(np.log(full_probs[mut_global_idx] + 1e-12))
            delta_log_p = float(mut_log_prob - wt_log_prob)

            mut = Mutation(ref_residue=wt_aa, position_1based=pos_1b, mut_residue=mut_aa)
            child_seq = apply_mutations(ref_seq, [mut])
            child_hash = compute_sequence_hash(child_seq)
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            records.append(
                ProposalRecord(
                    parent_hash=parent_hash,
                    child_sequence=child_seq,
                    child_hash=child_hash,
                    mutations=[mut],
                    generator_name=self.name,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    position_strategy=self.position_strategy,
                    seed=self.seed,
                    step_proposal_scores=[delta_log_p],
                    final_variant_score=delta_log_p,
                    is_accepted=True,
                    rejection_reasons=[],
                    generation_time_ms=elapsed_ms,
                    metadata={
                        "wt_log_prob": wt_log_prob,
                        "mut_log_prob": mut_log_prob,
                        "delta_log_p": delta_log_p,
                        "sampling_prob": float(cand_probs[sampled_idx]),
                    },
                )
            )

        return records
