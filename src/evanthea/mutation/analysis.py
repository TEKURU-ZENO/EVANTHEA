"""Statistical benchmarking and proposal distribution analysis comparing mutation operators."""

import itertools
from typing import Any, Dict, List, Optional
import numpy as np

from evanthea.constants import STANDARD_AMINO_ACIDS
from evanthea.mutation.base import MutationOperator, ProposalRecord
from evanthea.mutation.constraints import ConstraintPipeline
from evanthea.mutation.point_mutations import BLOSUM62_MATRIX
from evanthea.protein.language_model import ProteinLanguageModel
from evanthea.protein.sequence import compute_hamming_distance, validate_amino_acid_string

SORTED_AMINO_ACIDS = sorted(STANDARD_AMINO_ACIDS)


def compute_proposal_distribution_metrics(
    proposals: List[ProposalRecord],
    parent_sequence: str,
    plm: Optional[ProteinLanguageModel] = None,
    max_pairwise_samples: int = 200,
) -> Dict[str, Any]:
    """Computes distribution statistics, biophysical alignment, and sequence diversity of proposals.

    Args:
        proposals: List of generated ProposalRecord instances.
        parent_sequence: Reference parent amino acid string.
        plm: Optional language model to compute plausibility if not already in ProposalRecord.
        max_pairwise_samples: Maximum variants to compute pairwise Hamming distances over.

    Returns:
        Dictionary of statistical summary metrics.
    """
    if not proposals:
        return {"n_proposals": 0}

    ref_seq = validate_amino_acid_string(parent_sequence)
    n = len(proposals)

    # 1. Acceptance rate
    accepted_count = sum(1 for p in proposals if p.is_accepted)
    acceptance_rate = float(accepted_count / n)

    # 2. Plausibility scores (delta log p)
    delta_log_ps: List[float] = []
    for p in proposals:
        if p.final_variant_score is not None:
            delta_log_ps.append(p.final_variant_score)
        elif p.step_proposal_scores:
            delta_log_ps.append(sum(p.step_proposal_scores))
        elif plm is not None and len(p.mutations) > 0:
            try:
                score_res = plm.score_mutations(ref_seq, p.mutations, method="masked_marginal")
                delta_log_ps.append(sum(s.log_odds_score for s in score_res))
            except Exception:
                pass

    delta_arr = np.array(delta_log_ps, dtype=np.float64) if delta_log_ps else np.array([], dtype=np.float64)
    plausibility_stats = {
        "mean_delta_log_p": float(np.mean(delta_arr)) if len(delta_arr) > 0 else None,
        "median_delta_log_p": float(np.median(delta_arr)) if len(delta_arr) > 0 else None,
        "std_delta_log_p": float(np.std(delta_arr)) if len(delta_arr) > 0 else None,
        "min_delta_log_p": float(np.min(delta_arr)) if len(delta_arr) > 0 else None,
        "max_delta_log_p": float(np.max(delta_arr)) if len(delta_arr) > 0 else None,
    }

    # 3. BLOSUM62 Alignment Scores
    blosum_scores: List[float] = []
    for p in proposals:
        for m in p.mutations:
            pair = (m.ref_residue, m.mut_residue)
            if pair in BLOSUM62_MATRIX:
                blosum_scores.append(BLOSUM62_MATRIX[pair])

    blosum_arr = np.array(blosum_scores, dtype=np.float64) if blosum_scores else np.array([], dtype=np.float64)
    blosum_stats = {
        "mean_blosum62_score": float(np.mean(blosum_arr)) if len(blosum_arr) > 0 else None,
        "median_blosum62_score": float(np.median(blosum_arr)) if len(blosum_arr) > 0 else None,
        "std_blosum62_score": float(np.std(blosum_arr)) if len(blosum_arr) > 0 else None,
    }

    # 4. Residue Shannon Entropy (diversity of proposed mutant residues)
    mutant_aa_counts: Dict[str, int] = {aa: 0 for aa in SORTED_AMINO_ACIDS}
    total_mut_residues = 0
    for p in proposals:
        for m in p.mutations:
            if m.mut_residue in mutant_aa_counts:
                mutant_aa_counts[m.mut_residue] += 1
                total_mut_residues += 1

    if total_mut_residues > 0:
        aa_probs = np.array([count / total_mut_residues for count in mutant_aa_counts.values()], dtype=np.float64)
        non_zero = aa_probs[aa_probs > 0]
        residue_shannon_entropy = -float(np.sum(non_zero * np.log2(non_zero)))
    else:
        residue_shannon_entropy = 0.0

    # 5. Pairwise Hamming Diversity
    unique_seqs = list({p.child_sequence for p in proposals})
    if len(unique_seqs) > max_pairwise_samples:
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(len(unique_seqs), size=max_pairwise_samples, replace=False)
        eval_seqs = [unique_seqs[i] for i in sample_indices]
    else:
        eval_seqs = unique_seqs

    if len(eval_seqs) > 1:
        pairwise_dists = [
            compute_hamming_distance(s1, s2)
            for s1, s2 in itertools.combinations(eval_seqs, 2)
        ]
        mean_pairwise_dist = float(np.mean(pairwise_dists))
        median_pairwise_dist = float(np.median(pairwise_dists))
    else:
        mean_pairwise_dist = 0.0
        median_pairwise_dist = 0.0

    # 6. Latency / Throughput
    latencies = [p.generation_time_ms for p in proposals if p.generation_time_ms is not None]
    mean_latency_ms = float(np.mean(latencies)) if latencies else None

    return {
        "n_proposals": n,
        "n_unique_variants": len(unique_seqs),
        "acceptance_rate": acceptance_rate,
        "plausibility": plausibility_stats,
        "blosum62": blosum_stats,
        "residue_shannon_entropy_bits": residue_shannon_entropy,
        "mean_pairwise_hamming": mean_pairwise_dist,
        "median_pairwise_hamming": median_pairwise_dist,
        "mean_generation_time_ms": mean_latency_ms,
    }


def compare_proposal_generators(
    generators: Dict[str, MutationOperator],
    parent_sequence: str,
    n_proposals: int = 500,
    constraint_pipeline: Optional[ConstraintPipeline] = None,
    plm: Optional[ProteinLanguageModel] = None,
) -> Dict[str, Any]:
    """Empirically evaluates and compares multiple proposal generators against a reference protein."""
    ref_seq = validate_amino_acid_string(parent_sequence)
    pipeline = constraint_pipeline or ConstraintPipeline()

    comparison_results: Dict[str, Any] = {
        "parent_length": len(ref_seq),
        "n_proposals_per_generator": n_proposals,
        "generators": {},
    }

    for gen_name, gen_op in generators.items():
        raw_props = gen_op.propose(ref_seq, n_proposals=n_proposals)
        # Apply constraint pipeline
        pipeline.filter_proposals(raw_props, parent_sequence=ref_seq)

        metrics = compute_proposal_distribution_metrics(
            proposals=raw_props,
            parent_sequence=ref_seq,
            plm=plm,
        )
        comparison_results["generators"][gen_name] = metrics

    return comparison_results
