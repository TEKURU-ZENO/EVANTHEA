"""Batch embedding and masked-marginal score extraction pipeline with two-tier caching."""

import argparse
import csv
import logging
import time
from pathlib import Path

from evanthea.constants import DEFAULT_ESM2_150M_CHECKPOINT
from evanthea.data.cache import EmbeddingCache
from evanthea.data.schemas import Mutation, ProteinRecord, VariantRecord
from evanthea.protein.language_model import HuggingFaceESM2Model
from evanthea.protein.sequence import compute_sequence_hash, parse_mutation_string, validate_amino_acid_string

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evanthea.extract_embeddings")


def extract_and_cache(
    reference_sequence: str,
    variants: list[VariantRecord],
    cache_dir: Path,
    checkpoint: str = DEFAULT_ESM2_150M_CHECKPOINT,
    batch_size: int = 16,
):
    logger.info(f"Initializing EmbeddingCache at {cache_dir}...")
    cache = EmbeddingCache(cache_dir)

    logger.info(f"Loading ESM-2 model checkpoint: {checkpoint}...")
    plm = HuggingFaceESM2Model(checkpoint=checkpoint)

    logger.info(f"Beginning embedding and score extraction for {len(variants)} variants...")
    start_time = time.time()
    cache_hits = 0
    newly_computed = 0

    to_compute_variants: list[VariantRecord] = []
    to_compute_hashes: list[str] = []

    for v in variants:
        seq_hash = compute_sequence_hash(v.mutated_sequence)
        if cache.has_sequence(seq_hash):
            cache_hits += 1
        else:
            to_compute_variants.append(v)
            to_compute_hashes.append(seq_hash)

    logger.info(f"Cache check complete: {cache_hits} cached hits, {len(to_compute_variants)} pending computation.")

    # Process pending in batches
    for i in range(0, len(to_compute_variants), batch_size):
        batch_vars = to_compute_variants[i : i + batch_size]
        batch_hashes = to_compute_hashes[i : i + batch_size]
        batch_seqs = [v.mutated_sequence for v in batch_vars]

        # 1. Embeddings
        emb_res = plm.embed(batch_seqs, batch_size=batch_size)

        # 2. Score mutations & PLL
        for v, seq_hash, pooled_emb in zip(batch_vars, batch_hashes, emb_res.sequence_embeddings):
            # Compute score
            mm_score = None
            if v.mutations:
                score_res = plm.score_combinatorial_variant(
                    reference_sequence, v.mutations, method="additive"
                )
                mm_score = score_res.score

            cache.put_embedding(
                seq_hash=seq_hash,
                embedding=pooled_emb.numpy(),
                length=len(v.mutated_sequence),
                model_name=checkpoint,
                masked_marginal_score=mm_score,
                extra_metadata={"variant_id": v.variant_id, "mutations": v.mutation_string},
            )
            newly_computed += 1

    elapsed = time.time() - start_time
    total = len(variants)
    rate = total / elapsed if elapsed > 0 else 0.0

    logger.info(
        f"Extraction finished in {elapsed:.2f}s ({rate:.1f} seqs/s). "
        f"Cache Hits: {cache_hits}, Computed: {newly_computed}."
    )


def main():
    parser = argparse.ArgumentParser(description="Extract ESM-2 embeddings and scores.")
    parser.add_argument("--csv", type=Path, default=Path("tests/fixtures/sample_gfp_dms.csv"))
    parser.add_argument("--cache_dir", type=Path, default=Path("data/interim/cache"))
    parser.add_argument("--checkpoint", type=str, default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    # Load sample CSV
    variants: list[VariantRecord] = []
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            mut_str = row.get("mutant", "").strip()
            seq = row.get("mutated_sequence", "").strip()
            muts = parse_mutation_string(mut_str)
            v = VariantRecord(
                variant_id=f"var_{idx}_{mut_str}",
                protein_id="GFP",
                mutations=muts,
                mutated_sequence=seq,
                is_wildtype=len(muts) == 0,
                num_mutations=len(muts),
            )
            variants.append(v)

    wt_seq = variants[0].mutated_sequence if variants[0].is_wildtype else variants[0].mutated_sequence
    extract_and_cache(wt_seq, variants, args.cache_dir, checkpoint=args.checkpoint, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
