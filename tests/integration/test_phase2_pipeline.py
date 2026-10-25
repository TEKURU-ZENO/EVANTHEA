"""Integration test running the full Phase 2 fitness prediction pipeline end-to-end."""

import shutil
from pathlib import Path
import numpy as np
import pytest

from evanthea.data.ingestion import ProteinGymIngestor
from evanthea.data.schemas import ProteinRecord
from evanthea.data.splits import create_mutation_order_split, create_position_held_out_split, create_random_split
from evanthea.evaluation.benchmark import run_split_benchmark
from evanthea.predictors.features import FeatureConfig
from evanthea.protein.sequence import compute_sequence_hash

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "fixtures" / "sample_gfp_dms.csv"

GFP_WT = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQ"
    "HDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNG"
    "IKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)


def test_phase2_end_to_end_pipeline():
    """Runs complete Phase 2 benchmark pipeline on GFP dataset fixture."""
    protein = ProteinRecord(
        id="GFP_AEQVI_Sarkisyan_2016",
        name="GFP",
        wildtype_sequence=GFP_WT,
        length=len(GFP_WT),
    )

    ingestor = ProteinGymIngestor(protein=protein)
    variants, summary = ingestor.ingest_csv(SAMPLE_CSV)
    assert summary.valid_records == 10

    # Create synthetic mock embeddings (D=32) for all 10 variants to run offline
    D = 32
    np.random.seed(42)
    mock_embeddings = {}
    for v in variants:
        h = compute_sequence_hash(v.mutated_sequence)
        # Add correlation with raw_score so predictors have signal
        base_vec = np.random.randn(D).astype(np.float32) * 0.1
        base_vec[0] += (v.raw_score if v.raw_score is not None else 0.0)
        mock_embeddings[h] = base_vec

    wt_hash = compute_sequence_hash(GFP_WT)
    if wt_hash not in mock_embeddings:
        mock_embeddings[wt_hash] = np.zeros(D, dtype=np.float32)

    # 1. Random Split Benchmark
    random_split = create_random_split(protein.id, variants, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)

    results_random = run_split_benchmark(
        reference_protein=protein,
        all_variants=variants,
        split_manifest=random_split,
        feature_config=FeatureConfig.EMB_DIFF,
        target_scaling="identity",
        provided_embeddings=mock_embeddings,
        models_to_run=["Mean", "Ridge", "HistGBDT"],
    )
    assert len(results_random) == 3
    for res in results_random:
        assert res.test_metrics.n_samples == len(random_split.test_variant_ids)
        assert np.isfinite(res.test_metrics.mae)
        assert np.isfinite(res.test_metrics.rmse)

    # 2. Position-Held-Out Benchmark
    pos_split = create_position_held_out_split(
        protein.id, variants, sequence_length=protein.length, test_position_fraction=0.3, seed=42
    )
    results_pos = run_split_benchmark(
        reference_protein=protein,
        all_variants=variants,
        split_manifest=pos_split,
        feature_config=FeatureConfig.HYBRID,
        target_scaling="standard",
        provided_embeddings=mock_embeddings,
        models_to_run=["Ridge"],
    )
    assert len(results_pos) == 1
    assert results_pos[0].split_strategy == "position_held_out"

    # 3. Mutation-Order Benchmark
    order_split = create_mutation_order_split(protein.id, variants, seed=42)
    results_order = run_split_benchmark(
        reference_protein=protein,
        all_variants=variants,
        split_manifest=order_split,
        feature_config=FeatureConfig.MUTATION_FEATURES,
        target_scaling="identity",
        provided_embeddings=mock_embeddings,
        models_to_run=["Ridge"],
    )
    assert len(results_order) == 1
    assert results_order[0].split_strategy == "mutation_order"
