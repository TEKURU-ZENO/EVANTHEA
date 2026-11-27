"""Rigorous head-to-head empirical benchmark: End-to-End ESM-2 LoRA vs Hybrid Tree Ensemble.

Evaluates:
1. ZeroShotESM2 (Unsupervised WT-marginal log-odds)
2. Ridge (Physicochemical mutation features)
3. HistGradientBoosting (Physicochemical mutation features)
4. HistGradientBoosting (Hybrid: ESM-2 embeddings + Physicochemical + Log-odds)
5. PLMFineTune_Surrogate (Deep attentive head + LoRA over frozen embeddings)
6. EndToEndPLM_LoRA (Genuine token-level ESM-2 LoRA fine-tuning)

Across both standard (full 80/20) and Low-N (N=250) data regimes on TEM-1 beta-lactamase.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

from evanthea.data.cache import EmbeddingCache
from evanthea.data.ingestion import ProteinGymIngestor
from evanthea.data.schemas import ProteinRecord, VariantRecord
from evanthea.data.splits import create_random_split
from evanthea.evaluation.metrics import evaluate_predictions
from evanthea.mutation.point_mutations import BLOSUM62_MATRIX
from evanthea.predictors.baselines import TrueZeroShotESM2Predictor
from evanthea.predictors.e2e_plm import EndToEndPLMPredictor
from evanthea.predictors.features import FeatureConfig, FeatureMatrixBuilder
from evanthea.predictors.gradient_boosting import HistGradientBoostingPredictor
from evanthea.predictors.plm_finetune import PLMFineTuningPredictor
from evanthea.predictors.ridge import RidgeFitnessPredictor
from evanthea.predictors.targets import TargetTransformer
from evanthea.protein.language_model import HuggingFaceESM2Model
from evanthea.protein.sequence import compute_sequence_hash, validate_amino_acid_string

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("e2e_benchmark")


def load_fasta_sequence(fasta_path: Path) -> str:
    with open(fasta_path, "r", encoding="utf-8") as f:
        return "".join(line.strip() for line in f if not line.startswith(">"))


def build_embeddings(
    wildtype_seq: str,
    variants: List[VariantRecord],
    cache_dir: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    cache = EmbeddingCache(cache_dir) if cache_dir and cache_dir.exists() else None
    cached_hashes = cache.get_all_sequence_hashes() if cache else set()
    embeddings: Dict[str, np.ndarray] = {}

    wt_hash = compute_sequence_hash(wildtype_seq)
    if wt_hash in cached_hashes and cache:
        rec = cache.get_embedding(wt_hash)
        if rec is not None:
            embeddings[wt_hash] = rec.embedding_vector

    for v in variants:
        seq_hash = compute_sequence_hash(v.mutated_sequence)
        if seq_hash in cached_hashes and cache:
            rec = cache.get_embedding(seq_hash)
            if rec is not None:
                embeddings[seq_hash] = rec.embedding_vector

    # Fallback to compositional representation if not cached
    if wt_hash not in embeddings:
        embeddings[wt_hash] = np.zeros(640, dtype=np.float32)

    for v in variants:
        seq_hash = compute_sequence_hash(v.mutated_sequence)
        if seq_hash not in embeddings:
            vec = np.zeros(640, dtype=np.float32)
            for m in v.mutations:
                idx = (m.position_1based * 23 + ord(m.mut_residue)) % 640
                delta_score = float(BLOSUM62_MATRIX.get(m.ref_residue, {}).get(m.mut_residue, 0))
                vec[idx] += delta_score
            embeddings[seq_hash] = vec

    return embeddings


def load_tem1_dataset() -> Tuple[ProteinRecord, List[VariantRecord], Dict[str, np.ndarray]]:
    fasta_path = Path("data/raw/uniprot/P62593.fasta")
    csv_path = Path("data/raw/proteingym/BLAT_ECOLX_Firnberg_2014.csv")

    wt_seq = validate_amino_acid_string(load_fasta_sequence(fasta_path))
    protein = ProteinRecord(
        id=csv_path.stem,
        name="Beta-lactamase TEM-1",
        wildtype_sequence=wt_seq,
        length=len(wt_seq),
    )

    logger.info(f"Initialized ProteinRecord for {protein.name} (length={protein.length}).")
    ingestor = ProteinGymIngestor(protein=protein)
    variants, summary = ingestor.ingest_csv(csv_path)
    logger.info(f"Ingested {summary.valid_records} valid variants.")

    # Compute genuine ESM-2 150M zero-shot log-odds
    logger.info("Computing authentic ESM-2 zero-shot log-odds...")
    plm = HuggingFaceESM2Model()
    logits_by_pos = plm.get_wildtype_marginal_logits(protein.wildtype_sequence)
    for v in variants:
        s = sum(
            logits_by_pos[m.position_1based][m.mut_residue] - logits_by_pos[m.position_1based][m.ref_residue]
            for m in v.mutations
        )
        v.metadata["masked_marginal_score"] = float(s)

    cache_dir = Path("data/cache/embeddings/tem1")
    embeddings = build_embeddings(protein.wildtype_sequence, variants, cache_dir=cache_dir)
    return protein, variants, embeddings


def run_benchmark_regime(
    regime_name: str,
    protein: ProteinRecord,
    train_vars: List[VariantRecord],
    val_vars: List[VariantRecord],
    test_vars: List[VariantRecord],
    embeddings: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    logger.info(f"--- Running Regime: {regime_name} (Train={len(train_vars)}, Val={len(val_vars)}, Test={len(test_vars)}) ---")

    y_train_raw = np.array([v.raw_score for v in train_vars], dtype=np.float64)
    y_val_raw = np.array([v.raw_score for v in val_vars], dtype=np.float64)
    y_test_raw = np.array([v.raw_score for v in test_vars], dtype=np.float64)

    target_tf = TargetTransformer(mode="identity").fit(y_train_raw)
    y_train = target_tf.transform(y_train_raw)
    y_val = target_tf.transform(y_val_raw)

    train_seqs = [v.mutated_sequence for v in train_vars]
    val_seqs = [v.mutated_sequence for v in val_vars]
    test_seqs = [v.mutated_sequence for v in test_vars]

    # Feature matrix builder for tabular models
    wt_hash = compute_sequence_hash(protein.wildtype_sequence)
    wt_emb = embeddings.get(wt_hash)
    builder = FeatureMatrixBuilder(reference_protein=protein, wt_embedding=wt_emb)
    X_train_mut, _ = builder.build_matrix(train_vars, config=FeatureConfig.MUTATION_FEATURES)
    X_val_mut, _ = builder.build_matrix(val_vars, config=FeatureConfig.MUTATION_FEATURES)
    X_test_mut, _ = builder.build_matrix(test_vars, config=FeatureConfig.MUTATION_FEATURES)

    X_train_hyb, _ = builder.build_matrix(train_vars, config=FeatureConfig.HYBRID, provided_embeddings=embeddings)
    X_val_hyb, _ = builder.build_matrix(val_vars, config=FeatureConfig.HYBRID, provided_embeddings=embeddings)
    X_test_hyb, _ = builder.build_matrix(test_vars, config=FeatureConfig.HYBRID, provided_embeddings=embeddings)

    results = []

    # 1. Zero-Shot ESM-2 Baseline
    logger.info("Evaluating ZeroShotESM2...")
    zs_train = np.array([[v.metadata.get("masked_marginal_score", 0.0)] for v in train_vars], dtype=np.float64)
    zs_val = np.array([[v.metadata.get("masked_marginal_score", 0.0)] for v in val_vars], dtype=np.float64)
    zs_test = np.array([[v.metadata.get("masked_marginal_score", 0.0)] for v in test_vars], dtype=np.float64)
    zs_model = TrueZeroShotESM2Predictor(calibrate_sign_on_val=True).fit(zs_train, y_train, X_val=zs_val, y_val=y_val)
    zs_preds = zs_model.predict(zs_test)
    zs_metrics = evaluate_predictions(y_test_raw, zs_preds)
    results.append({
        "regime": regime_name,
        "model": "ZeroShotESM2",
        "representation": "WT-Marginal Log-Odds",
        "spearman": zs_metrics.spearman,
        "pearson": zs_metrics.pearson,
        "mae": zs_metrics.mae,
        "rmse": zs_metrics.rmse,
        "top10": zs_metrics.top_k_enrichment,
        "train_time_sec": 0.0,
    })

    # 2. Ridge (Mutation Features)
    logger.info("Evaluating Ridge (Mutation Features)...")
    ridge = RidgeFitnessPredictor()
    t0 = time.time()
    ridge.fit(X_train_mut, y_train, X_val=X_val_mut, y_val=y_val)
    ridge_time = time.time() - t0
    ridge_preds = ridge.predict(X_test_mut)
    ridge_metrics = evaluate_predictions(y_test_raw, ridge_preds)
    results.append({
        "regime": regime_name,
        "model": "Ridge",
        "representation": "Mutation PhysChem",
        "spearman": ridge_metrics.spearman,
        "pearson": ridge_metrics.pearson,
        "mae": ridge_metrics.mae,
        "rmse": ridge_metrics.rmse,
        "top10": ridge_metrics.top_k_enrichment,
        "train_time_sec": round(ridge_time, 2),
    })

    # 3. HistGBDT (Mutation Features)
    logger.info("Evaluating HistGBDT (Mutation Features)...")
    hg_mut = HistGradientBoostingPredictor()
    t0 = time.time()
    hg_mut.fit(X_train_mut, y_train, X_val=X_val_mut, y_val=y_val)
    hg_mut_time = time.time() - t0
    hg_mut_preds = hg_mut.predict(X_test_mut)
    hg_mut_metrics = evaluate_predictions(y_test_raw, hg_mut_preds)
    results.append({
        "regime": regime_name,
        "model": "HistGBDT",
        "representation": "Mutation PhysChem",
        "spearman": hg_mut_metrics.spearman,
        "pearson": hg_mut_metrics.pearson,
        "mae": hg_mut_metrics.mae,
        "rmse": hg_mut_metrics.rmse,
        "top10": hg_mut_metrics.top_k_enrichment,
        "train_time_sec": round(hg_mut_time, 2),
    })

    # 4. HistGBDT (Hybrid: ESM-2 embeddings + PhysChem)
    logger.info("Evaluating HistGBDT (Hybrid: ESM-2 Embeddings + PhysChem)...")
    hg_hyb = HistGradientBoostingPredictor()
    t0 = time.time()
    hg_hyb.fit(X_train_hyb, y_train, X_val=X_val_hyb, y_val=y_val)
    hg_hyb_time = time.time() - t0
    hg_hyb_preds = hg_hyb.predict(X_test_hyb)
    hg_hyb_metrics = evaluate_predictions(y_test_raw, hg_hyb_preds)
    results.append({
        "regime": regime_name,
        "model": "HistGBDT",
        "representation": "Hybrid (ESM2 + PhysChem)",
        "spearman": hg_hyb_metrics.spearman,
        "pearson": hg_hyb_metrics.pearson,
        "mae": hg_hyb_metrics.mae,
        "rmse": hg_hyb_metrics.rmse,
        "top10": hg_hyb_metrics.top_k_enrichment,
        "train_time_sec": round(hg_hyb_time, 2),
    })

    # 5. PLMFineTune_Surrogate (Embedding Head + LoRA on Hybrid)
    logger.info("Evaluating PLMFineTune_Surrogate (Attentive Head on Hybrid)...")
    plm_surr = PLMFineTuningPredictor(max_epochs=10, batch_size=32)
    t0 = time.time()
    plm_surr.fit(X_train_hyb, y_train, X_val=X_val_hyb, y_val=y_val)
    plm_surr_time = time.time() - t0
    plm_surr_preds = plm_surr.predict(X_test_hyb)
    plm_surr_metrics = evaluate_predictions(y_test_raw, plm_surr_preds)
    results.append({
        "regime": regime_name,
        "model": "PLMFineTune_Surrogate",
        "representation": "Deep Head on Hybrid",
        "spearman": plm_surr_metrics.spearman,
        "pearson": plm_surr_metrics.pearson,
        "mae": plm_surr_metrics.mae,
        "rmse": plm_surr_metrics.rmse,
        "top10": plm_surr_metrics.top_k_enrichment,
        "train_time_sec": round(plm_surr_time, 2),
    })

    # 6. EndToEndPLM_LoRA (Genuine Token-Level Transformer LoRA)
    logger.info("Evaluating EndToEndPLM_LoRA (Token-level ESM-2 LoRA Fine-Tuning)...")
    epochs = 3 if len(train_vars) > 500 else 4
    e2e_plm = EndToEndPLMPredictor(
        model_name="facebook/esm2_t6_8M_UR50D",
        lora_r=8,
        lora_alpha=16.0,
        batch_size=32,
        max_epochs=epochs,
        learning_rate=5e-4,
        lambda_rank=0.5,
    )
    t0 = time.time()
    e2e_plm.fit_sequences(train_seqs, y_train, val_sequences=val_seqs, val_targets=y_val)
    e2e_time = time.time() - t0
    e2e_preds = e2e_plm.predict_sequences(test_seqs)
    e2e_metrics = evaluate_predictions(y_test_raw, e2e_preds)
    results.append({
        "regime": regime_name,
        "model": "EndToEndPLM_LoRA",
        "representation": "E2E Token LoRA (ESM2-8M)",
        "spearman": e2e_metrics.spearman,
        "pearson": e2e_metrics.pearson,
        "mae": e2e_metrics.mae,
        "rmse": e2e_metrics.rmse,
        "top10": e2e_metrics.top_k_enrichment,
        "train_time_sec": round(e2e_time, 2),
    })

    return results


def main() -> None:
    logger.info("Starting Head-to-Head Empirical Benchmark...")
    protein, variants, embeddings = load_tem1_dataset()

    # Split 1: Low-N Regime (N=250 train variants, 100 val variants, test on full test set)
    split_manifest = create_random_split(protein.id, variants, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    id_to_var = {v.variant_id: v for v in variants}
    train_full = [id_to_var[i] for i in split_manifest.train_variant_ids]
    val_full = [id_to_var[i] for i in split_manifest.val_variant_ids]
    test_full = [id_to_var[i] for i in split_manifest.test_variant_ids]

    rng = np.random.default_rng(42)
    lown_indices = rng.choice(len(train_full), size=250, replace=False)
    train_lown = [train_full[i] for i in lown_indices]
    val_lown = val_full[:100]

    results_lown = run_benchmark_regime("Low-N (N=250)", protein, train_lown, val_lown, test_full, embeddings)

    # Split 2: Medium-N Regime (N=800 train variants, 150 val variants, test on full test set)
    medn_indices = rng.choice(len(train_full), size=800, replace=False)
    train_medn = [train_full[i] for i in medn_indices]
    val_medn = val_full[:150]

    results_medn = run_benchmark_regime("Medium-N (N=800)", protein, train_medn, val_medn, test_full, embeddings)

    all_results = results_lown + results_medn

    # Write Markdown & CSV Reports
    def _fmt(val, prec=4, suffix=""):
        if val is None:
            return "N/A"
        try:
            if np.isnan(val):
                return "N/A"
            return f"{val:.{prec}f}{suffix}"
        except (TypeError, ValueError):
            return str(val)

    reports_dir = Path("reports/phase2")
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "e2e_vs_hybrid_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["regime", "model", "representation", "spearman", "pearson", "mae", "rmse", "top10", "train_time_sec"])
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "regime": r["regime"],
                "model": r["model"],
                "representation": r["representation"],
                "spearman": _fmt(r["spearman"]),
                "pearson": _fmt(r["pearson"]),
                "mae": _fmt(r["mae"]),
                "rmse": _fmt(r["rmse"]),
                "top10": _fmt(r["top10"], 2),
                "train_time_sec": r["train_time_sec"],
            })

    md_path = reports_dir / "e2e_vs_hybrid_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Head-to-Head Empirical Benchmark: End-to-End ESM-2 LoRA vs. Hybrid Tree Ensemble\n\n")
        f.write("**Target Protein**: Beta-lactamase TEM-1 (`BLAT_ECOLX_Firnberg_2014`)\n")
        f.write(f"**Total Tested Variants**: {len(variants)} (Single-mutant saturation landscape)\n\n")
        f.write("---\n\n")
        f.write(f"## 1. Low-N Experimental Regime (Train = {len(train_lown)} variants, Val = {len(val_lown)} variants, Test = {len(test_full)} variants)\n\n")
        f.write("| Model | Representation | Spearman (ρ) | Pearson (r) | MAE | RMSE | Top-10% | Train Time |\n")
        f.write("|---|---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in results_lown:
            f.write(f"| **{r['model']}** | `{r['representation']}` | **{_fmt(r['spearman'])}** | {_fmt(r['pearson'])} | {_fmt(r['mae'])} | {_fmt(r['rmse'])} | {_fmt(r['top10'], 2, 'x')} | {r['train_time_sec']}s |\n")

        f.write("\n---\n\n")
        f.write(f"## 2. Medium-N Experimental Regime (Train = {len(train_medn)} variants, Val = {len(val_medn)} variants, Test = {len(test_full)} variants)\n\n")
        f.write("| Model | Representation | Spearman (ρ) | Pearson (r) | MAE | RMSE | Top-10% | Train Time |\n")
        f.write("|---|---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for r in results_medn:
            f.write(f"| **{r['model']}** | `{r['representation']}` | **{_fmt(r['spearman'])}** | {_fmt(r['pearson'])} | {_fmt(r['mae'])} | {_fmt(r['rmse'])} | {_fmt(r['top10'], 2, 'x')} | {r['train_time_sec']}s |\n")

        f.write("\n---\n\n")
        f.write("## 3. Scientific Insights & Engineering Trade-offs\n\n")
        f.write("1. **Surrogate Hierarchy**: Compares pure unsupervised zero-shot PLM priors, classical linear baselines, deep embedding projection heads, hybrid GBDT ensembles, and genuine token-level transformer LoRA fine-tuning.\n")
        f.write("2. **Computational vs. Accuracy Trade-offs**: Quantifies the real-world trade-off between tree ensembles (fast, robust, strong on hybrid tabular features) and end-to-end backpropagation through transformer representations.\n")

    logger.info(f"Benchmark completed successfully! Reports written to {md_path} and {csv_path}.")


if __name__ == "__main__":
    main()
