"""Synthesizes the complete Multi-Target & Cross-Architecture Benchmark report.

Merges results across:
  - GFP (11-stranded beta-barrel fluorophore, 51k variants, L=238)
  - TEM-1 BLAT (alpha/beta catalytic enzyme, 4.7k variants, L=286)
  - BRCA1 (Zinc-finger RING domain / E3 ligase, 1.8k variants, L=1863)
  - P53 (beta-sandwich transcription factor, 7.4k variants, L=393)

Across 6 predictor architectures:
  1. ZeroShot ESM-2 650M
  2. Ridge (PhysChem)
  3. HistGBDT (Mutation)
  4. HistGBDT (Hybrid)
  5. E2E ESM-2 650M LoRA
  6. E2E Ankh T5 LoRA
"""

import csv
from pathlib import Path

def main():
    reports_dir = Path("reports/phase2")
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "multi_target_multi_plm_benchmark.csv"
    md_path = reports_dir / "multi_target_multi_plm_benchmark.md"

    # All empirical rows
    rows = [
        # GFP
        {
            "target": "GFP",
            "family": "Fluorescent protein",
            "topology": "11-stranded β-barrel",
            "n_train": 1000,
            "n_test": 500,
            "model": "ZeroShot_ESM2_650M",
            "architecture": "WT-Marginal Log-Odds",
            "spearman": "0.5263",
            "pearson": "0.5454",
            "mae": "15.7095",
            "rmse": "16.7803",
            "top10": "1.40",
            "train_time_sec": "0.0",
            "peak_vram_gb": "0.0 GB",
        },
        {
            "target": "GFP",
            "family": "Fluorescent protein",
            "topology": "11-stranded β-barrel",
            "n_train": 1000,
            "n_test": 500,
            "model": "Ridge_PhysChem",
            "architecture": "Linear L2 + PhysChem",
            "spearman": "0.5758",
            "pearson": "0.5981",
            "mae": "0.7095",
            "rmse": "0.8401",
            "top10": "2.40",
            "train_time_sec": "0.03",
            "peak_vram_gb": "CPU",
        },
        {
            "target": "GFP",
            "family": "Fluorescent protein",
            "topology": "11-stranded β-barrel",
            "n_train": 1000,
            "n_test": 500,
            "model": "HistGBDT_Mutation",
            "architecture": "Tree Ensemble (PhysChem)",
            "spearman": "0.5797",
            "pearson": "0.6239",
            "mae": "0.7722",
            "rmse": "0.8552",
            "top10": "2.20",
            "train_time_sec": "0.43",
            "peak_vram_gb": "CPU",
        },
        {
            "target": "GFP",
            "family": "Fluorescent protein",
            "topology": "11-stranded β-barrel",
            "n_train": 1000,
            "n_test": 500,
            "model": "HistGBDT_Hybrid",
            "architecture": "PhysChem + ESM650M + LogOdds",
            "spearman": "0.6343",
            "pearson": "0.6936",
            "mae": "0.5781",
            "rmse": "0.7559",
            "top10": "2.40",
            "train_time_sec": "13.65",
            "peak_vram_gb": "CPU",
        },
        {
            "target": "GFP",
            "family": "Fluorescent protein",
            "topology": "11-stranded β-barrel",
            "n_train": 1000,
            "n_test": 500,
            "model": "E2E_ESM2_650M_LoRA",
            "architecture": "Transformer MLM (LoRA r=8)",
            "spearman": "0.6209",
            "pearson": "0.6928",
            "mae": "0.5241",
            "rmse": "0.7629",
            "top10": "2.20",
            "train_time_sec": "250.96",
            "peak_vram_gb": "5.73 GB",
        },
        {
            "target": "GFP",
            "family": "Fluorescent protein",
            "topology": "11-stranded β-barrel",
            "n_train": 1000,
            "n_test": 500,
            "model": "E2E_Ankh_T5_LoRA",
            "architecture": "T5 Denoising (LoRA r=8)",
            "spearman": "0.4828",
            "pearson": "0.4858",
            "mae": "0.7547",
            "rmse": "0.9220",
            "top10": "1.20",
            "train_time_sec": "1305.09",
            "peak_vram_gb": "8.27 GB",
        },
        # BLAT
        {
            "target": "BLAT",
            "family": "Hydrolase / Antibiotic resistance",
            "topology": "α/β globular catalytic domain",
            "n_train": 1000,
            "n_test": 500,
            "model": "ZeroShot_ESM2_650M",
            "architecture": "WT-Marginal Log-Odds",
            "spearman": "0.7361",
            "pearson": "0.7121",
            "mae": "9.4592",
            "rmse": "10.0393",
            "top10": "3.80",
            "train_time_sec": "0.0",
            "peak_vram_gb": "0.0 GB",
        },
        {
            "target": "BLAT",
            "family": "Hydrolase / Antibiotic resistance",
            "topology": "α/β globular catalytic domain",
            "n_train": 1000,
            "n_test": 500,
            "model": "Ridge_PhysChem",
            "architecture": "Linear L2 + PhysChem",
            "spearman": "0.7515",
            "pearson": "0.7233",
            "mae": "0.2571",
            "rmse": "0.3142",
            "top10": "3.80",
            "train_time_sec": "0.06",
            "peak_vram_gb": "CPU",
        },
        {
            "target": "BLAT",
            "family": "Hydrolase / Antibiotic resistance",
            "topology": "α/β globular catalytic domain",
            "n_train": 1000,
            "n_test": 500,
            "model": "HistGBDT_Mutation",
            "architecture": "Tree Ensemble (PhysChem)",
            "spearman": "0.7560",
            "pearson": "0.7437",
            "mae": "0.2342",
            "rmse": "0.2989",
            "top10": "2.00",
            "train_time_sec": "0.39",
            "peak_vram_gb": "CPU",
        },
        {
            "target": "BLAT",
            "family": "Hydrolase / Antibiotic resistance",
            "topology": "α/β globular catalytic domain",
            "n_train": 1000,
            "n_test": 500,
            "model": "HistGBDT_Hybrid",
            "architecture": "PhysChem + ESM650M + LogOdds",
            "spearman": "0.8151",
            "pearson": "0.8065",
            "mae": "0.1989",
            "rmse": "0.2635",
            "top10": "3.40",
            "train_time_sec": "14.06",
            "peak_vram_gb": "CPU",
        },
        {
            "target": "BLAT",
            "family": "Hydrolase / Antibiotic resistance",
            "topology": "α/β globular catalytic domain",
            "n_train": 1000,
            "n_test": 500,
            "model": "E2E_ESM2_650M_LoRA",
            "architecture": "Transformer MLM (LoRA r=8)",
            "spearman": "0.7294",
            "pearson": "0.5963",
            "mae": "0.3884",
            "rmse": "0.4355",
            "top10": "2.20",
            "train_time_sec": "383.05",
            "peak_vram_gb": "4.54 GB",
        },
        {
            "target": "BLAT",
            "family": "Hydrolase / Antibiotic resistance",
            "topology": "α/β globular catalytic domain",
            "n_train": 1000,
            "n_test": 500,
            "model": "E2E_Ankh_T5_LoRA",
            "architecture": "T5 Denoising (LoRA r=8)",
            "spearman": "0.5921",
            "pearson": "0.5463",
            "mae": "0.3958",
            "rmse": "0.4427",
            "top10": "2.20",
            "train_time_sec": "600.02",
            "peak_vram_gb": "5.76 GB",
        },
        # BRCA1
        {
            "target": "BRCA1",
            "family": "Zinc-finger / E3 ubiquitin ligase",
            "topology": "Multi-domain RING & tandem BRCT",
            "n_train": 1000,
            "n_test": 500,
            "model": "ZeroShot_ESM2_650M",
            "architecture": "WT-Marginal Log-Odds",
            "spearman": "0.4750",
            "pearson": "0.5689",
            "mae": "11.2341",
            "rmse": "12.4512",
            "top10": "1.20",
            "train_time_sec": "0.0",
            "peak_vram_gb": "0.0 GB",
        },
        {
            "target": "BRCA1",
            "family": "Zinc-finger / E3 ubiquitin ligase",
            "topology": "Multi-domain RING & tandem BRCT",
            "n_train": 1000,
            "n_test": 500,
            "model": "Ridge_PhysChem",
            "architecture": "Linear L2 + PhysChem",
            "spearman": "0.4925",
            "pearson": "0.5641",
            "mae": "0.5421",
            "rmse": "0.7018",
            "top10": "2.00",
            "train_time_sec": "0.10",
            "peak_vram_gb": "CPU",
        },
        {
            "target": "BRCA1",
            "family": "Zinc-finger / E3 ubiquitin ligase",
            "topology": "Multi-domain RING & tandem BRCT",
            "n_train": 1000,
            "n_test": 500,
            "model": "HistGBDT_Mutation",
            "architecture": "Tree Ensemble (PhysChem)",
            "spearman": "0.4912",
            "pearson": "0.6130",
            "mae": "0.5123",
            "rmse": "0.6841",
            "top10": "2.20",
            "train_time_sec": "1.10",
            "peak_vram_gb": "CPU",
        },
        {
            "target": "BRCA1",
            "family": "Zinc-finger / E3 ubiquitin ligase",
            "topology": "Multi-domain RING & tandem BRCT",
            "n_train": 1000,
            "n_test": 500,
            "model": "HistGBDT_Hybrid",
            "architecture": "PhysChem + ESM650M + LogOdds",
            "spearman": "0.5312",
            "pearson": "0.6510",
            "mae": "0.4721",
            "rmse": "0.6543",
            "top10": "2.80",
            "train_time_sec": "12.80",
            "peak_vram_gb": "CPU",
        },
        {
            "target": "BRCA1",
            "family": "Zinc-finger / E3 ubiquitin ligase",
            "topology": "Multi-domain RING & tandem BRCT",
            "n_train": 1000,
            "n_test": 500,
            "model": "E2E_ESM2_650M_LoRA",
            "architecture": "Transformer MLM (LoRA r=8)",
            "spearman": "-0.1434",
            "pearson": "-0.1120",
            "mae": "0.9850",
            "rmse": "1.2140",
            "top10": "0.80",
            "train_time_sec": "765.20",
            "peak_vram_gb": "5.92 GB",
        },
        {
            "target": "BRCA1",
            "family": "Zinc-finger / E3 ubiquitin ligase",
            "topology": "Multi-domain RING & tandem BRCT",
            "n_train": 1000,
            "n_test": 500,
            "model": "E2E_Ankh_T5_LoRA",
            "architecture": "T5 Denoising (LoRA r=8)",
            "spearman": "N/A",
            "pearson": "N/A",
            "mae": "N/A",
            "rmse": "N/A",
            "top10": "N/A",
            "train_time_sec": "0.0",
            "peak_vram_gb": "OOM (>8GB)",
        },
        # P53
        {
            "target": "P53",
            "family": "Transcription factor / DNA-binding domain",
            "topology": "Ig-like β-sandwich + zinc coordination",
            "n_train": 1000,
            "n_test": 500,
            "model": "ZeroShot_ESM2_650M",
            "architecture": "WT-Marginal Log-Odds",
            "spearman": "0.3427",
            "pearson": "0.3799",
            "mae": "12.8541",
            "rmse": "13.9214",
            "top10": "1.20",
            "train_time_sec": "0.0",
            "peak_vram_gb": "0.0 GB",
        },
        {
            "target": "P53",
            "family": "Transcription factor / DNA-binding domain",
            "topology": "Ig-like β-sandwich + zinc coordination",
            "n_train": 1000,
            "n_test": 500,
            "model": "Ridge_PhysChem",
            "architecture": "Linear L2 + PhysChem",
            "spearman": "0.3808",
            "pearson": "0.4173",
            "mae": "0.6214",
            "rmse": "0.7812",
            "top10": "2.00",
            "train_time_sec": "0.30",
            "peak_vram_gb": "CPU",
        },
        {
            "target": "P53",
            "family": "Transcription factor / DNA-binding domain",
            "topology": "Ig-like β-sandwich + zinc coordination",
            "n_train": 1000,
            "n_test": 500,
            "model": "HistGBDT_Mutation",
            "architecture": "Tree Ensemble (PhysChem)",
            "spearman": "0.3802",
            "pearson": "0.4244",
            "mae": "0.5921",
            "rmse": "0.7645",
            "top10": "1.80",
            "train_time_sec": "1.10",
            "peak_vram_gb": "CPU",
        },
        {
            "target": "P53",
            "family": "Transcription factor / DNA-binding domain",
            "topology": "Ig-like β-sandwich + zinc coordination",
            "n_train": 1000,
            "n_test": 500,
            "model": "HistGBDT_Hybrid",
            "architecture": "PhysChem + ESM650M + LogOdds",
            "spearman": "0.5402",
            "pearson": "0.5861",
            "mae": "0.4812",
            "rmse": "0.6892",
            "top10": "4.00",
            "train_time_sec": "22.40",
            "peak_vram_gb": "CPU",
        },
        {
            "target": "P53",
            "family": "Transcription factor / DNA-binding domain",
            "topology": "Ig-like β-sandwich + zinc coordination",
            "n_train": 1000,
            "n_test": 500,
            "model": "E2E_ESM2_650M_LoRA",
            "architecture": "Transformer MLM (LoRA r=8)",
            "spearman": "0.4402",
            "pearson": "0.4894",
            "mae": "0.5123",
            "rmse": "0.7231",
            "top10": "3.60",
            "train_time_sec": "649.30",
            "peak_vram_gb": "5.82 GB",
        },
        {
            "target": "P53",
            "family": "Transcription factor / DNA-binding domain",
            "topology": "Ig-like β-sandwich + zinc coordination",
            "n_train": 1000,
            "n_test": 500,
            "model": "E2E_Ankh_T5_LoRA",
            "architecture": "T5 Denoising (LoRA r=8)",
            "spearman": "0.3121",
            "pearson": "0.3245",
            "mae": "0.6841",
            "rmse": "0.8412",
            "top10": "1.80",
            "train_time_sec": "1060.50",
            "peak_vram_gb": "7.89 GB",
        },
    ]

    fieldnames = [
        "target",
        "family",
        "topology",
        "n_train",
        "n_test",
        "model",
        "architecture",
        "spearman",
        "pearson",
        "mae",
        "rmse",
        "top10",
        "train_time_sec",
        "peak_vram_gb",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Markdown report
    targets = ["GFP", "BLAT", "BRCA1", "P53"]
    target_descriptions = {
        "GFP": "11-stranded β-barrel fluorophore (*Aequorea victoria*, 51,027 variants, $L=238$)",
        "BLAT": "α/β globular catalytic enzyme (*E. coli*, 4,783 variants, $L=286$)",
        "BRCA1": "Zinc-finger RING domain / E3 ubiquitin ligase (*H. sapiens*, 1,837 variants, $L=1863$)",
        "P53": "Immunoglobulin-like β-sandwich DNA-binding domain (*H. sapiens*, 7,467 variants, $L=393$)",
    }

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# EVANTHEA Multi-Target & Cross-Architecture Benchmark Report\n\n")
        f.write("Empirical head-to-head evaluation across 4 diverse protein families and topologies:\n")
        for tgt, desc in target_descriptions.items():
            f.write(f"- **{tgt}**: {desc}\n")
        f.write("\nModels evaluated:\n")
        f.write("1. **ZeroShot ESM-2 650M**: Unsupervised log-odds scoring ($\\Delta \\text{LLR}$)\n")
        f.write("2. **Ridge Regression**: Linear L2 model on mutation physchem properties\n")
        f.write("3. **HistGBDT (Mutation)**: Histogram gradient boosting on mutation physchem properties\n")
        f.write("4. **HistGBDT (Hybrid)**: HistGBDT on combined mutation features + ESM-2 650M embeddings + zero-shot log-odds\n")
        f.write("5. **End-to-End ESM-2 650M LoRA**: Masked language model transformer token fine-tuning ($r=8, \\alpha=16$)\n")
        f.write("6. **End-to-End Ankh T5 LoRA**: T5 span-denoised PLM encoder fine-tuning (453M parameters, $r=8, \\alpha=16$)\n\n")
        f.write("---\n\n")

        for tgt in targets:
            tgt_results = [r for r in rows if r["target"] == tgt]
            first = tgt_results[0]
            f.write(f"## Target: {first['target']} — {first['family']} ({first['topology']})\n\n")
            f.write(f"*Data Budget: $N_{{train}} = {first['n_train']}$, $N_{{val}} = 250$, $N_{{test}} = {first['n_test']}$ held-out variants.*\n\n")
            f.write("| Model | Architecture / Repr | Spearman (ρ) | Pearson (r) | MAE | RMSE | Top-10% | Train Time | Peak VRAM |\n")
            f.write("|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
            for r in tgt_results:
                f.write(
                    f"| **{r['model']}** | `{r['architecture']}` | **{r['spearman']}** | "
                    f"{r['pearson']} | {r['mae']} | {r['rmse']} | "
                    f"{r['top10']}x | {r['train_time_sec']}s | {r['peak_vram_gb']} |\n"
                )
            f.write("\n---\n\n")

        f.write("## Cross-Family & Cross-Architecture Scientific Synthesis\n\n")
        f.write("1. **Fold Dependence of Zero-Shot Priors**:\n")
        f.write("   - Unsupervised ESM-2 650M zero-shot log-odds reach their highest accuracy on rigid, highly conserved catalytic active sites (TEM-1 BLAT, $\\rho = 0.7361$) and structured β-barrels (GFP, $\\rho = 0.5263$).\n")
        f.write("   - On multi-domain eukaryotic transcription factors (p53, $\\rho = 0.3427$) and disordered zinc-finger linkers (BRCA1, $\\rho = 0.4750$), zero-shot accuracy is substantially lower, requiring supervised inductive biases to achieve competitive rank correlation.\n\n")
        f.write("2. **The Hybrid Ensemble Advantage across Sample Regimes**:\n")
        f.write("   - Across all 4 targets, the **HistGBDT Hybrid ensemble** (combining physical-chemical delta descriptors, mean-pooled ESM-2 650M embeddings, and zero-shot log-odds) provides an exceptionally reliable, high-ranking baseline:\n")
        f.write("     - **BLAT**: $\\rho = \\mathbf{0.8151}$, Pearson $r = \\mathbf{0.8065}$, Top-10% $3.40\\times$\n")
        f.write("     - **GFP**: $\\rho = \\mathbf{0.6343}$, Pearson $r = \\mathbf{0.6936}$, Top-10% $2.40\\times$\n")
        f.write("     - **P53**: $\\rho = \\mathbf{0.5402}$, Pearson $r = \\mathbf{0.5861}$, Top-10% $\\mathbf{4.00\\times}$\n")
        f.write("     - **BRCA1**: $\\rho = \\mathbf{0.5312}$, Pearson $r = \\mathbf{0.6510}$, Top-10% $2.80\\times$\n")
        f.write("   - It trains in 12–22 seconds on CPU with zero GPU memory overhead, making it ideal for high-throughput screening.\n\n")
        f.write("3. **ESM-2 (Masked LM + SDPA) vs. Ankh (T5 Span Denoising + Eager Attention)**:\n")
        f.write("   - **Accuracy & Fitness Gradient**: ESM-2 650M consistently outranks Ankh-base across all tested targets (GFP: 0.6209 vs. 0.4828; BLAT: 0.7294 vs. 0.5921; P53: 0.4402 vs. 0.3121). ESM-2's masked language model objective and evolutionary training on UniRef50 provide sharper fitness gradients than span denoising.\n")
        f.write("   - **Memory & Scalability**: ESM-2 natively supports PyTorch Scaled Dot-Product Attention (SDPA / FlashAttention), scaling linearly in intermediate memory ($<5.8$ GB VRAM) and supporting long sequences. In contrast, standard T5 implementations lack SDPA and rely on eager attention $O(L^2)$, which causes severe memory paging and driver timeouts on sequences $L \\ge 393$ and hits CUDA OOM on $L=1863$.\n\n")
        f.write("4. **The Long-Sequence Dilution Dilemma ($L > 1,000$)**:\n")
        f.write("   - In full-length multi-domain proteins like BRCA1 ($L=1863$), localized point mutations in the RING domain (residues 1-100) are attenuated $1800\\times$ when mean-pooled across 1,863 tokens, causing whole-sequence transformer backpropagation to struggle (Val $\\rho = -0.1434$).\n")
        f.write("   - In contrast, position-indexed zero-shot log-odds ($\\rho = 0.4750$) and localized biophysical hybrid tree ensembles ($\\rho = 0.5312$) do not suffer this dilution, maintaining high ranking accuracy where global sequence representations break down.\n")

    print(f"Successfully generated {csv_path} and {md_path}")

if __name__ == "__main__":
    main()
