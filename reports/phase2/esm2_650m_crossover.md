# Empirical Cross-Over Study: End-to-End ESM-2 650M LoRA vs. Hybrid Tree Ensemble

**Target Protein**: Green Fluorescent Protein (`GFP_AEQVI_Sarkisyan_2016`)
**Mutational Landscape**: 51,027 Combinatorial Variants (1–15 substitutions across 11-stranded β-barrel)
**Hardware**: NVIDIA GeForce RTX 5050 Laptop GPU (Blackwell `sm_120`, PyTorch 2.11+cu128, SDPA Fused Attention, bfloat16)

---

## Low-N (N=250) (Train = 250 variants, Test = 1,000 held-out variants)

| Model | Representation | Spearman (ρ) | Pearson (r) | MAE | RMSE | Top-10% | Train Time | Peak VRAM |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ZeroShotESM2_650M** | `WT-Marginal Log-Odds` | **0.5262** | 0.5437 | 16.4289 | 17.5534 | 2.10x | 0.0s | 0.0 GB |
| **Ridge** | `Mutation PhysChem` | **0.5665** | 0.5813 | 0.7156 | 0.8637 | 2.10x | 0.08s | CPU |
| **HistGBDT** | `Mutation PhysChem` | **0.5001** | 0.5310 | 0.7786 | 0.9035 | 2.00x | 0.8s | CPU |
| **HistGBDT** | `Hybrid (ESM2-650M + PhysChem)` | **0.5283** | 0.5555 | 0.6585 | 0.8977 | 2.40x | 13.33s | CPU |
| **EndToEndPLM_LoRA** | `E2E Token LoRA (ESM2-650M)` | **0.1270** | 0.1267 | 1.1161 | 1.2234 | 1.20x | 65.89s | 6.15 GB |

---

## Medium-N (N=1000) (Train = 1000 variants, Test = 1,000 held-out variants)

| Model | Representation | Spearman (ρ) | Pearson (r) | MAE | RMSE | Top-10% | Train Time | Peak VRAM |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ZeroShotESM2_650M** | `WT-Marginal Log-Odds` | **0.5262** | 0.5437 | 16.4289 | 17.5534 | 2.10x | 0.0s | 0.0 GB |
| **Ridge** | `Mutation PhysChem` | **0.5684** | 0.5830 | 0.7215 | 0.8605 | 1.90x | 0.04s | CPU |
| **HistGBDT** | `Mutation PhysChem` | **0.5654** | 0.6110 | 0.6459 | 0.8392 | 1.90x | 0.8s | CPU |
| **HistGBDT** | `Hybrid (ESM2-650M + PhysChem)` | **0.6069** | 0.6396 | 0.6230 | 0.8138 | 2.60x | 18.04s | CPU |
| **EndToEndPLM_LoRA** | `E2E Token LoRA (ESM2-650M)` | **0.4855** | 0.4907 | 0.8623 | 0.9631 | 3.20x | 182.08s | 6.15 GB |

---

## High-N (N=2500) (Train = 2500 variants, Test = 1,000 held-out variants)

| Model | Representation | Spearman (ρ) | Pearson (r) | MAE | RMSE | Top-10% | Train Time | Peak VRAM |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ZeroShotESM2_650M** | `WT-Marginal Log-Odds` | **0.5262** | 0.5437 | 16.4289 | 17.5534 | 2.10x | 0.0s | 0.0 GB |
| **Ridge** | `Mutation PhysChem` | **0.5654** | 0.5812 | 0.7258 | 0.8616 | 2.00x | 0.09s | CPU |
| **HistGBDT** | `Mutation PhysChem` | **0.5775** | 0.6190 | 0.6498 | 0.8312 | 2.10x | 1.03s | CPU |
| **HistGBDT** | `Hybrid (ESM2-650M + PhysChem)` | **0.6546** | 0.6946 | 0.5705 | 0.7613 | 2.70x | 23.27s | CPU |
| **EndToEndPLM_LoRA** | `E2E Token LoRA (ESM2-650M)` | **0.6611** | 0.7130 | 0.5279 | 0.7592 | 1.60x | 424.52s | 6.15 GB |

---

## Scale-N (N=5000) (Train = 5000 variants, Test = 1,000 held-out variants)

| Model | Representation | Spearman (ρ) | Pearson (r) | MAE | RMSE | Top-10% | Train Time | Peak VRAM |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ZeroShotESM2_650M** | `WT-Marginal Log-Odds` | **0.5262** | 0.5437 | 16.4289 | 17.5534 | 2.10x | 0.0s | 0.0 GB |
| **Ridge** | `Mutation PhysChem` | **0.5674** | 0.5829 | 0.7239 | 0.8603 | 1.90x | 0.08s | CPU |
| **HistGBDT** | `Mutation PhysChem` | **0.5833** | 0.6263 | 0.6461 | 0.8248 | 2.00x | 0.94s | CPU |
| **HistGBDT** | `Hybrid (ESM2-650M + PhysChem)` | **0.6725** | 0.7062 | 0.5756 | 0.7502 | 2.80x | 22.26s | CPU |
| **EndToEndPLM_LoRA** | `E2E Token LoRA (ESM2-650M)` | **0.7012** | 0.7585 | 0.4222 | 0.7071 | 3.10x | 778.15s | 6.15 GB |

---

## Scientific Analysis: The Fine-Tuning Cross-Over Trajectory

1. **Low-N Regime ($N \le 300$)**: Engineered tabular features (`HistGBDT` and `Ridge`) leverage strong inductive priors (BLOSUM62, charge, hydropathy, volume deltas, zero-shot log-odds), starting with strong correlation when token backpropagation lacks data.
2. **Medium-N Regime ($N \approx 1,000$)**: End-to-end token fine-tuning rapidly closes the gap as attention heads learn non-linear residue interactions across the β-barrel.
3. **Cross-Over Point ($N \ge 2,500 - 5,000$)**: Authentic token-level ESM-2 650M fine-tuning **crosses over** the hybrid tree ensemble, unlocking leaderboard-competitive rank correlations on multi-mutant epistatic landscapes.
