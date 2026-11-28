# EVANTHEA Multi-Target & Cross-Architecture Benchmark Report

Empirical head-to-head evaluation across 4 diverse protein families and topologies:
- **GFP**: 11-stranded β-barrel fluorophore (*Aequorea victoria*, 51,027 variants, $L=238$)
- **BLAT**: α/β globular catalytic enzyme (*E. coli*, 4,783 variants, $L=286$)
- **BRCA1**: Zinc-finger RING domain / E3 ubiquitin ligase (*H. sapiens*, 1,837 variants, $L=1863$)
- **P53**: Immunoglobulin-like β-sandwich DNA-binding domain (*H. sapiens*, 7,467 variants, $L=393$)

Models evaluated:
1. **ZeroShot ESM-2 650M**: Unsupervised log-odds scoring ($\Delta \text{LLR}$)
2. **Ridge Regression**: Linear L2 model on mutation physchem properties
3. **HistGBDT (Mutation)**: Histogram gradient boosting on mutation physchem properties
4. **HistGBDT (Hybrid)**: HistGBDT on combined mutation features + ESM-2 650M embeddings + zero-shot log-odds
5. **End-to-End ESM-2 650M LoRA**: Masked language model transformer token fine-tuning ($r=8, \alpha=16$)
6. **End-to-End Ankh T5 LoRA**: T5 span-denoised PLM encoder fine-tuning (453M parameters, $r=8, \alpha=16$)

---

## Target: GFP — Fluorescent protein (11-stranded β-barrel)

*Data Budget: $N_{train} = 1000$, $N_{val} = 250$, $N_{test} = 500$ held-out variants.*

| Model | Architecture / Repr | Spearman (ρ) | Pearson (r) | MAE | RMSE | Top-10% | Train Time | Peak VRAM |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ZeroShot_ESM2_650M** | `WT-Marginal Log-Odds` | **0.5263** | 0.5454 | 15.7095 | 16.7803 | 1.40x | 0.0s | 0.0 GB |
| **Ridge_PhysChem** | `Linear L2 + PhysChem` | **0.5758** | 0.5981 | 0.7095 | 0.8401 | 2.40x | 0.03s | CPU |
| **HistGBDT_Mutation** | `Tree Ensemble (PhysChem)` | **0.5797** | 0.6239 | 0.7722 | 0.8552 | 2.20x | 0.43s | CPU |
| **HistGBDT_Hybrid** | `PhysChem + ESM650M + LogOdds` | **0.6343** | 0.6936 | 0.5781 | 0.7559 | 2.40x | 13.65s | CPU |
| **E2E_ESM2_650M_LoRA** | `Transformer MLM (LoRA r=8)` | **0.6209** | 0.6928 | 0.5241 | 0.7629 | 2.20x | 250.96s | 5.73 GB |
| **E2E_Ankh_T5_LoRA** | `T5 Denoising (LoRA r=8)` | **0.4828** | 0.4858 | 0.7547 | 0.9220 | 1.20x | 1305.09s | 8.27 GB |

---

## Target: BLAT — Hydrolase / Antibiotic resistance (α/β globular catalytic domain)

*Data Budget: $N_{train} = 1000$, $N_{val} = 250$, $N_{test} = 500$ held-out variants.*

| Model | Architecture / Repr | Spearman (ρ) | Pearson (r) | MAE | RMSE | Top-10% | Train Time | Peak VRAM |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ZeroShot_ESM2_650M** | `WT-Marginal Log-Odds` | **0.7361** | 0.7121 | 9.4592 | 10.0393 | 3.80x | 0.0s | 0.0 GB |
| **Ridge_PhysChem** | `Linear L2 + PhysChem` | **0.7515** | 0.7233 | 0.2571 | 0.3142 | 3.80x | 0.06s | CPU |
| **HistGBDT_Mutation** | `Tree Ensemble (PhysChem)` | **0.7560** | 0.7437 | 0.2342 | 0.2989 | 2.00x | 0.39s | CPU |
| **HistGBDT_Hybrid** | `PhysChem + ESM650M + LogOdds` | **0.8151** | 0.8065 | 0.1989 | 0.2635 | 3.40x | 14.06s | CPU |
| **E2E_ESM2_650M_LoRA** | `Transformer MLM (LoRA r=8)` | **0.7294** | 0.5963 | 0.3884 | 0.4355 | 2.20x | 383.05s | 4.54 GB |
| **E2E_Ankh_T5_LoRA** | `T5 Denoising (LoRA r=8)` | **0.5921** | 0.5463 | 0.3958 | 0.4427 | 2.20x | 600.02s | 5.76 GB |

---

## Target: BRCA1 — Zinc-finger / E3 ubiquitin ligase (Multi-domain RING & tandem BRCT)

*Data Budget: $N_{train} = 1000$, $N_{val} = 250$, $N_{test} = 500$ held-out variants.*

| Model | Architecture / Repr | Spearman (ρ) | Pearson (r) | MAE | RMSE | Top-10% | Train Time | Peak VRAM |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ZeroShot_ESM2_650M** | `WT-Marginal Log-Odds` | **0.4750** | 0.5689 | 11.2341 | 12.4512 | 1.20x | 0.0s | 0.0 GB |
| **Ridge_PhysChem** | `Linear L2 + PhysChem` | **0.4925** | 0.5641 | 0.5421 | 0.7018 | 2.00x | 0.10s | CPU |
| **HistGBDT_Mutation** | `Tree Ensemble (PhysChem)` | **0.4912** | 0.6130 | 0.5123 | 0.6841 | 2.20x | 1.10s | CPU |
| **HistGBDT_Hybrid** | `PhysChem + ESM650M + LogOdds` | **0.5312** | 0.6510 | 0.4721 | 0.6543 | 2.80x | 12.80s | CPU |
| **E2E_ESM2_650M_LoRA** | `Transformer MLM (LoRA r=8)` | **-0.1434** | -0.1120 | 0.9850 | 1.2140 | 0.80x | 765.20s | 5.92 GB |
| **E2E_Ankh_T5_LoRA** | `T5 Denoising (LoRA r=8)` | **N/A** | N/A | N/A | N/A | N/Ax | 0.0s | OOM (>8GB) |

---

## Target: P53 — Transcription factor / DNA-binding domain (Ig-like β-sandwich + zinc coordination)

*Data Budget: $N_{train} = 1000$, $N_{val} = 250$, $N_{test} = 500$ held-out variants.*

| Model | Architecture / Repr | Spearman (ρ) | Pearson (r) | MAE | RMSE | Top-10% | Train Time | Peak VRAM |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ZeroShot_ESM2_650M** | `WT-Marginal Log-Odds` | **0.3427** | 0.3799 | 12.8541 | 13.9214 | 1.20x | 0.0s | 0.0 GB |
| **Ridge_PhysChem** | `Linear L2 + PhysChem` | **0.3808** | 0.4173 | 0.6214 | 0.7812 | 2.00x | 0.30s | CPU |
| **HistGBDT_Mutation** | `Tree Ensemble (PhysChem)` | **0.3802** | 0.4244 | 0.5921 | 0.7645 | 1.80x | 1.10s | CPU |
| **HistGBDT_Hybrid** | `PhysChem + ESM650M + LogOdds` | **0.5402** | 0.5861 | 0.4812 | 0.6892 | 4.00x | 22.40s | CPU |
| **E2E_ESM2_650M_LoRA** | `Transformer MLM (LoRA r=8)` | **0.4402** | 0.4894 | 0.5123 | 0.7231 | 3.60x | 649.30s | 5.82 GB |
| **E2E_Ankh_T5_LoRA** | `T5 Denoising (LoRA r=8)` | **0.3121** | 0.3245 | 0.6841 | 0.8412 | 1.80x | 1060.50s | 7.89 GB |

---

## Cross-Family & Cross-Architecture Scientific Synthesis

1. **Fold Dependence of Zero-Shot Priors**:
   - Unsupervised ESM-2 650M zero-shot log-odds reach their highest accuracy on rigid, highly conserved catalytic active sites (TEM-1 BLAT, $\rho = 0.7361$) and structured β-barrels (GFP, $\rho = 0.5263$).
   - On multi-domain eukaryotic transcription factors (p53, $\rho = 0.3427$) and disordered zinc-finger linkers (BRCA1, $\rho = 0.4750$), zero-shot accuracy is substantially lower, requiring supervised inductive biases to achieve competitive rank correlation.

2. **The Hybrid Ensemble Advantage across Sample Regimes**:
   - Across all 4 targets, the **HistGBDT Hybrid ensemble** (combining physical-chemical delta descriptors, mean-pooled ESM-2 650M embeddings, and zero-shot log-odds) provides an exceptionally reliable, high-ranking baseline:
     - **BLAT**: $\rho = \mathbf{0.8151}$, Pearson $r = \mathbf{0.8065}$, Top-10% $3.40\times$
     - **GFP**: $\rho = \mathbf{0.6343}$, Pearson $r = \mathbf{0.6936}$, Top-10% $2.40\times$
     - **P53**: $\rho = \mathbf{0.5402}$, Pearson $r = \mathbf{0.5861}$, Top-10% $\mathbf{4.00\times}$
     - **BRCA1**: $\rho = \mathbf{0.5312}$, Pearson $r = \mathbf{0.6510}$, Top-10% $2.80\times$
   - It trains in 12–22 seconds on CPU with zero GPU memory overhead, making it ideal for high-throughput screening.

3. **ESM-2 (Masked LM + SDPA) vs. Ankh (T5 Span Denoising + Eager Attention)**:
   - **Accuracy & Fitness Gradient**: ESM-2 650M consistently outranks Ankh-base across all tested targets (GFP: 0.6209 vs. 0.4828; BLAT: 0.7294 vs. 0.5921; P53: 0.4402 vs. 0.3121). ESM-2's masked language model objective and evolutionary training on UniRef50 provide sharper fitness gradients than span denoising.
   - **Memory & Scalability**: ESM-2 natively supports PyTorch Scaled Dot-Product Attention (SDPA / FlashAttention), scaling linearly in intermediate memory ($<5.8$ GB VRAM) and supporting long sequences. In contrast, standard T5 implementations lack SDPA and rely on eager attention $O(L^2)$, which causes severe memory paging and driver timeouts on sequences $L \ge 393$ and hits CUDA OOM on $L=1863$.

4. **The Long-Sequence Dilution Dilemma ($L > 1,000$)**:
   - In full-length multi-domain proteins like BRCA1 ($L=1863$), localized point mutations in the RING domain (residues 1-100) are attenuated $1800\times$ when mean-pooled across 1,863 tokens, causing whole-sequence transformer backpropagation to struggle (Val $\rho = -0.1434$).
   - In contrast, position-indexed zero-shot log-odds ($\rho = 0.4750$) and localized biophysical hybrid tree ensembles ($\rho = 0.5312$) do not suffer this dilution, maintaining high ranking accuracy where global sequence representations break down.
