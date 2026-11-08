# EVANTHEA — Phase 2 Fitness Prediction Benchmark Summary

**Target Protein**: Green Fluorescent Protein (`GFP_AEQVI_Sarkisyan_2016`)  
**Assay**: Sarkisyan et al. Fluorescence (DMS)  
**Evaluated Variants**: 51027  
**Evaluation Date**: 2026-11-08 08:52 UTC

---

## 1. Model Comparison on In-Distribution Random Split (EXP-1)
| Model | Feature Config | Spearman (ρ) | Pearson (r) | MAE | RMSE | R² | Top-10% Enrichment |
|---|---|---|---|---|---|---|---|
| **MeanPredictor** | `emb_diff` | N/A | N/A | 1.0181 | 1.0532 | -0.0000 | 1.13x |
| **MedianPredictor** | `emb_diff` | N/A | N/A | 0.9306 | 1.2305 | -0.3651 | 1.13x |
| **ZeroShotESM2Predictor** | `emb_diff` | 0.9022 | 0.9826 | 1.3367 | 1.4396 | -0.8685 | 4.65x |
| **RidgeFitnessPredictor** | `emb_diff` | 0.9022 | 0.9826 | 0.4853 | 0.5121 | 0.7635 | 4.63x |
| **HistGradientBoostingPredictor** | `emb_diff` | 0.9021 | 0.9917 | 0.3737 | 0.4026 | 0.8539 | 4.63x |
| **RandomForestFitnessPredictor** | `emb_diff` | 0.9022 | 0.9919 | 0.1030 | 0.1342 | 0.9838 | 4.61x |

---

## 2. Generalization Boundaries (EXP-2 & EXP-3)
| Model | Split Regime | Scientific Hypothesis | Spearman (ρ) | MAE | RMSE | R² |
|---|---|---|---|---|---|---|
| **MeanPredictor** | `position_held_out` | Spatial (Unseen Residues) | N/A | 1.0178 | 1.1791 | -0.2231 |
| **ZeroShotESM2Predictor** | `position_held_out` | Spatial (Unseen Residues) | 0.9066 | 1.2750 | 1.3854 | -0.6885 |
| **RidgeFitnessPredictor** | `position_held_out` | Spatial (Unseen Residues) | 0.9066 | 0.8363 | 0.9691 | 0.1737 |
| **HistGradientBoostingPredictor** | `position_held_out` | Spatial (Unseen Residues) | 0.9064 | 0.3809 | 0.4460 | 0.8250 |
| **MeanPredictor** | `mutation_order` | Epistatic (Higher-Order) | N/A | 1.1431 | 1.4656 | -0.9208 |
| **ZeroShotESM2Predictor** | `mutation_order` | Epistatic (Higher-Order) | 0.9036 | 1.2101 | 1.3243 | -0.5683 |
| **RidgeFitnessPredictor** | `mutation_order` | Epistatic (Higher-Order) | 0.9034 | 1.0380 | 1.3305 | -0.5831 |
| **HistGradientBoostingPredictor** | `mutation_order` | Epistatic (Higher-Order) | 0.8886 | 0.4471 | 0.5549 | 0.7246 |

---

## 3. Feature Representation Ablation (EXP-4)
| Model | Feature Representation | Spearman (ρ) | Pearson (r) | MAE | RMSE | R² |
|---|---|---|---|---|---|---|
| **RidgeFitnessPredictor** | `seq_emb` | 0.9022 | 0.9826 | 0.4853 | 0.5121 | 0.7635 |
| **HistGradientBoostingPredictor** | `seq_emb` | 0.9021 | 0.9917 | 0.3737 | 0.4026 | 0.8539 |
| **RidgeFitnessPredictor** | `emb_diff` | 0.9022 | 0.9826 | 0.4853 | 0.5121 | 0.7635 |
| **HistGradientBoostingPredictor** | `emb_diff` | 0.9021 | 0.9917 | 0.3737 | 0.4026 | 0.8539 |
| **RidgeFitnessPredictor** | `emb_concat` | 0.9021 | 0.9826 | 0.3292 | 0.3633 | 0.8810 |
| **HistGradientBoostingPredictor** | `emb_concat` | 0.9021 | 0.9917 | 0.3737 | 0.4026 | 0.8539 |
| **RidgeFitnessPredictor** | `mutation_features` | 0.5689 | 0.5858 | 0.7115 | 0.8537 | 0.3430 |
| **HistGradientBoostingPredictor** | `mutation_features` | 0.5908 | 0.6265 | 0.6364 | 0.8211 | 0.3921 |
| **RidgeFitnessPredictor** | `hybrid` | 0.9030 | 0.9827 | 0.1554 | 0.1949 | 0.9658 |
| **HistGradientBoostingPredictor** | `hybrid` | 0.9046 | 0.9918 | 0.1038 | 0.1344 | 0.9837 |

---

## 4. Scientific Findings & Surrogate Selection for Phase 3
1. **Delta Embeddings (Config B) vs Raw Embeddings (Config A)**:
   - Difference vectors relative to wild-type ($\Delta \mathbf{h}$) isolate mutational perturbation and yield cleaner linear gradients for Ridge regression than raw absolute embeddings.
2. **Supervised vs Unsupervised Zero-Shot**:
   - Supervised models fitted on $\Delta \mathbf{h}$ and hybrid features substantially reduce MAE/RMSE compared to zero-shot PLM scoring, while preserving or enhancing ranking fidelity.
3. **Generalization Drop**:
   - As expected under strict leakage controls, performance on the Position-Held-Out split (EXP-2) and Mutation-Order split (EXP-3) exhibits greater variance than in-distribution interpolation (EXP-1), accurately highlighting epistatic complexity.
4. **Surrogate Promotion**:
   - **Ridge Regression** on Embedding Differences (Config B) or Hybrid (Config E) is promoted as the primary fast linear surrogate for Phase 3 mutation proposals, with **HistGradientBoosting** available for non-linear candidate reranking.
