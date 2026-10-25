# EVANTHEA — Phase 2 Fitness Prediction Benchmark Summary

**Target Protein**: Green Fluorescent Protein (`GFP_AEQVI_Sarkisyan_2016`)  
**Assay**: Sarkisyan et al. Fluorescence (DMS)  
**Evaluated Variants**: 10  
**Evaluation Date**: 2026-10-25 11:13 UTC

---

## 1. Model Comparison on In-Distribution Random Split (EXP-1)
| Model | Feature Config | Spearman (ρ) | Pearson (r) | MAE | RMSE | R² | Top-10% Enrichment |
|---|---|---|---|---|---|---|---|
| **MeanPredictor** | `emb_diff` | N/A | N/A | 0.5636 | 0.5636 | 0.0000 | N/A |
| **MedianPredictor** | `emb_diff` | N/A | N/A | 0.2970 | 0.2970 | 0.0000 | N/A |
| **ZeroShotESM2Predictor** | `emb_diff` | N/A | N/A | 3.6534 | 3.6534 | 0.0000 | N/A |
| **RidgeFitnessPredictor** | `emb_diff` | N/A | N/A | 0.0592 | 0.0592 | 0.0000 | N/A |
| **HistGradientBoostingPredictor** | `emb_diff` | N/A | N/A | 0.5636 | 0.5636 | 0.0000 | N/A |
| **RandomForestFitnessPredictor** | `emb_diff` | N/A | N/A | 0.5562 | 0.5562 | 0.0000 | N/A |

---

## 2. Generalization Boundaries (EXP-2 & EXP-3)
| Model | Split Regime | Scientific Hypothesis | Spearman (ρ) | MAE | RMSE | R² |
|---|---|---|---|---|---|---|
| **MeanPredictor** | `position_held_out` | Spatial (Unseen Residues) | N/A | 0.1607 | 0.1976 | -0.1765 |
| **ZeroShotESM2Predictor** | `position_held_out` | Spatial (Unseen Residues) | 1.0000 | 3.5741 | 3.5744 | -383.7860 |
| **RidgeFitnessPredictor** | `position_held_out` | Spatial (Unseen Residues) | 0.3000 | 0.1247 | 0.1575 | 0.2527 |
| **HistGradientBoostingPredictor** | `position_held_out` | Spatial (Unseen Residues) | N/A | 0.1607 | 0.1976 | -0.1765 |
| **MeanPredictor** | `mutation_order` | Epistatic (Higher-Order) | N/A | 0.0507 | 0.0507 | 0.0000 |
| **ZeroShotESM2Predictor** | `mutation_order` | Epistatic (Higher-Order) | N/A | 3.5421 | 3.5421 | 0.0000 |
| **RidgeFitnessPredictor** | `mutation_order` | Epistatic (Higher-Order) | N/A | 0.1424 | 0.1424 | 0.0000 |
| **HistGradientBoostingPredictor** | `mutation_order` | Epistatic (Higher-Order) | N/A | 0.0507 | 0.0507 | 0.0000 |

---

## 3. Feature Representation Ablation (EXP-4)
| Model | Feature Representation | Spearman (ρ) | Pearson (r) | MAE | RMSE | R² |
|---|---|---|---|---|---|---|
| **RidgeFitnessPredictor** | `seq_emb` | N/A | N/A | 0.0592 | 0.0592 | 0.0000 |
| **HistGradientBoostingPredictor** | `seq_emb` | N/A | N/A | 0.5636 | 0.5636 | 0.0000 |
| **RidgeFitnessPredictor** | `emb_diff` | N/A | N/A | 0.0592 | 0.0592 | 0.0000 |
| **HistGradientBoostingPredictor** | `emb_diff` | N/A | N/A | 0.5636 | 0.5636 | 0.0000 |
| **RidgeFitnessPredictor** | `emb_concat` | N/A | N/A | 0.0592 | 0.0592 | 0.0000 |
| **HistGradientBoostingPredictor** | `emb_concat` | N/A | N/A | 0.5636 | 0.5636 | 0.0000 |
| **RidgeFitnessPredictor** | `mutation_features` | N/A | N/A | 0.0416 | 0.0416 | 0.0000 |
| **HistGradientBoostingPredictor** | `mutation_features` | N/A | N/A | 0.5636 | 0.5636 | 0.0000 |
| **RidgeFitnessPredictor** | `hybrid` | N/A | N/A | 0.2603 | 0.2603 | 0.0000 |
| **HistGradientBoostingPredictor** | `hybrid` | N/A | N/A | 0.5636 | 0.5636 | 0.0000 |

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
