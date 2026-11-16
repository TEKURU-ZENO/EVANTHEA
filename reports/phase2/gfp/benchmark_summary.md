# EVANTHEA — Phase 2 Fitness Prediction Benchmark Summary

**Target Protein**: Green Fluorescent Protein (`GFP_AEQVI_Sarkisyan_2016`)  
**Evaluated Variants**: 51027  
**Evaluation Date**: 2026-11-16 17:14 UTC  
**Leakage Control Protocol**: Strict Zero-Leakage (Decoupled Sequence Representations & Strict Partitioning)

---

## 1. Model Comparison on In-Distribution Random Split (EXP-1)
| Model | Feature Config | Spearman (ρ) | Pearson (r) | MAE | RMSE | R² | Top-10% Enrichment |
|---|---|---|---|---|---|---|---|
| **MeanPredictor** | `mutation_features` | N/A | N/A | 1.0181 | 1.0532 | -0.0000 | 1.13x |
| **MedianPredictor** | `mutation_features` | N/A | N/A | 0.9306 | 1.2305 | -0.3651 | 1.13x |
| **ZeroShotESM2Predictor** | `mutation_features` | 0.3478 | 0.3723 | 4.0937 | 4.8959 | -20.6114 | 1.47x |
| **RidgeFitnessPredictor** | `mutation_features` | 0.6183 | 0.6263 | 0.6760 | 0.8211 | 0.3921 | 2.80x |
| **HistGradientBoostingPredictor** | `mutation_features` | 0.6201 | 0.6595 | 0.5980 | 0.7918 | 0.4348 | 2.54x |
| **RandomForestFitnessPredictor** | `mutation_features` | 0.6126 | 0.6520 | 0.6069 | 0.7986 | 0.4249 | 2.58x |

---

## 2. Generalization Boundaries (EXP-2 & EXP-3)
| Model | Split Regime | Scientific Hypothesis | Spearman (ρ) | MAE | RMSE | R² |
|---|---|---|---|---|---|---|
| **MeanPredictor** | `strict_position_held_out` | Spatial (Strict Position-Held-Out) | N/A | 0.7398 | 0.8171 | -0.1994 |
| **ZeroShotESM2Predictor** | `strict_position_held_out` | Spatial (Strict Position-Held-Out) | 0.1530 | 3.4497 | 3.9589 | -27.1537 |
| **RidgeFitnessPredictor** | `strict_position_held_out` | Spatial (Strict Position-Held-Out) | 0.3330 | 0.4447 | 0.6924 | 0.1388 |
| **HistGradientBoostingPredictor** | `strict_position_held_out` | Spatial (Strict Position-Held-Out) | 0.2848 | 0.5507 | 0.7092 | 0.0965 |
| **MeanPredictor** | `mutation_order` | Epistatic (Higher-Order) | N/A | 1.1431 | 1.4656 | -0.9208 |
| **ZeroShotESM2Predictor** | `mutation_order` | Epistatic (Higher-Order) | 0.3321 | 4.1890 | 5.0819 | -22.0957 |
| **RidgeFitnessPredictor** | `mutation_order` | Epistatic (Higher-Order) | 0.3508 | 1.1125 | 1.4053 | -0.7661 |
| **HistGradientBoostingPredictor** | `mutation_order` | Epistatic (Higher-Order) | 0.2674 | 1.1181 | 1.4307 | -0.8304 |

---

## 3. Feature Representation Ablation (EXP-4)
| Model | Feature Representation | Spearman (ρ) | Pearson (r) | MAE | RMSE | R² |
|---|---|---|---|---|---|---|
| **RidgeFitnessPredictor** | `seq_emb` | 0.4358 | 0.4759 | 0.8079 | 0.9263 | 0.2263 |
| **HistGradientBoostingPredictor** | `seq_emb` | 0.6149 | 0.6554 | 0.6180 | 0.7955 | 0.4294 |
| **RidgeFitnessPredictor** | `emb_diff` | 0.4358 | 0.4758 | 0.8080 | 0.9263 | 0.2263 |
| **HistGradientBoostingPredictor** | `emb_diff` | 0.6149 | 0.6554 | 0.6180 | 0.7955 | 0.4294 |
| **RidgeFitnessPredictor** | `emb_concat` | 0.4358 | 0.4758 | 0.8080 | 0.9264 | 0.2263 |
| **HistGradientBoostingPredictor** | `emb_concat` | 0.6149 | 0.6554 | 0.6180 | 0.7955 | 0.4294 |
| **RidgeFitnessPredictor** | `mutation_features` | 0.6183 | 0.6263 | 0.6760 | 0.8211 | 0.3921 |
| **HistGradientBoostingPredictor** | `mutation_features` | 0.6201 | 0.6595 | 0.5980 | 0.7918 | 0.4348 |
| **RidgeFitnessPredictor** | `hybrid` | 0.6409 | 0.6471 | 0.6563 | 0.8031 | 0.4185 |
| **HistGradientBoostingPredictor** | `hybrid` | 0.6554 | 0.7006 | 0.5529 | 0.7516 | 0.4907 |

---

## 4. Scientific Findings & Surrogate Selection for Phase 3
1. **Model Ranking Consistency**:
   - GBDT and Ridge surrogates on `hybrid` and `mutation_features` consistently outperform naive baselines while adhering strictly to zero-leakage constraints.
2. **Positional Generalization Drop**:
   - Spatial extrapolation on strict position-held-out splits exhibits expected performance degradation relative to random split interpolation, highlighting true regional structural variation.
3. **Surrogate Promotion**:
   - **HistGradientBoosting** on `hybrid` features is promoted as the primary candidate ranker, paired with **Ridge Regression** for rapid single-point mutation evaluations.
