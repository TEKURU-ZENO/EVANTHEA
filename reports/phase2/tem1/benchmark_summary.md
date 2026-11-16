# EVANTHEA — Phase 2 Fitness Prediction Benchmark Summary

**Target Protein**: Beta-lactamase TEM-1 (`BLAT_ECOLX_Firnberg_2014`)  
**Evaluated Variants**: 4783  
**Evaluation Date**: 2026-11-16 17:12 UTC  
**Leakage Control Protocol**: Strict Zero-Leakage (Decoupled Sequence Representations & Strict Partitioning)

---

## 1. Model Comparison on In-Distribution Random Split (EXP-1)
| Model | Feature Config | Spearman (ρ) | Pearson (r) | MAE | RMSE | R² | Top-10% Enrichment |
|---|---|---|---|---|---|---|---|
| **MeanPredictor** | `mutation_features` | N/A | N/A | 0.3968 | 0.4363 | -0.0092 | 0.62x |
| **MedianPredictor** | `mutation_features` | N/A | N/A | 0.3872 | 0.4358 | -0.0068 | 0.62x |
| **ZeroShotESM2Predictor** | `mutation_features` | 0.3350 | 0.3503 | 1.9959 | 2.3098 | -27.2817 | 2.08x |
| **RidgeFitnessPredictor** | `mutation_features` | 0.4728 | 0.4608 | 0.3388 | 0.3886 | 0.1995 | 2.29x |
| **HistGradientBoostingPredictor** | `mutation_features` | 0.4834 | 0.4854 | 0.3286 | 0.3826 | 0.2239 | 2.49x |
| **RandomForestFitnessPredictor** | `mutation_features` | 0.4654 | 0.4702 | 0.3307 | 0.3858 | 0.2108 | 2.70x |

---

## 2. Generalization Boundaries (EXP-2 & EXP-3)
| Model | Split Regime | Scientific Hypothesis | Spearman (ρ) | MAE | RMSE | R² |
|---|---|---|---|---|---|---|
| **MeanPredictor** | `strict_position_held_out` | Spatial (Strict Position-Held-Out) | N/A | 0.3882 | 0.4320 | -0.0196 |
| **ZeroShotESM2Predictor** | `strict_position_held_out` | Spatial (Strict Position-Held-Out) | 0.3835 | 2.0022 | 2.3106 | -28.1728 |
| **RidgeFitnessPredictor** | `strict_position_held_out` | Spatial (Strict Position-Held-Out) | 0.4506 | 0.3317 | 0.3916 | 0.1621 |
| **HistGradientBoostingPredictor** | `strict_position_held_out` | Spatial (Strict Position-Held-Out) | 0.4242 | 0.3261 | 0.3998 | 0.1267 |
| *EXP-3* | `mutation_order` | *Omitted (Target is a saturation single-mutant landscape)* | N/A | N/A | N/A | N/A |

---

## 3. Feature Representation Ablation (EXP-4)
| Model | Feature Representation | Spearman (ρ) | Pearson (r) | MAE | RMSE | R² |
|---|---|---|---|---|---|---|
| **RidgeFitnessPredictor** | `seq_emb` | 0.3446 | 0.3270 | 0.3718 | 0.4152 | 0.0863 |
| **HistGradientBoostingPredictor** | `seq_emb` | 0.4887 | 0.4934 | 0.3298 | 0.3822 | 0.2255 |
| **RidgeFitnessPredictor** | `emb_diff` | 0.3446 | 0.3270 | 0.3718 | 0.4152 | 0.0863 |
| **HistGradientBoostingPredictor** | `emb_diff` | 0.4887 | 0.4934 | 0.3298 | 0.3822 | 0.2255 |
| **RidgeFitnessPredictor** | `emb_concat` | 0.3464 | 0.3306 | 0.3695 | 0.4138 | 0.0923 |
| **HistGradientBoostingPredictor** | `emb_concat` | 0.4887 | 0.4934 | 0.3298 | 0.3822 | 0.2255 |
| **RidgeFitnessPredictor** | `mutation_features` | 0.4728 | 0.4608 | 0.3388 | 0.3886 | 0.1995 |
| **HistGradientBoostingPredictor** | `mutation_features` | 0.4834 | 0.4854 | 0.3286 | 0.3826 | 0.2239 |
| **RidgeFitnessPredictor** | `hybrid` | 0.5037 | 0.4877 | 0.3264 | 0.3813 | 0.2291 |
| **HistGradientBoostingPredictor** | `hybrid` | 0.5043 | 0.5156 | 0.3234 | 0.3766 | 0.2480 |

---

## 4. Scientific Findings & Surrogate Selection for Phase 3
1. **Model Ranking Consistency**:
   - GBDT and Ridge surrogates on `hybrid` and `mutation_features` consistently outperform naive baselines while adhering strictly to zero-leakage constraints.
2. **Positional Generalization Drop**:
   - Spatial extrapolation on strict position-held-out splits exhibits expected performance degradation relative to random split interpolation, highlighting true regional structural variation.
3. **Surrogate Promotion**:
   - **HistGradientBoosting** on `hybrid` features is promoted as the primary candidate ranker, paired with **Ridge Regression** for rapid single-point mutation evaluations.
