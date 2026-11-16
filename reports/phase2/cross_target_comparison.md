# EVANTHEA — Cross-Target DMS Benchmark Comparison

Comparative evaluation of EVANTHEA fitness predictors across two canonical experimental Deep Mutational Scanning (DMS) targets from ProteinGym under strict zero-leakage protocols.

---

## Target Profiles

| Property | Green Fluorescent Protein (GFP) | Beta-Lactamase TEM-1 (TEM-1) |
|---|---|---|
| **DMS Assay Reference** | Sarkisyan et al. (*Nature* 2016) | Firnberg et al. (*Mol. Biol. Evol.* 2014) |
| **Protein Accession** | UniProt `P42212` | UniProt `P62593` |
| **Organism** | *Aequorea victoria* | *Escherichia coli* |
| **Sequence Length** | 238 amino acids | 286 amino acids |
| **Valid Ingested Variants** | **51,027** | **4,783** |
| **Mutational Regime** | Combinatorial Multi-Mutant (1–15 substitutions) | Complete Saturation Single-Mutant Scan |
| **Phenotype Measured** | In vivo fluorescence intensity | Ampicillin minimum inhibitory concentration (MIC) |

---

## 1. In-Distribution Performance Comparison (Random 80/20 Split)

| Model | Feature Config | GFP Spearman (ρ) | GFP Pearson (r) | TEM-1 Spearman (ρ) | TEM-1 Pearson (r) |
|---|---|:---:|:---:|:---:|:---:|
| **ZeroShotBaseline** | BLOSUM62 Prior | 0.3478 | 0.3723 | 0.3350 | 0.3503 |
| **RidgeFitnessPredictor** | `mutation_features` | 0.6183 | 0.6263 | 0.4728 | 0.4608 |
| **HistGradientBoosting** | `mutation_features` | 0.6201 | 0.6595 | 0.4834 | 0.4854 |
| **RandomForest** | `mutation_features` | 0.6126 | 0.6520 | 0.4654 | 0.4702 |
| **RidgeFitnessPredictor** | `hybrid` ($\Delta \mathbf{h}$ + PhysChem) | 0.6409 | 0.6471 | 0.5037 | 0.4877 |
| **HistGradientBoosting** | `hybrid` ($\Delta \mathbf{h}$ + PhysChem) | **0.6554** | **0.7006** | **0.5043** | **0.5156** |

---

## 2. Generalization Under Strict Spatial Partitioning (EXP-2)
*Strict Position-Held-Out Split: Test variants have zero mutations at training positions.*

| Model | GFP Spearman (ρ) | GFP MAE | TEM-1 Spearman (ρ) | TEM-1 MAE |
|---|:---:|:---:|:---:|:---:|
| **ZeroShotBaseline** | 0.1530 | 3.4497 | 0.3835 | 2.0022 |
| **RidgeFitnessPredictor** | **0.3330** | **0.4447** | **0.4506** | **0.3317** |
| **HistGradientBoosting** | 0.2848 | 0.5507 | 0.4242 | 0.3261 |

---

## 3. Scientific Insights Across Structural Contexts

1. **Saturation Single-Mutant vs Combinatorial Multi-Mutant Dynamics**:
   - In TEM-1 (100% single substitutions), linear models (`Ridge`, $\rho = 0.4728$) closely track non-linear models (`HistGBDT`, $\rho = 0.4834$), reflecting additive local perturbations across the catalytic scaffold.
   - In GFP (multi-mutants with higher-order epistasis), `HistGBDT` on `hybrid` features achieves substantial gains over pure linear models ($\rho = 0.6554$ vs $0.6183$), successfully modeling non-linear epistatic interactions.
2. **Spatial Transferability**:
   - In strict position-held-out tests, TEM-1 retains higher spatial ranking fidelity ($\rho = 0.4506$) than GFP ($\rho = 0.3330$). This is biologically consistent with TEM-1's globular $\alpha/\beta$ domain having localized active-site sensitivity, whereas GFP's 11-stranded $\beta$-barrel requires cooperative tertiary packing throughout the cylindrical shell.
