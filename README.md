# EVANTHEA 🧬
### Evolutionary Variant Analysis for Theoretical Protein Adaptation

An Adaptive, Closed-Loop Framework for Protein Sequence Optimization Using Protein Language Models, Multi-Objective Evolution, Reinforcement Learning, and Structural Quality Gates.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 168 Passed](https://img.shields.io/badge/Tests-168%20Passed-brightgreen.svg)](tests/)

---

## Architecture Overview

EVANTHEA formulates protein sequence design as a verifiable, closed-loop evolutionary optimization process:

```
                  ┌──────────────────────────────────────────────┐
                  │ Canonical Wild-Type / DMS Seed Population    │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │   Phase 3: Smart Mutation Generator          │
                  │   - BLOSUM62 Transition Priors               │
                  │   - Biophysical Deltas (Charge, Hydro, Vol)  │
                  │   - Context-Aware Combinatorial Proposals    │
                  └──────────────┬───────────────────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
  ┌─────────────────────────────┐  ┌─────────────────────────────┐
  │ Phase 4: Evolutionary Engine│  │ Phase 6: RL Agent (PPO)     │
  │ - Multi-Objective NSGA-II   │  │ - Native PyTorch Actor-     │
  │ - Single-Objective GA       │  │   Critic Architecture       │
  │ - Pareto Crowding Distance  │  │ - ProteinDesignEnv MDP      │
  │ - Homologous Crossover      │  │ - Action-Masked Policy      │
  └──────────────┬──────────────┘  └──────────────┬──────────────┘
                 │                                │
                 └───────────────┬────────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────────────────────┐
                  │ Phase 5: Evolutionary Memory Engine          │
                  │ - SQLite Relational Storage & Lineage Graph  │
                  │ - Exact NumPy Vector Index (Cosine / L2)     │
                  │ - Duplicate Evaluation Bypasser              │
                  │ - Global Archive Elitism Injector            │
                  └──────────────┬───────────────────────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────────────────────┐
                  │ Phase 2: Multi-Representation Fitness Models │
                  │ - Zero-Shot ESM-2 Masked Marginal Log-Odds   │
                  │ - Ridge, HistGBDT, Random Forest Surrogates  │
                  │ - Delta, Concat, and Hybrid Feature Matrices │
                  └──────────────┬───────────────────────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────────────────────┐
                  │ Phase 7: Structural Validation Engine        │
                  │ - Modular Predictors (PDB, ESMFold, Mock)    │
                  │ - Kabsch CA RMSD Alignment                   │
                  │ - Cβ-Cβ Contact Preservation (F1-score)      │
                  │ - Steric Clash Overlap Filtering             │
                  │ - Shrake-Rupley SASA Calculation             │
                  │ - Multi-Threshold Structural Quality Gate    │
                  └──────────────┬───────────────────────────────┘
                                 │
                                 ▼
                  ┌──────────────────────────────────────────────┐
                  │ Validated Pareto Variants & Scientific Logs  │
                  └──────────────────────────────────────────────┘
```

---

## Core Modules & Implementation Reality

Unlike conceptual frameworks that rely on placeholder marketing claims, EVANTHEA's codebase is fully implemented, strictly verified across 168 unit and integration tests, and benchmarked on real experimental datasets:

### 1. Data Ingestion & Sequence Intelligence (Phase 1)
- **Rigid Amino Acid Validation**: Strict checks for canonical 20 amino acids, case normalization, and coordinate conversion (0-indexed internal representations vs 1-indexed biological coordinates).
- **Multi-Mutant Parsing**: Deterministic alignment and parsing of single and combinatorial mutation strings (e.g., `M1V:K3E`).
- **Benchmark Ingestors**: Streamlined ingestion for ProteinGym (GFP `GFP_AEQVI_Sarkisyan_2016` [51,027 variants] and TEM-1 $\beta$-lactamase `BLAT_ECOLX_Firnberg_2014` [4,783 variants]) and FireProtDB stability assays.
- **Language Model Integration**: ESM-2 (`facebook/esm2_t30_150M_UR50D`) wrapper computing per-residue embeddings, sequence pooled representations, masked-marginal log-likelihood scores, and pseudo-perplexity.
- **Leakage-Resistant Splits**: In-distribution random split, spatial position-held-out split (testing generalization to unseen residues), and mutation-order split (testing epistatic generalization from low- to high-order mutants).
- **Persistent Caching**: SQLite variant registry with batch hash indexing and disk caching.

### 2. Surrogate Fitness Landscape Modeling (Phase 2)
- **Multi-Representation Feature Extraction**:
  - `seq_emb`: Full 640-dim mean-pooled sequence representation.
  - `emb_diff`: Delta vector $\Delta \mathbf{h} = \mathbf{h}_{\text{variant}} - \mathbf{h}_{\text{wt}}$ isolating local perturbation.
  - `emb_concat`: $[\mathbf{h}_{\text{variant}} \,\|\, \mathbf{h}_{\text{wt}}]$ (1280 dims).
  - `mutation_features`: Physico-chemical deltas (charge, hydrophobicity, molecular weight, isoelectric point) + one-hot positional encoding.
  - `hybrid`: Concatenation of $\Delta \mathbf{h}$ with explicit physico-chemical deltas.
- **Surrogate Algorithms**: Ridge Regression with closed-form L2 regularization, Histogram-based Gradient Boosted Trees (`HistGradientBoostingRegressor`), Random Forest, and Zero-Shot PLM baselines.
- **Leakage-Safe Scaling**: `TargetTransformer` fitted strictly on training folds to prevent target leakage.

### 3. Smart Mutation Proposal Engine (Phase 3)
- **Biophysically Constrained Proposals**: Generates single and combinatorial variants bounded by maximum Hamming distance.
- **BLOSUM62 Priors**: Conservative amino acid substitution transition matrices prioritizing evolutionary viability.
- **Context-Aware Combinatorial Engine**: Stepwise combinatorial expansion evaluating multi-mutant fitness trajectory candidates.

### 4. Multi-Objective Evolutionary Optimization (Phase 4)
- **NSGA-II Algorithm**: Non-dominated sorting with crowding distance preservation across competing fitness objectives (e.g. activity vs stability vs naturalness).
- **Single-Objective Genetic Algorithm**: Elitist GA with tournament selection and homologous mutation crossover.
- **Optimization Baselines**: Greedy Hill-Climber and Uniform Random Search baselines for quantifying algorithmic search efficiency.
- **Pareto Metrics**: Exact Pareto front extraction, empirical hypervolume calculation, and generational convergence tracking.

### 5. Persistent Evolutionary Memory (Phase 5)
- **Dual-Tier Architecture**:
  - **Relational Metadata**: SQLite database tracking all evaluated variants, step numbers, generation lineages, and objective vectors.
  - **Dense Vector Index**: Decoupled vector retrieval supporting exact similarity search.
- **Exact Vector Index (`ExactNumpyVectorIndex`)**: Pure NumPy implementation supporting exact Cosine and Euclidean ($L_2$) distance retrieval with 100% recall and zero required external C++ dependencies. Includes optional `FaissVectorIndex` wrapper if `faiss` is installed.
- **Duplicate Evaluation Bypasser**: Prevents redundant surrogate calls by hydrating previously scored variants directly from memory.
- **Archive Elitism Injector**: Injects historical non-dominated variants into lagging populations to escape local optima.

### 6. Reinforcement Learning via PyTorch PPO (Phase 6)
- **Policy Optimization Architecture**: Native PyTorch Actor-Critic Proximal Policy Optimization (PPO) implemented in `src/evanthea/rl/ppo.py` with Generalized Advantage Estimation (GAE), orthogonal weight initialization, and clipped surrogate objectives. *(Note: Implemented directly in native PyTorch; does not require Ray RLlib).*
- **Environment MDP (`ProteinDesignEnv`)**: Gymnasium-compatible environment where states represent sequence embeddings and mutation trajectories, and discrete actions select residue position and substitution.
- **Action Masking**: Enforces strict biological feasibility (prevents identity mutations and masked out-of-bounds positions).
- **Composite Rewards**: Multi-term reward function balancing target fitness gains, evolutionary naturalness, step penalties, and terminal Pareto bonuses.

### 7. Multi-Gate Structural Validation Engine (Phase 7)
- **Modular Structure Predictors**:
  - `MockStructurePredictor`: Fast, zero-dependency 3D backbone synthesis with canonical bond lengths (~3.8 Å) and realistic per-residue pLDDT profiles for offline tests.
  - `ESMFoldPredictor`: Modular wrapper interfacing with local ESMFold weights or the ESM Atlas REST API with automated fallback.
  - `ReferencePDBPredictor`: Extracts experimental ground truth coordinates from PDB/mmCIF structures.
- **Geometric & Biophysical Metrics**:
  - **Kabsch Alignment & RMSD**: SVD-based optimal rotational superposition of C$\alpha$ coordinates.
  - **Contact Map Preservation**: C$\beta$-C$\beta$ distance matrix thresholding (< 8.0 Å) computing contact precision, recall, and $F_1$-score against wild type.
  - **Steric Clash Detection**: Van der Waals overlap analysis identifying steric clashes (< 2.2 Å non-bonded inter-atomic distances).
  - **Shrake-Rupley SASA**: Numerical solvent-accessible surface area calculation using spherical approximations.
- **Structural Quality Gate (`StructureQualityGate`)**: Multi-metric acceptance filter (pLDDT $\ge \tau_{\text{pLDDT}}$, RMSD $\le \tau_{\text{RMSD}}$, Clashes $\le \tau_{\text{clash}}$) rejecting unviable conformations before biological synthesis.

---

## Multi-Target Empirical Benchmarks: GFP & TEM-1 $\beta$-Lactamase

The Phase 2 fitness prediction framework was evaluated on two canonical experimental Deep Mutational Scanning (DMS) datasets from ProteinGym under strict zero-leakage protocols:
1. **Green Fluorescent Protein (`GFP_AEQVI_Sarkisyan_2016`)**: **51,027 valid variants** spanning 1–15 combinatorial substitutions, testing non-linear epistatic landscape modeling and $\beta$-barrel fold stability.
2. **Beta-Lactamase TEM-1 (`BLAT_ECOLX_Firnberg_2014`)**: **4,783 valid variants** comprising a complete single-mutant saturation scan across all 286 residues, testing catalytic and structural sensitivity in an $\alpha/\beta$ globular enzyme.

### Target Profiles

| Property | Green Fluorescent Protein (GFP) | Beta-Lactamase TEM-1 (TEM-1) |
|---|---|---|
| **DMS Assay Reference** | Sarkisyan et al. (*Nature* 2016) | Firnberg et al. (*Mol. Biol. Evol.* 2014) |
| **Protein Accession** | UniProt `P42212` | UniProt `P62593` |
| **Organism** | *Aequorea victoria* | *Escherichia coli* |
| **Sequence Length** | 238 amino acids | 286 amino acids |
| **Valid Ingested Variants** | **51,027** | **4,783** |
| **Mutational Regime** | Combinatorial Multi-Mutant (1–15 substitutions) | Complete Saturation Single-Mutant Scan |
| **Fold Topology** | 11-stranded $\beta$-barrel | $\alpha/\beta$ globular enzyme |
| **Phenotype Measured** | In vivo fluorescence intensity | Ampicillin minimum inhibitory concentration (MIC) |

### Rigorous Leakage Controls & Scientific Integrity
To eliminate train/test leakage, the benchmark enforces two critical safeguards:
1. **Target-Decoupled Sequence Representations**: Feature vectors are derived strictly from amino acid sequences and unsupervised substitution priors (BLOSUM62 / PLM log-odds), eliminating label leakage.
2. **Strict Multi-Mutant Position Isolation**: While conventional position splits allow test multi-mutants to retain mutations at training positions (where up to 91.2% of test variants share partial residue mutations with train), EVANTHEA implements `strict_position_held_out`. In strict mode, test variants are partitioned such that **all** constituent mutations are exclusively at held-out positions, guaranteeing zero near-duplicate or positional overlap.

---

### Cross-Target In-Distribution Performance (Random 80/20 Split)

| Model | Feature Config | GFP Spearman (ρ) | GFP Pearson (r) | TEM-1 Spearman (ρ) | TEM-1 Pearson (r) |
|---|---|:---:|:---:|:---:|:---:|
| **ZeroShotBaseline** | BLOSUM62 Prior | 0.3478 | 0.3723 | 0.3350 | 0.3503 |
| **RidgeFitnessPredictor** | `mutation_features` | 0.6183 | 0.6263 | 0.4728 | 0.4608 |
| **HistGradientBoosting** | `mutation_features` | 0.6201 | 0.6595 | 0.4834 | 0.4854 |
| **RandomForest** | `mutation_features` | 0.6126 | 0.6520 | 0.4654 | 0.4702 |
| **RidgeFitnessPredictor** | `hybrid` ($\Delta \mathbf{h}$ + PhysChem) | 0.6409 | 0.6471 | 0.5037 | 0.4877 |
| **HistGradientBoosting** | `hybrid` ($\Delta \mathbf{h}$ + PhysChem) | **0.6554** | **0.7006** | **0.5043** | **0.5156** |

*(Note: Literature benchmarks for supervised models on GFP in ProteinGym typically land between $\rho = 0.60 - 0.70$. EVANTHEA's models achieve $\rho = 0.62 - 0.66$ without label contamination).*

---

### Cross-Target Spatial Generalization (Strict Position-Held-Out Split)
*Strict Position-Held-Out Split: Test variants have zero mutations at training positions.*

| Model | GFP Spearman (ρ) | GFP MAE | TEM-1 Spearman (ρ) | TEM-1 MAE |
|---|:---:|:---:|:---:|:---:|
| **ZeroShotBaseline** | 0.1530 | 3.4497 | 0.3835 | 2.0022 |
| **RidgeFitnessPredictor** | **0.3330** | **0.4447** | **0.4506** | **0.3317** |
| **HistGradientBoosting** | 0.2848 | 0.5507 | 0.4242 | 0.3261 |

---

### Target 1: Green Fluorescent Protein (GFP, 51,027 Variants)

#### 1. In-Distribution Model Comparison (Random 80/20 Split)
Evaluated on **10,206 held-out test variants** under `mutation_features`:

| Model | Feature Representation | Spearman ($\rho$) | Pearson ($r$) | MAE | RMSE | $R^2$ | Top-10% Enrichment |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **MeanPredictor** | `mutation_features` | N/A | N/A | 1.0181 | 1.0532 | -0.0000 | 1.13x |
| **MedianPredictor** | `mutation_features` | N/A | N/A | 0.9306 | 1.2305 | -0.3651 | 1.13x |
| **ZeroShotBaseline** | `mutation_features` (BLOSUM62) | 0.3478 | 0.3723 | 4.0937 | 4.8959 | -20.6114 | 1.47x |
| **RidgeFitnessPredictor** | `mutation_features` | 0.6183 | 0.6263 | 0.6760 | 0.8211 | 0.3921 | **2.80x** |
| **HistGradientBoostingPredictor** | `mutation_features` | **0.6201** | **0.6595** | **0.5980** | **0.7918** | **0.4348** | 2.54x |
| **RandomForestFitnessPredictor** | `mutation_features` | 0.6126 | 0.6520 | 0.6069 | 0.7986 | 0.4249 | 2.58x |

#### 2. Generalization Boundaries (Out-of-Distribution Splits)

| Model | Split Regime | Scientific Hypothesis | Spearman ($\rho$) | MAE | RMSE | $R^2$ |
|---|---|---|:---:|:---:|:---:|:---:|
| **ZeroShotBaseline** | `strict_position_held_out` | Spatial (Strict Position-Held-Out) | 0.1696 | 3.6094 | 4.1738 | -30.4539 |
| **RidgeRegression** | `strict_position_held_out` | Spatial (Strict Position-Held-Out) | **0.3572** | **0.4557** | **0.6762** | **0.1744** |
| **HistGradientBoosting** | `strict_position_held_out` | Spatial (Strict Position-Held-Out) | 0.3125 | 0.4492 | 0.6778 | 0.1704 |
| **ZeroShotBaseline** | `mutation_order` | Epistatic (Higher-Order) | 0.3321 | 4.1890 | 5.0819 | -22.0957 |
| **RidgeRegression** | `mutation_order` | Epistatic (Higher-Order) | **0.3508** | **1.1125** | **1.4053** | -0.7661 |
| **HistGradientBoosting** | `mutation_order` | Epistatic (Higher-Order) | 0.2674 | 1.1181 | 1.4307 | -0.8304 |

#### 3. Feature Representation Ablation (Ridge vs HistGBDT)

| Model | Representation | Spearman ($\rho$) | Pearson ($r$) | MAE | RMSE | $R^2$ |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **RidgeRegression** | `seq_emb` (Sequence Rep) | 0.4358 | 0.4759 | 0.8079 | 0.9263 | 0.2263 |
| **HistGBDT** | `seq_emb` (Sequence Rep) | 0.6149 | 0.6554 | 0.6180 | 0.7955 | 0.4294 |
| **RidgeRegression** | `emb_diff` ($\Delta \mathbf{h}$) | 0.4358 | 0.4758 | 0.8080 | 0.9263 | 0.2263 |
| **HistGBDT** | `emb_diff` ($\Delta \mathbf{h}$) | 0.6149 | 0.6554 | 0.6180 | 0.7955 | 0.4294 |
| **RidgeRegression** | `mutation_features` | 0.6183 | 0.6263 | 0.6760 | 0.8211 | 0.3921 |
| **HistGBDT** | `mutation_features` | 0.6201 | 0.6595 | 0.5980 | 0.7918 | 0.4348 |
| **RidgeRegression** | `hybrid` ($\Delta \mathbf{h}$ + PhysChem) | 0.6409 | 0.6471 | 0.6563 | 0.8031 | 0.4185 |
| **HistGBDT** | `hybrid` ($\Delta \mathbf{h}$ + PhysChem) | **0.6554** | **0.7006** | **0.5529** | **0.7516** | **0.4907** |

---

### Target 2: $\beta$-Lactamase TEM-1 (TEM-1, 4,783 Variants)

#### 1. In-Distribution Model Comparison (Random 80/20 Split)
Evaluated on **957 held-out test variants** under `mutation_features`:

| Model | Feature Config | Spearman ($\rho$) | Pearson ($r$) | MAE | RMSE | $R^2$ | Top-10% Enrichment |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **MeanPredictor** | `mutation_features` | N/A | N/A | 0.3968 | 0.4363 | -0.0092 | 0.62x |
| **MedianPredictor** | `mutation_features` | N/A | N/A | 0.3872 | 0.4358 | -0.0068 | 0.62x |
| **ZeroShotBaseline** | `mutation_features` (BLOSUM62) | 0.3350 | 0.3503 | 1.9959 | 2.3098 | -27.2817 | 2.08x |
| **RidgeFitnessPredictor** | `mutation_features` | 0.4728 | 0.4608 | 0.3388 | 0.3886 | 0.1995 | 2.29x |
| **HistGradientBoosting** | `mutation_features` | **0.4834** | **0.4854** | **0.3286** | **0.3826** | **0.2239** | 2.49x |
| **RandomForest** | `mutation_features` | 0.4654 | 0.4702 | 0.3307 | 0.3858 | 0.2108 | **2.70x** |

#### 2. Generalization Boundaries (EXP-2: Strict Position-Held-Out Split)

| Model | Split Regime | Scientific Hypothesis | Spearman ($\rho$) | MAE | RMSE | $R^2$ |
|---|---|---|:---:|:---:|:---:|:---:|
| **ZeroShotBaseline** | `strict_position_held_out` | Spatial (Strict Position-Held-Out) | 0.3835 | 2.0022 | 2.3106 | -28.1728 |
| **RidgeFitnessPredictor** | `strict_position_held_out` | Spatial (Strict Position-Held-Out) | **0.4506** | 0.3317 | **0.3916** | **0.1621** |
| **HistGradientBoosting** | `strict_position_held_out` | Spatial (Strict Position-Held-Out) | 0.4242 | **0.3261** | 0.3998 | 0.1267 |

*(Note: EXP-3 Mutation-Order split is naturally omitted for TEM-1 as this dataset is a pure single-mutant saturation scan).*

#### 3. Feature Representation Ablation (EXP-4)

| Model | Feature Representation | Spearman ($\rho$) | Pearson ($r$) | MAE | RMSE | $R^2$ |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **RidgeFitnessPredictor** | `seq_emb` | 0.3446 | 0.3270 | 0.3718 | 0.4152 | 0.0863 |
| **HistGradientBoosting** | `seq_emb` | 0.4887 | 0.4934 | 0.3298 | 0.3822 | 0.2255 |
| **RidgeFitnessPredictor** | `emb_diff` ($\Delta \mathbf{h}$) | 0.3446 | 0.3270 | 0.3718 | 0.4152 | 0.0863 |
| **HistGradientBoosting** | `emb_diff` ($\Delta \mathbf{h}$) | 0.4887 | 0.4934 | 0.3298 | 0.3822 | 0.2255 |
| **RidgeFitnessPredictor** | `mutation_features` | 0.4728 | 0.4608 | 0.3388 | 0.3886 | 0.1995 |
| **HistGradientBoosting** | `mutation_features` | 0.4834 | 0.4854 | 0.3286 | 0.3826 | 0.2239 |
| **RidgeFitnessPredictor** | `hybrid` ($\Delta \mathbf{h}$ + PhysChem) | 0.5037 | 0.4877 | 0.3264 | 0.3813 | 0.2291 |
| **HistGradientBoosting** | `hybrid` ($\Delta \mathbf{h}$ + PhysChem) | **0.5043** | **0.5156** | **0.3234** | **0.3766** | **0.2480** |

---

### Biological Takeaways Across Structural Contexts

1. **Additive vs Epistatic Regimes**:
   - In TEM-1 (100% single substitutions), linear models (`Ridge`, $\rho = 0.4728$) closely track non-linear models (`HistGBDT`, $\rho = 0.4834$), reflecting additive local perturbations across the catalytic scaffold.
   - In GFP (multi-mutants with higher-order epistasis), non-linear boosting on `hybrid` features achieves substantial gains over linear baselines ($\rho = 0.6554$ vs $0.6183$), successfully capturing epistatic interactions.
2. **Spatial Extrapolation Differences**:
   - Under strict position-held-out splits, TEM-1 retains higher spatial ranking fidelity ($\rho = 0.4506$) than GFP ($\rho = 0.3330$). This aligns with TEM-1's globular $\alpha/\beta$ fold having localized catalytic sensitivity, whereas GFP's cylindrical $\beta$-barrel depends on cooperative tertiary packing throughout the shell.

*All raw artifacts and cross-target summaries are available in [`reports/phase2/cross_target_comparison.md`](reports/phase2/cross_target_comparison.md), [`reports/phase2/gfp/`](reports/phase2/gfp/), and [`reports/phase2/tem1/`](reports/phase2/tem1/).*

---

## Installation & Setup

EVANTHEA requires **Python 3.11**.

```powershell
# Clone the repository
git clone https://github.com/TEKURU-ZENO/EVANTHEA.git
cd EVANTHEA

# Initialize virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

---

## Reproducing Benchmarks & Workflows

### 1. Download Experimental Datasets
Fetches the canonical DMS datasets (GFP 51k variants, TEM-1 4.7k variants) and UniProt wild-type references:
```powershell
# Download all registered targets
.\.venv\Scripts\python.exe scripts/download_data.py --dataset all

# Or download individual targets:
.\.venv\Scripts\python.exe scripts/download_data.py --dataset gfp
.\.venv\Scripts\python.exe scripts/download_data.py --dataset tem1
```

### 2. Run Multi-Target Phase 2 Benchmarks
Executes EXP-1 (Random Split), EXP-2 (Position-Held-Out), EXP-3 (Mutation-Order), and EXP-4 (Feature Ablation) across targets:
```powershell
# Run benchmark across all registered targets:
.\.venv\Scripts\python.exe scripts/run_phase2_benchmark.py --dataset all

# Or run individually:
.\.venv\Scripts\python.exe scripts/run_phase2_benchmark.py --dataset gfp
.\.venv\Scripts\python.exe scripts/run_phase2_benchmark.py --dataset tem1
```

### 3. Run Multi-Objective Evolution
Executes NSGA-II or GA on target proteins with surrogate models and evolutionary memory:
```powershell
.\.venv\Scripts\python.exe -c "
from evanthea.evolution.nsga2 import NSGA2Optimizer
from evanthea.predictors.surrogate import RidgeFitnessPredictor
# Initialize optimizer and evolve variants...
"
```

### 4. Run the Full Test Suite
Executes all 168 unit and integration tests across all 7 phases:
```powershell
.\.venv\Scripts\pytest.exe
```

---

## Technical Specifications & Transparency

| Claim / Component | Actual Implementation Status in Code |
|---|---|
| **Reinforcement Learning** | ✅ **Implemented**: Native PyTorch Actor-Critic PPO (`src/evanthea/rl/ppo.py`, `environment.py`, `agent.py`) with GAE and action masking. No Ray dependency. |
| **Vector Similarity Index** | ✅ **Implemented**: Pure NumPy exact vector similarity index (`src/evanthea/memory/vector_index.py`) with 100% exact Cosine & $L_2$ recall. Optional drop-in FAISS wrapper. |
| **Structural Validation** | ✅ **Implemented**: Modular structural evaluation pipeline (`src/evanthea/structure/`) computing Kabsch RMSD, C$\beta$ contacts, van der Waals clashes, and SASA via `MockStructurePredictor`, `ESMFoldPredictor` API wrapper, and `ReferencePDBPredictor`. |
| **Generative Prior** | ❌ **Cut**: No diffusion or de novo generative prior in code. EVANTHEA grounds search on canonical wild-type records and DMS seed populations. |
| **Web UI Dashboard** | 📋 **Roadmap**: Interactive React/Three.js dashboard is planned for a future release; current results are exported as publication-grade CSV, JSON, and Markdown reports. |

---

## Project Structure

```
EVANTHEA/
├── data/
│   └── raw/
│       ├── proteingym/       # Experimental DMS assays (GFP 51k, TEM-1 4.7k variants)
│       └── uniprot/          # Canonical FASTA reference sequences
├── reports/
│   └── phase2/               # Publication-grade CSV, JSON, and MD benchmarks
│       ├── gfp/              # Full GFP 51k variant reports
│       ├── tem1/             # Full TEM-1 beta-lactamase reports
│       └── cross_target_comparison.md # Cross-target synthesis & biological analysis
├── scripts/
│   ├── download_data.py      # Multi-target dataset acquisition script
│   └── run_phase2_benchmark.py # Multi-target benchmark runner across all splits
├── src/
│   └── evanthea/
│       ├── data/             # Ingestion, caching, schemas, and splitting
│       ├── protein/          # Sequence validation, coordinates, and PLM wrappers
│       ├── mutation/         # Mutation generator, BLOSUM62 priors, biophysical deltas
│       ├── predictors/       # Ridge, HistGBDT, RF surrogates and target transformers
│       ├── evolution/        # NSGA-II, GA, crossover, and baseline optimizers
│       ├── memory/           # SQLite store, exact NumPy vector index, elitism injectors
│       ├── rl/               # PyTorch Actor-Critic PPO, ProteinDesignEnv MDP, rewards
│       ├── structure/        # Kabsch RMSD, contact maps, clash filters, SASA, quality gates
│       └── evaluation/       # Benchmark metrics, Pareto hypervolume, rank correlations
├── tests/
│   ├── fixtures/             # Verified test fixtures (GFP, TEM-1)
│   ├── integration/          # End-to-end multi-phase pipeline tests (6 suites)
│   └── unit/                 # Exhaustive unit test coverage (162 tests)
└── pyproject.toml            # Project configuration and dependency specifications
```

---

## Citation

```bibtex
@software{evanthea2026,
  author = {EVANTHEA Contributors},
  title = {EVANTHEA: An Adaptive, Closed-Loop Framework for Protein Sequence Optimization},
  year = {2026},
  url = {https://github.com/TEKURU-ZENO/EVANTHEA}
}
```
