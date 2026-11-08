# EVANTHEA 🧬
### Evolutionary Variant Analysis for Theoretical Protein Adaptation

An Adaptive, Closed-Loop Framework for Protein Sequence Optimization Using Protein Language Models, Multi-Objective Evolution, Reinforcement Learning, and Structural Quality Gates.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 167 Passed](https://img.shields.io/badge/Tests-167%20Passed-brightgreen.svg)](tests/)

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

Unlike conceptual frameworks that rely on placeholder marketing claims, EVANTHEA's codebase is fully implemented, strictly verified across 167 unit and integration tests, and benchmarked on real experimental datasets:

### 1. Data Ingestion & Sequence Intelligence (Phase 1)
- **Rigid Amino Acid Validation**: Strict checks for canonical 20 amino acids, case normalization, and coordinate conversion (0-indexed internal representations vs 1-indexed biological coordinates).
- **Multi-Mutant Parsing**: Deterministic alignment and parsing of single and combinatorial mutation strings (e.g., `M1V:K3E`).
- **Benchmark Ingestors**: Streamlined ingestion for ProteinGym (`GFP_AEQVI_Sarkisyan_2016`) and FireProtDB stability assays.
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

## Empirical Benchmark: Full 50k GFP Dataset

The Phase 2 fitness prediction framework was evaluated on the complete experimental Deep Mutational Scanning (DMS) dataset for Green Fluorescent Protein from Sarkisyan et al. (*Nature* 2016), comprising **51,027 valid mutational variants**.

### 1. In-Distribution Model Comparison (Random 80/20 Split)
Evaluated on **10,206 held-out test variants**:

| Model | Feature Representation | Spearman ($\rho$) | Pearson ($r$) | MAE | RMSE | $R^2$ | Top-10% Enrichment |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **MeanPredictor** | `emb_diff` | N/A | N/A | 1.0181 | 1.0532 | -0.0000 | 1.13x |
| **MedianPredictor** | `emb_diff` | N/A | N/A | 0.9306 | 1.2305 | -0.3651 | 1.13x |
| **ZeroShotESM2Predictor** | `emb_diff` | 0.9022 | 0.9826 | 1.3367 | 1.4396 | -0.8685 | 4.65x |
| **RidgeFitnessPredictor** | `emb_diff` | 0.9022 | 0.9826 | 0.4853 | 0.5121 | 0.7635 | 4.63x |
| **HistGradientBoostingPredictor** | `emb_diff` | 0.9021 | 0.9917 | 0.3737 | 0.4026 | 0.8539 | 4.63x |
| **RandomForestFitnessPredictor** | `emb_diff` | **0.9022** | **0.9919** | **0.1030** | **0.1342** | **0.9838** | 4.61x |

### 2. Generalization Boundaries (Out-of-Distribution Splits)

| Model | Split Regime | Scientific Hypothesis | Spearman ($\rho$) | MAE | RMSE | $R^2$ |
|---|---|---|:---:|:---:|:---:|:---:|
| **ZeroShotESM2** | `position_held_out` | Spatial (Unseen Residues) | 0.9066 | 1.2750 | 1.3854 | -0.6885 |
| **RidgeRegression** | `position_held_out` | Spatial (Unseen Residues) | 0.9066 | 0.8363 | 0.9691 | 0.1737 |
| **HistGradientBoosting** | `position_held_out` | Spatial (Unseen Residues) | **0.9064** | **0.3809** | **0.4460** | **0.8250** |
| **ZeroShotESM2** | `mutation_order` | Epistatic (Higher-Order) | 0.9036 | 1.2101 | 1.3243 | -0.5683 |
| **RidgeRegression** | `mutation_order` | Epistatic (Higher-Order) | 0.9034 | 1.0380 | 1.3305 | -0.5831 |
| **HistGradientBoosting** | `mutation_order` | Epistatic (Higher-Order) | **0.8886** | **0.4471** | **0.5549** | **0.7246** |

### 3. Feature Representation Ablation

| Model | Representation | Spearman ($\rho$) | Pearson ($r$) | MAE | RMSE | $R^2$ |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **RidgeRegression** | `seq_emb` (640-d) | 0.9022 | 0.9826 | 0.4853 | 0.5121 | 0.7635 |
| **RidgeRegression** | `emb_diff` ($\Delta \mathbf{h}$) | 0.9022 | 0.9826 | 0.4853 | 0.5121 | 0.7635 |
| **RidgeRegression** | `emb_concat` (1280-d) | 0.9021 | 0.9826 | 0.3292 | 0.3633 | 0.8810 |
| **RidgeRegression** | `mutation_features` | 0.5689 | 0.5858 | 0.7115 | 0.8537 | 0.3430 |
| **HistGBDT** | `hybrid` ($\Delta \mathbf{h}$ + PhysChem) | **0.9046** | **0.9918** | **0.1038** | **0.1344** | **0.9837** |

*All raw artifacts and summary tables are available in [`reports/phase2/`](reports/phase2/).*

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
Fetches the full Sarkisyan et al. GFP dataset (51k variants) and UniProt wild-type reference:
```powershell
.\.venv\Scripts\python.exe scripts/download_data.py
```

### 2. Run the Full Phase 2 Benchmark Matrix
Executes EXP-1 (Random Split), EXP-2 (Position-Held-Out), EXP-3 (Mutation-Order), and EXP-4 (Feature Ablation) across all 51k variants:
```powershell
.\.venv\Scripts\python.exe scripts/run_phase2_benchmark.py --csv data/raw/proteingym/GFP_AEQVI_Sarkisyan_2016.csv --reports_dir reports/phase2
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
Executes all 167 unit and integration tests across all 7 phases:
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
│       ├── proteingym/       # Experimental DMS assays (GFP 51k variants)
│       └── uniprot/          # Canonical FASTA reference sequences
├── reports/
│   └── phase2/               # Publication-grade CSV, JSON, and MD benchmarks
├── scripts/
│   ├── download_data.py      # Dataset acquisition script
│   └── run_phase2_benchmark.py # Full benchmark runner across all splits
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
│   ├── integration/          # End-to-end multi-phase pipeline tests (6 suites)
│   └── unit/                 # Exhaustive unit test coverage (161 tests)
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
