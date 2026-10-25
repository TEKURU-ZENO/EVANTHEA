# EVANTHEA 🧬
### Evolutionary Variant Analysis for Theoretical Protein Adaptation

An Adaptive, Closed-Loop Framework for Protein Sequence Optimization Using Protein Language Models, Reinforcement Learning, and Multi-Objective Evolution.

---

## Architecture Overview
EVANTHEA formulates protein sequence design as a closed-loop evolutionary optimization process:
1. **Generative Prior**: High-quality biological sequence initialization.
2. **Smart Mutation Generator**: ESM-2 guided, biologically plausible substitutions and candidate proposals.
3. **Multi-Objective Fitness Engine**: Resolves trade-offs across functional activity, stability, and naturalness.
4. **Evolution Engine**: Population-based evolutionary algorithms (GA, NSGA-II) and reinforcement learning (PPO).
5. **Evolutionary Memory**: FAISS + SQLite persistent tracking preventing redundant exploration and identifying recurrent motifs.
6. **Structural Validation Engine**: Hierarchical structural verification via ESMFold and physics-informed relaxation.
7. **Research Dashboard**: Interactive inspection of Pareto fronts, phylogenetic lineages, and 3D molecular conformations.

---

## Phase 1: Dataset Ingestion and Protein Intelligence
Phase 1 implements:
- Rigid amino acid sequence validation and coordinate conversion (0-indexed vs 1-indexed).
- Multi-mutant parsing and wild-type alignment assertions.
- Benchmark ingestion for ProteinGym (GFP `GFP_AEQVI_Sarkisyan_2016`) and FireProtDB stability datasets.
- Frozen ESM-2 (`facebook/esm2_t30_150M_UR50D`) inference wrapper via Hugging Face Transformers.
- Masked-marginal scoring, sequence embeddings, and pseudo-perplexity calculation.
- Leakage-resistant splitting strategies (random, position-held-out, mutation-order).
- Persistent SQLite metadata and HDF5 embedding cache.

---

## Getting Started
Ensure Python 3.11 is installed.
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```
