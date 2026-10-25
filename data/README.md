# Data Storage & Directory Guidelines

In accordance with EVANTHEA repository design rules:
1. **Never commit raw or processed datasets to Git**.
2. All raw data files are stored under `data/raw/` (ignored by Git).
3. Ingested, standardized tabular records are stored under `data/processed/`.
4. Dataset splits manifests are stored under `data/splits/`.
5. Preprocessing scripts in `scripts/` reproduce all datasets with verified SHA-256 checksums.

### Subdirectories
- `data/raw/proteingym/`: Raw CSV files downloaded from the official ProteinGym repository.
- `data/raw/fireprotdb/`: Raw TSV/CSV stability records from FireProtDB.
- `data/raw/uniprot/`: Wild-type FASTA sequences fetched from UniProtKB.
- `data/processed/`: Standardized parquet/feather/csv variants with normalized coordinates.
- `data/splits/`: JSON split manifests containing variant hashes.
