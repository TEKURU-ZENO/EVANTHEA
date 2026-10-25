"""Ingestion, coordinate reconciliation, and split generation for ProteinGym GFP benchmark."""

import argparse
import json
import logging
from pathlib import Path

from evanthea.data.ingestion import ProteinGymIngestor
from evanthea.data.provenance import compute_file_sha256, create_provenance_manifest
from evanthea.data.schemas import ProteinRecord
from evanthea.data.splits import (
    create_mutation_order_split,
    create_position_held_out_split,
    create_random_split,
)
from evanthea.protein.sequence import validate_amino_acid_string

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evanthea.prepare_proteingym")

# Canonical wild-type sequence for GFP_AEQVI_Sarkisyan_2016 (238 residues)
# Sarkisyan et al. engineered avGFP construct
GFP_SARKISYAN_WT = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQ"
    "HDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNG"
    "IKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)


def run_pipeline(
    raw_csv_path: Path,
    output_dir: Path,
    splits_dir: Path,
    reports_dir: Path,
    max_rows: int | None = None,
):
    logger.info("Initializing GFP canonical ProteinRecord...")
    wt_seq = validate_amino_acid_string(GFP_SARKISYAN_WT)
    gfp_protein = ProteinRecord(
        id="GFP_AEQVI_Sarkisyan_2016",
        name="Green Fluorescent Protein",
        wildtype_sequence=wt_seq,
        length=len(wt_seq),
        organism="Aequorea victoria",
        uniprot_accession="P42212",
        description="avGFP engineered construct from Sarkisyan et al. (2016)",
        metadata={"assay": "Fluorescence", "source": "ProteinGym"},
    )

    logger.info(f"Ingesting ProteinGym CSV: {raw_csv_path} (max_rows={max_rows})...")
    ingestor = ProteinGymIngestor(protein=gfp_protein)
    valid_variants, summary = ingestor.ingest_csv(raw_csv_path, max_rows=max_rows)

    logger.info(
        f"Validation summary: {summary.valid_records}/{summary.total_records} valid records "
        f"({summary.validity_rate*100:.2f}%). WT={summary.wildtype_records}, "
        f"Singles={summary.single_mutants}, Multi={summary.multi_mutants}."
    )

    if summary.issues:
        logger.warning(f"Detected {len(summary.issues)} validation issues. First 3: {summary.issues[:3]}")

    # Generate Leakage-Resistant Splits
    logger.info("Generating leakage-resistant splits...")
    splits_dir.mkdir(parents=True, exist_ok=True)
    dataset_id = gfp_protein.id

    # 1. Random Split
    random_manifest = create_random_split(dataset_id, valid_variants, seed=42)
    with open(splits_dir / f"{dataset_id}_random_split.json", "w", encoding="utf-8") as f:
        json.dump(random_manifest.model_dump(), f, indent=2)

    # 2. Position-Held-Out Split
    position_manifest = create_position_held_out_split(
        dataset_id, valid_variants, sequence_length=gfp_protein.length, seed=42
    )
    with open(splits_dir / f"{dataset_id}_position_held_out_split.json", "w", encoding="utf-8") as f:
        json.dump(position_manifest.model_dump(), f, indent=2)

    # 3. Mutation-Order Split
    order_manifest = create_mutation_order_split(dataset_id, valid_variants, seed=42)
    with open(splits_dir / f"{dataset_id}_mutation_order_split.json", "w", encoding="utf-8") as f:
        json.dump(order_manifest.model_dump(), f, indent=2)

    # Save Provenance & Audit Report
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_data = {
        "dataset_id": dataset_id,
        "raw_csv_sha256": compute_file_sha256(raw_csv_path),
        "validation_summary": summary.model_dump(),
        "splits": {
            "random": random_manifest.leakage_verification,
            "position_held_out": position_manifest.leakage_verification,
            "mutation_order": order_manifest.leakage_verification,
        },
    }
    with open(reports_dir / f"{dataset_id}_audit_report.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    logger.info(f"Pipeline complete. Outputs saved to {splits_dir} and {reports_dir}.")


def main():
    parser = argparse.ArgumentParser(description="Ingest ProteinGym GFP dataset.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/raw/proteingym/GFP_AEQVI_Sarkisyan_2016.csv"),
        help="Path to raw ProteinGym CSV file",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--splits_dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--reports_dir", type=Path, default=Path("reports"))
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()

    run_pipeline(
        raw_csv_path=args.csv,
        output_dir=args.output_dir,
        splits_dir=args.splits_dir,
        reports_dir=args.reports_dir,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
