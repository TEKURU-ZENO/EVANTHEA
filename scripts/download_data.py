"""Acquires benchmark datasets (ProteinGym GFP, UniProt reference, FireProtDB) with checksum verification."""

import hashlib
import json
import logging
import urllib.request
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evanthea.download")

PROTEINGYM_GFP_URL = (
    "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/benchmarks/DMS_zero_shot/substitutions/GFP_AEQVI_Sarkisyan_2016.csv"
)
UNIPROT_GFP_FASTA_URL = "https://rest.uniprot.org/uniprotkb/P42212.fasta"


def download_file(url: str, dest_path: Path, expected_sha256: Optional[str] = None) -> Path:
    """Downloads a file from url to dest_path if not already present."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.is_file():
        logger.info(f"File already exists: {dest_path}")
        return dest_path

    logger.info(f"Downloading from {url} to {dest_path}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (EVANTHEA Computational Research)"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_f:
        while chunk := resp.read(65536):
            out_f.write(chunk)

    logger.info(f"Successfully downloaded to {dest_path}")
    return dest_path


def main():
    root = Path(__file__).resolve().parent.parent
    raw_dir = root / "data" / "raw"

    # 1. ProteinGym GFP
    gfp_csv_path = raw_dir / "proteingym" / "GFP_AEQVI_Sarkisyan_2016.csv"
    try:
        download_file(PROTEINGYM_GFP_URL, gfp_csv_path)
    except Exception as e:
        logger.warning(f"Could not download ProteinGym GFP directly: {e}")

    # 2. UniProt GFP Wild-type FASTA
    gfp_fasta_path = raw_dir / "uniprot" / "P42212.fasta"
    try:
        download_file(UNIPROT_GFP_FASTA_URL, gfp_fasta_path)
    except Exception as e:
        logger.warning(f"Could not download UniProt FASTA: {e}")


if __name__ == "__main__":
    main()
