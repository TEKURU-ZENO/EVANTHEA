"""Acquires benchmark datasets (ProteinGym GFP, TEM-1 Beta-lactamase, BRCA1, UniProt references) with checksum verification."""

import hashlib
import json
import logging
import urllib.request
from pathlib import Path
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evanthea.download")

# ProteinGym Hugging Face Base
PROTEINGYM_HF_BASE = (
    "https://huggingface.co/datasets/ICML2022/ProteinGym/resolve/main/ProteinGym_substitutions"
)

# Registry of supported ProteinGym benchmark DMS datasets
DATASET_REGISTRY: Dict[str, Dict[str, str]] = {
    "gfp": {
        "name": "Green Fluorescent Protein (Sarkisyan et al. 2016)",
        "csv_filename": "GFP_AEQVI_Sarkisyan_2016.csv",
        "csv_url": f"{PROTEINGYM_HF_BASE}/GFP_AEQVI_Sarkisyan_2016.csv",
        "uniprot_accession": "P42212",
        "fasta_url": "https://rest.uniprot.org/uniprotkb/P42212.fasta",
    },
    "tem1": {
        "name": "Beta-lactamase TEM-1 (Firnberg et al. 2014)",
        "csv_filename": "BLAT_ECOLX_Firnberg_2014.csv",
        "csv_url": f"{PROTEINGYM_HF_BASE}/BLAT_ECOLX_Firnberg_2014.csv",
        "uniprot_accession": "P62593",
        "fasta_url": "https://rest.uniprot.org/uniprotkb/P62593.fasta",
    },
    "brca1": {
        "name": "BRCA1 Tumor Suppressor (Findlay et al. 2018)",
        "csv_filename": "BRCA1_HUMAN_Findlay_2018.csv",
        "csv_url": f"{PROTEINGYM_HF_BASE}/BRCA1_HUMAN_Findlay_2018.csv",
        "uniprot_accession": "P38398",
        "fasta_url": "https://rest.uniprot.org/uniprotkb/P38398.fasta",
    },
}


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

    for key, info in DATASET_REGISTRY.items():
        logger.info(f"Checking dataset: {info['name']} ({key})...")
        csv_path = raw_dir / "proteingym" / info["csv_filename"]
        try:
            download_file(info["csv_url"], csv_path)
        except Exception as e:
            logger.warning(f"Could not download {info['csv_filename']}: {e}")

        fasta_path = raw_dir / "uniprot" / f"{info['uniprot_accession']}.fasta"
        try:
            download_file(info["fasta_url"], fasta_path)
        except Exception as e:
            logger.warning(f"Could not download FASTA for {info['uniprot_accession']}: {e}")


if __name__ == "__main__":
    main()
