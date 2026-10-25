"""Data provenance, integrity verification, and SHA-256 manifest management."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from evanthea.data.schemas import ProvenanceMetadata


def compute_file_sha256(file_path: Path | str, chunk_size: int = 65536) -> str:
    """Computes the SHA-256 checksum of a file on disk."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for checksum computation: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_file_checksum(file_path: Path | str, expected_sha256: str) -> bool:
    """Verifies that a file matches its expected SHA-256 checksum."""
    actual_sha256 = compute_file_sha256(file_path)
    return actual_sha256.lower() == expected_sha256.lower()


def create_provenance_manifest(
    dataset_name: str,
    raw_file_path: Path | str,
    record_count: int,
    output_manifest_path: Path | str,
    source_url: Optional[str] = None,
    license_str: Optional[str] = None,
    citation: Optional[str] = None,
    notes: Optional[str] = None,
) -> ProvenanceMetadata:
    """Creates and persists an immutable JSON provenance manifest for an ingested dataset."""
    file_path = Path(raw_file_path)
    sha256 = compute_file_sha256(file_path)

    metadata = ProvenanceMetadata(
        dataset_name=dataset_name,
        source_url=source_url,
        retrieval_date=datetime.now(timezone.utc).isoformat(),
        raw_file_sha256=sha256,
        record_count=record_count,
        license=license_str,
        citation=citation,
        notes=notes,
    )

    out_path = Path(output_manifest_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata.model_dump(), f, indent=2)

    return metadata
