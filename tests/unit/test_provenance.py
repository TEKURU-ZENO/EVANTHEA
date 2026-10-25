"""Unit tests for SHA-256 provenance calculation and manifest generation."""

from pathlib import Path
from evanthea.data.provenance import compute_file_sha256, create_provenance_manifest, verify_file_checksum

FIXTURE_PATH = Path("tests/fixtures/sample_gfp_dms.csv")


def test_compute_file_sha256():
    sha = compute_file_sha256(FIXTURE_PATH)
    assert len(sha) == 64
    assert verify_file_checksum(FIXTURE_PATH, sha) is True
    assert verify_file_checksum(FIXTURE_PATH, "wrong_hash") is False


def test_create_provenance_manifest(tmp_path: Path):
    manifest_out = tmp_path / "manifest.json"
    meta = create_provenance_manifest(
        dataset_name="TestDataset",
        raw_file_path=FIXTURE_PATH,
        record_count=10,
        output_manifest_path=manifest_out,
        citation="Test citation",
    )
    assert meta.dataset_name == "TestDataset"
    assert meta.record_count == 10
    assert manifest_out.is_file()
