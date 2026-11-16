"""Unit tests for ProteinGym and FireProtDB ingestion pipelines."""

import csv
import io
from pathlib import Path

from evanthea.data.ingestion import FireProtDBIngestor, ProteinGymIngestor, compute_robust_z_scores
from evanthea.data.schemas import ProteinRecord
from evanthea.protein.sequence import validate_amino_acid_string

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "fixtures" / "sample_gfp_dms.csv"
SAMPLE_TEM1_CSV = Path(__file__).resolve().parent.parent / "fixtures" / "sample_tem1_dms.csv"
SAMPLE_FIREPROT = Path(__file__).resolve().parent.parent / "fixtures" / "sample_fireprotdb.tsv"

GFP_WT = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQ"
    "HDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNG"
    "IKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
)

TEM1_WT = (
    "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYS"
    "QNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLL"
    "TGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"
)


def test_compute_robust_z_scores():
    scores = [1.0, 2.0, 3.0, 4.0, 5.0]
    z = compute_robust_z_scores(scores)
    assert len(z) == 5
    assert z[2] == 0.0  # Median (3.0) maps to 0.0


def test_proteingym_ingestor_csv():
    protein = ProteinRecord(
        id="GFP_AEQVI_Sarkisyan_2016",
        name="GFP",
        wildtype_sequence=GFP_WT,
        length=len(GFP_WT),
    )
    ingestor = ProteinGymIngestor(protein=protein)
    variants, summary = ingestor.ingest_csv(SAMPLE_CSV)

    assert summary.total_records == 10
    assert summary.valid_records == 10
    assert summary.invalid_records == 0
    assert summary.wildtype_records == 1
    assert summary.single_mutants == 6
    assert summary.multi_mutants == 3

    # Check normalized scores populated
    assert all(v.normalized_score is not None for v in variants)


def test_fireprotdb_ingestor():
    protein = ProteinRecord(
        id="P42212",
        name="GFP",
        wildtype_sequence=GFP_WT,
        length=len(GFP_WT),
    )
    with open(SAMPLE_FIREPROT, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    var, issue = FireProtDBIngestor.parse_stability_record(rows[0], protein)
    assert issue is None
    assert var is not None
    assert var.metadata["ddG_kcal_mol"] == -0.45
    assert var.metadata["dTm_celsius"] == 1.2
    assert var.metadata["pH"] == "7.4"


def test_proteingym_ingestor_tem1():
    """Validates TEM-1 Beta-lactamase DMS assay ingestion against canonical P62593 reference."""
    protein = ProteinRecord(
        id="BLAT_ECOLX_Firnberg_2014",
        name="Beta-lactamase TEM-1",
        wildtype_sequence=TEM1_WT,
        length=len(TEM1_WT),
        organism="Escherichia coli",
        uniprot_accession="P62593",
    )
    ingestor = ProteinGymIngestor(protein=protein)
    variants, summary = ingestor.ingest_csv(SAMPLE_TEM1_CSV)

    assert summary.total_records == 10
    assert summary.valid_records == 10
    assert summary.invalid_records == 0
    assert summary.single_mutants == 10
    assert summary.multi_mutants == 0

    # Ensure all variants mutated at position 24 (H24) and have valid scores
    for v in variants:
        assert v.num_mutations == 1
        assert v.mutations[0].position_1based == 24
        assert v.mutations[0].ref_residue == "H"
        assert v.raw_score is not None
        assert v.normalized_score is not None
