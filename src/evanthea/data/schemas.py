"""Pydantic V2 schemas representing proteins, variants, evaluations, and provenance."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from evanthea.constants import STANDARD_AMINO_ACIDS


class Mutation(BaseModel):
    """Represents a single amino acid substitution at a 1-indexed position."""
    ref_residue: str = Field(..., min_length=1, max_length=1, description="Wild-type amino acid (1-letter)")
    position_1based: int = Field(..., gt=0, description="1-indexed sequence position")
    mut_residue: str = Field(..., min_length=1, max_length=1, description="Mutant amino acid (1-letter)")

    @field_validator("ref_residue", "mut_residue")
    @classmethod
    def validate_aa(cls, v: str) -> str:
        v_upper = v.upper()
        if v_upper not in STANDARD_AMINO_ACIDS:
            raise ValueError(f"Invalid amino acid '{v}'. Must be one of canonical 20: {sorted(STANDARD_AMINO_ACIDS)}")
        return v_upper

    @property
    def position_0based(self) -> int:
        """Returns 0-indexed position for Python slicing and tensor operations."""
        return self.position_1based - 1

    def to_string(self) -> str:
        """Standard HGVS-like notation e.g. 'M1A'."""
        return f"{self.ref_residue}{self.position_1based}{self.mut_residue}"

    def __str__(self) -> str:
        return self.to_string()

    def __hash__(self) -> int:
        return hash((self.ref_residue, self.position_1based, self.mut_residue))


class ProteinRecord(BaseModel):
    """Represents a reference protein sequence and its biological metadata."""
    id: str = Field(..., description="Unique accession ID or assay identifier")
    name: str = Field(..., description="Common protein name (e.g. GFP, TEM-1)")
    wildtype_sequence: str = Field(..., min_length=1, description="Full wild-type canonical sequence")
    length: int = Field(..., gt=0, description="Length of sequence in residues")
    organism: Optional[str] = Field(None, description="Host or source organism")
    uniprot_accession: Optional[str] = Field(None, description="UniProtKB accession identifier")
    description: Optional[str] = Field(None, description="Protein functional description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary provenance and metadata")

    @model_validator(mode="before")
    @classmethod
    def sync_length(cls, data: Any) -> Any:
        if isinstance(data, dict):
            seq = data.get("wildtype_sequence", "")
            if seq and "length" not in data:
                data["length"] = len(seq)
        return data


class VariantRecord(BaseModel):
    """Represents a specific protein variant (wild-type or mutated) with experimental/fitness scores."""
    variant_id: str = Field(..., description="Unique variant identifier (hash or string)")
    protein_id: str = Field(..., description="Reference protein ID this variant was derived from")
    mutations: List[Mutation] = Field(default_factory=list, description="Ordered list of point mutations")
    mutated_sequence: str = Field(..., min_length=1, description="Full variant amino acid sequence")
    raw_score: Optional[float] = Field(None, description="Raw experimental assay measurement")
    normalized_score: Optional[float] = Field(None, description="Robust z-scored or min-max normalized fitness")
    score_metric_name: Optional[str] = Field(None, description="Name of experimental assay score")
    is_wildtype: bool = Field(False, description="True if variant has 0 mutations")
    num_mutations: int = Field(0, ge=0, description="Count of substitutions")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional assay metadata")

    @property
    def mutation_string(self) -> str:
        """Returns standard colon-separated representation e.g. 'M1A:G2C' or 'WT'."""
        if not self.mutations:
            return "WT"
        return ":".join(m.to_string() for m in self.mutations)


class EvaluationRecord(BaseModel):
    """Structured multi-objective evaluation result for a protein variant."""
    variant_id: str = Field(..., description="Variant identifier")
    sequence_plausibility: Optional[float] = Field(None, description="Pseudo-perplexity or sequence likelihood")
    masked_marginal_score: Optional[float] = Field(None, description="ESM-2 zero-shot log-odds score")
    predicted_stability: Optional[float] = Field(None, description="Predicted dTm or ddG")
    assay_fitness_estimate: Optional[float] = Field(None, description="Learned model fitness estimate")
    is_feasible: bool = Field(True, description="True if candidate satisfies feasibility constraints")
    feasibility_violations: List[str] = Field(default_factory=list, description="List of violated constraints")
    evaluation_provenance: Dict[str, Any] = Field(default_factory=dict, description="Model checkpoint and config provenance")


class ProvenanceMetadata(BaseModel):
    """Immutable provenance record for ingested datasets."""
    dataset_name: str
    source_url: Optional[str] = None
    retrieval_date: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    raw_file_sha256: str
    record_count: int
    license: Optional[str] = None
    citation: Optional[str] = None
    notes: Optional[str] = None


class DatasetSplitManifest(BaseModel):
    """Manifest defining leakage-resistant dataset partitions."""
    dataset_id: str
    split_strategy: str = Field(..., description="'random', 'position_held_out', 'mutation_order', or 'protein_family'")
    scientific_question: str = Field(..., description="The exact generalization hypothesis tested by this split")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    seed: int
    train_variant_ids: List[str]
    val_variant_ids: List[str]
    test_variant_ids: List[str]
    leakage_verification: Dict[str, Any] = Field(default_factory=dict)
