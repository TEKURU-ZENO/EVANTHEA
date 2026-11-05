"""Pydantic V2 schemas representing 3D protein structures, confidence metrics, and biophysical assessments."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ClashRecord(BaseModel):
    """Details of a severe non-bonded atomic steric clash."""
    atom1_name: str = Field(..., description="Atom 1 identifier (e.g. CA, CB, CG)")
    atom2_name: str = Field(..., description="Atom 2 identifier")
    res1_id: int = Field(..., description="1-indexed sequence position of residue 1")
    res2_id: int = Field(..., description="1-indexed sequence position of residue 2")
    res1_name: str = Field(..., description="Residue 1 amino acid (e.g. LEU, VAL)")
    res2_name: str = Field(..., description="Residue 2 amino acid")
    distance: float = Field(..., description="Interatomic Euclidean distance in Angstroms")
    overlap: float = Field(..., description="Overlap depth beyond van der Waals contact tolerance in Angstroms")


class StructurePredictionResult(BaseModel):
    """Raw structure prediction outcome including atomic PDB representation and confidence scores."""
    sequence_hash: str = Field(..., description="SHA-256 hash of predicted amino acid sequence")
    variant_id: str = Field(..., description="Candidate variant identifier")
    sequence: str = Field(..., description="Full amino acid sequence")
    pdb_text: str = Field(..., description="Full PDB-formatted text containing 3D atomic coordinates")
    plddt_scores: List[float] = Field(default_factory=list, description="Per-residue pLDDT confidence scores in [0, 100]")
    mean_plddt: float = Field(..., ge=0.0, le=100.0, description="Global mean pLDDT confidence score")
    ptm_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Predicted TM-score (pTM) if provided by model")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Predictor provenance and runtime metrics")


class StructuralAssessment(BaseModel):
    """Comprehensive biophysical quality gate report for a candidate protein variant."""
    sequence_hash: str = Field(..., description="SHA-256 hash of assessed sequence")
    variant_id: str = Field(..., description="Candidate variant identifier")
    sequence: str = Field(..., description="Assessed amino acid sequence")
    mean_plddt: float = Field(..., ge=0.0, le=100.0, description="Global average pLDDT confidence")
    plddt_per_residue: List[float] = Field(default_factory=list, description="Per-residue pLDDT confidence profile")
    disordered_residues_count: int = Field(0, ge=0, description="Count of residues with pLDDT < 50 (disordered/unconfident)")
    rmsd_to_reference: Optional[float] = Field(None, ge=0.0, description="Kabsch C-alpha RMSD against wild-type reference in Angstroms")
    clash_count: int = Field(0, ge=0, description="Total count of severe non-bonded steric clashes")
    clashes: List[ClashRecord] = Field(default_factory=list, description="List of detected steric clashes")
    contact_preservation_ratio: Optional[float] = Field(None, ge=0.0, le=1.0, description="Fraction of wild-type tertiary contacts preserved")
    sasa_total: Optional[float] = Field(None, ge=0.0, description="Total Solvent Accessible Surface Area in Angstroms squared")
    sasa_hydrophobic: Optional[float] = Field(None, ge=0.0, description="Hydrophobic Solvent Accessible Surface Area in Angstroms squared")
    is_structurally_viable: bool = Field(True, description="True if candidate passes all biophysical quality gates")
    violations: List[str] = Field(default_factory=list, description="Explanations of violated structural criteria")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Assessment metadata and execution timestamps")
