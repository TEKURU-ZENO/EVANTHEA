"""Structural validation coordinator applying biophysical quality gates and persistent caching."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from Bio.PDB import PDBParser

from evanthea.memory.storage import PersistentMemoryStore
from evanthea.protein.sequence import compute_sequence_hash, validate_amino_acid_string
from evanthea.structure.metrics import (
    compute_contact_map,
    compute_contact_preservation,
    compute_rmsd,
    compute_sasa,
    detect_steric_clashes,
)
from evanthea.structure.predictor import MockStructurePredictor, ReferencePDBLoader, StructurePredictor
from evanthea.structure.relaxation import StructureRelaxer
from evanthea.structure.schema import ClashRecord, StructuralAssessment, StructurePredictionResult


class StructuralValidator:
    """Coordinates 3D structure prediction, biophysical metric computation, and quality gate auditing."""

    def __init__(
        self,
        predictor: Optional[StructurePredictor] = None,
        min_plddt: float = 70.0,
        max_rmsd: float = 2.5,
        max_clashes: int = 5,
        min_contact_preservation: float = 0.85,
        max_hydrophobic_sasa_delta: float = 600.0,
        sasa_points: int = 48,
        relaxer: Optional[StructureRelaxer] = None,
        memory_store: Optional[PersistentMemoryStore] = None,
    ) -> None:
        self.predictor = predictor or MockStructurePredictor()
        self.min_plddt = min_plddt
        self.max_rmsd = max_rmsd
        self.max_clashes = max_clashes
        self.min_contact_preservation = min_contact_preservation
        self.max_hydrophobic_sasa_delta = max_hydrophobic_sasa_delta
        self.sasa_points = sasa_points
        self.relaxer = relaxer or StructureRelaxer()
        self.memory_store = memory_store
        self.reference_loader = ReferencePDBLoader()

        # Reference structure data
        self.ref_ca_coords: Optional[np.ndarray] = None
        self.ref_contacts: Optional[np.ndarray] = None
        self.ref_sequence: Optional[str] = None
        self.ref_hydrophobic_sasa: Optional[float] = None
        self.ref_pdb_text: Optional[str] = None

    def set_reference_from_text(self, pdb_text: str, structure_id: str = "REF") -> None:
        """Sets wild-type reference structure from PDB text."""
        struct = self.reference_loader.load_from_text(pdb_text, structure_id=structure_id)
        coords, seq = self.reference_loader.extract_ca_coordinates(struct)
        contacts = compute_contact_map(struct)
        _, hydro_sasa = compute_sasa(struct, n_points=self.sasa_points)

        self.ref_ca_coords = coords
        self.ref_contacts = contacts
        self.ref_sequence = seq
        self.ref_hydrophobic_sasa = hydro_sasa
        self.ref_pdb_text = pdb_text

    def set_reference_from_file(self, pdb_path: Union[Path, str], structure_id: str = "REF") -> None:
        """Sets wild-type reference structure from PDB file on disk."""
        with open(pdb_path, "r", encoding="utf-8") as f:
            self.set_reference_from_text(f.read(), structure_id=structure_id)

    def validate_candidate(
        self,
        sequence: str,
        variant_id: str = "VAR",
    ) -> Tuple[StructuralAssessment, str]:
        """Performs full 3D prediction, biophysical metric evaluation, and quality gate auditing.

        Args:
            sequence: Variant amino acid sequence.
            variant_id: Identifier for candidate variant.

        Returns:
            Tuple of (StructuralAssessment, pdb_text).
        """
        cleaned = validate_amino_acid_string(sequence)
        seq_hash = compute_sequence_hash(cleaned)

        # 1. Check persistent memory store cache
        if self.memory_store is not None:
            cached = self.memory_store.get_structural_evaluation(seq_hash)
            if cached is not None:
                assess_dict, pdb_text = cached
                clashes = [ClashRecord(**c) for c in assess_dict.get("clashes", [])]
                assess_dict["clashes"] = clashes
                assess_dict["sequence"] = cleaned
                assessment = StructuralAssessment(**assess_dict)
                assessment.metadata["cached"] = True
                return assessment, pdb_text

        # 2. Predict 3D structure
        pred_res = self.predictor.predict_structure(cleaned, variant_id=variant_id)
        raw_pdb = pred_res.pdb_text

        # 3. Optional relaxation / clash relief
        relaxed_pdb, was_relaxed = self.relaxer.relax_pdb_text(raw_pdb)
        final_pdb = relaxed_pdb

        # 4. Parse structure and compute biophysical metrics
        struct = self.reference_loader.load_from_text(final_pdb, structure_id=variant_id)
        mut_coords, _ = self.reference_loader.extract_ca_coordinates(struct)

        # Confidence profile
        plddt_per_res = pred_res.plddt_scores
        mean_plddt = pred_res.mean_plddt
        disordered_count = sum(1 for p in plddt_per_res if p < 50.0)

        # Steric clashes
        clashes = detect_steric_clashes(struct)
        clash_count = len(clashes)

        # Contact map & preservation
        mut_contacts = compute_contact_map(struct)
        contact_preservation = None
        if self.ref_contacts is not None and self.ref_contacts.shape == mut_contacts.shape:
            contact_preservation = compute_contact_preservation(self.ref_contacts, mut_contacts)

        # Kabsch RMSD to reference
        rmsd = None
        if self.ref_ca_coords is not None and len(self.ref_ca_coords) == len(mut_coords):
            rmsd = compute_rmsd(self.ref_ca_coords, mut_coords)

        # SASA
        total_sasa, hydro_sasa = compute_sasa(struct, n_points=self.sasa_points)

        # 5. Quality gate evaluation
        violations: List[str] = []
        if mean_plddt < self.min_plddt:
            violations.append(f"Mean pLDDT ({mean_plddt:.1f}) below threshold ({self.min_plddt:.1f})")

        if rmsd is not None and rmsd > self.max_rmsd:
            violations.append(f"C-alpha RMSD ({rmsd:.2f} A) exceeds maximum threshold ({self.max_rmsd:.2f} A)")

        if clash_count > self.max_clashes:
            violations.append(f"Severe steric clashes count ({clash_count}) exceeds threshold ({self.max_clashes})")

        if contact_preservation is not None and contact_preservation < self.min_contact_preservation:
            violations.append(f"Contact preservation ({contact_preservation:.2f}) below threshold ({self.min_contact_preservation:.2f})")

        if self.ref_hydrophobic_sasa is not None:
            delta_hydro = hydro_sasa - self.ref_hydrophobic_sasa
            if delta_hydro > self.max_hydrophobic_sasa_delta:
                violations.append(f"Excess exposed hydrophobic SASA (+{delta_hydro:.1f} A^2) exceeds aggregation risk limit ({self.max_hydrophobic_sasa_delta:.1f} A^2)")

        is_viable = len(violations) == 0

        assessment = StructuralAssessment(
            sequence_hash=seq_hash,
            variant_id=variant_id,
            sequence=cleaned,
            mean_plddt=mean_plddt,
            plddt_per_residue=plddt_per_res,
            disordered_residues_count=disordered_count,
            rmsd_to_reference=rmsd,
            clash_count=clash_count,
            clashes=clashes,
            contact_preservation_ratio=contact_preservation,
            sasa_total=total_sasa,
            sasa_hydrophobic=hydro_sasa,
            is_structurally_viable=is_viable,
            violations=violations,
            metadata={
                "was_relaxed": was_relaxed,
                "predictor": pred_res.metadata.get("predictor", "unknown"),
            },
        )

        # 6. Save to persistent memory store cache
        if self.memory_store is not None:
            assess_payload = assessment.model_dump()
            self.memory_store.save_structural_evaluation(assess_payload, final_pdb)

        return assessment, final_pdb

    def batch_validate(
        self,
        variants: List[Tuple[str, str]],
    ) -> List[Tuple[StructuralAssessment, str]]:
        """Validates a batch of (sequence, variant_id) tuples."""
        results = []
        for seq, var_id in variants:
            res = self.validate_candidate(seq, variant_id=var_id)
            results.append(res)
        return results
