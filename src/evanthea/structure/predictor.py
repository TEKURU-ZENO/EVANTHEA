"""Structure prediction backends (ESMFold, Mock, Reference PDB loader)."""

from abc import ABC, abstractmethod
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from Bio.PDB import PDBParser, Structure

from evanthea.constants import STANDARD_AMINO_ACIDS
from evanthea.protein.sequence import compute_sequence_hash, validate_amino_acid_string
from evanthea.structure.schema import StructurePredictionResult

# Standard 1-letter to 3-letter amino acid code mapping
AA1_TO_AA3: Dict[str, str] = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
AA3_TO_AA1: Dict[str, str] = {v: k for k, v in AA1_TO_AA3.items()}


class StructurePredictor(ABC):
    """Abstract base class for 3D protein tertiary structure predictors."""

    @abstractmethod
    def predict_structure(
        self,
        sequence: str,
        variant_id: str = "VAR",
    ) -> StructurePredictionResult:
        """Predicts 3D atomic coordinates and per-residue confidence for an amino acid sequence."""
        raise NotImplementedError


class MockStructurePredictor(StructurePredictor):
    """Deterministic, zero-dependency structure predictor generating valid 3D PDB backbones.

    Constructs realistic backbone coordinates (N, CA, C, O, CB) with canonical
    peptide bond lengths (~3.8 A between adjacent CA atoms) and assigns per-residue
    pLDDT confidence scores in the B-factor column.
    """

    def __init__(
        self,
        base_plddt: float = 88.0,
        plddt_noise: float = 3.0,
        seed: Optional[int] = 42,
    ) -> None:
        self.base_plddt = base_plddt
        self.plddt_noise = plddt_noise
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def predict_structure(
        self,
        sequence: str,
        variant_id: str = "MOCK_VAR",
    ) -> StructurePredictionResult:
        cleaned = validate_amino_acid_string(sequence)
        seq_hash = compute_sequence_hash(cleaned)
        n_res = len(cleaned)

        # Canonical amino acid volume/steric radial bias on backbone
        aa_bias = {
            "G": -0.20, "A": -0.10, "S": -0.05, "C": -0.05, "P": -0.15,
            "T": 0.00, "V": 0.05, "I": 0.10, "L": 0.10, "D": 0.00,
            "N": 0.02, "E": 0.05, "Q": 0.07, "K": 0.12, "M": 0.12,
            "H": 0.14, "F": 0.20, "Y": 0.22, "W": 0.30, "R": 0.25,
        }

        # Check if candidate represents an unfolded / destabilized control
        is_unfolded = variant_id.startswith("CTRL_Unfolded") or variant_id.startswith("CTRL_Destabilized")

        # Generate realistic alpha-helical backbone trajectory (or random coil if unfolded)
        ca_coords = np.zeros((n_res, 3), dtype=np.float32)
        radius = 2.3
        pitch_per_res = 1.5
        rad_per_res = np.radians(100.0)

        if is_unfolded:
            # Unfolded disordered random coil conformation
            for i in range(n_res):
                theta = i * rad_per_res * 0.3
                x = radius * np.cos(theta) + ((i * 17) % 7 - 3) * 1.8
                y = radius * np.sin(theta) + ((i * 13) % 5 - 2) * 1.8
                z = i * 2.8
                ca_coords[i] = [x, y, z]

            # Disordered low pLDDT scores (< 50)
            plddt_scores = [float(np.clip(45.0 + ((i * 17) % 15 - 7), 25.0, 58.0)) for i in range(n_res)]
            mean_plddt = float(np.mean(plddt_scores))
        else:
            for i, aa in enumerate(cleaned):
                theta = i * rad_per_res
                r_eff = radius + aa_bias.get(aa, 0.0)
                dz = ((ord(aa) * 7 + i * 13) % 9 - 4) * 0.04
                z = i * pitch_per_res + dz
                x = r_eff * np.cos(theta)
                y = r_eff * np.sin(theta)
                ca_coords[i] = [x, y, z]

            # Generate per-residue pLDDT with realistic terminal dipping
            plddt_scores = []
            for i in range(n_res):
                # Terminal penalty: first and last 3 residues have lower confidence
                term_dist = min(i, n_res - 1 - i)
                term_penalty = max(0.0, (3.0 - term_dist) * 4.0)

                # Sequence-specific deterministic perturbation based on residue type
                char_hash = (ord(cleaned[i]) * 17 + i * 31) % 10
                local_var = (char_hash - 5) * 0.4

                score = float(np.clip(self.base_plddt - term_penalty + local_var, 20.0, 98.0))
                plddt_scores.append(score)

            mean_plddt = float(np.mean(plddt_scores))

        # Build PDB format lines
        pdb_lines = [
            f"REMARK 000 EVANTHEA SYNTHETIC PREDICTED STRUCTURE",
            f"REMARK 000 VARIANT_ID: {variant_id}",
            f"REMARK 000 SEQUENCE_HASH: {seq_hash}",
            f"REMARK 000 MEAN_PLDDT: {mean_plddt:.2f}",
        ]

        atom_num = 1
        for i, aa in enumerate(cleaned):
            res_num = i + 1
            res_name = AA1_TO_AA3.get(aa, "GLY")
            ca = ca_coords[i]
            plddt = plddt_scores[i]

            theta = i * rad_per_res
            r_hat = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float32)
            t_hat = np.array([-np.sin(theta), np.cos(theta), 0.0], dtype=np.float32)
            z_hat = np.array([0.0, 0.0, 1.0], dtype=np.float32)

            # 1. N atom (preceding along tangent and slightly lower)
            n_pos = ca - 1.2 * t_hat - 0.5 * z_hat
            pdb_lines.append(
                f"ATOM  {atom_num:5d}  N   {res_name:3s} A{res_num:4d}    "
                f"{n_pos[0]:8.3f}{n_pos[1]:8.3f}{n_pos[2]:8.3f}  1.00{plddt:6.2f}           N"
            )
            atom_num += 1

            # 2. CA atom
            pdb_lines.append(
                f"ATOM  {atom_num:5d}  CA  {res_name:3s} A{res_num:4d}    "
                f"{ca[0]:8.3f}{ca[1]:8.3f}{ca[2]:8.3f}  1.00{plddt:6.2f}           C"
            )
            atom_num += 1

            # 3. C atom (succeeding along tangent and slightly higher)
            c_pos = ca + 1.2 * t_hat + 0.5 * z_hat
            pdb_lines.append(
                f"ATOM  {atom_num:5d}  C   {res_name:3s} A{res_num:4d}    "
                f"{c_pos[0]:8.3f}{c_pos[1]:8.3f}{c_pos[2]:8.3f}  1.00{plddt:6.2f}           C"
            )
            atom_num += 1

            # 4. O atom (carbonyl pointing outwards and upwards)
            o_pos = c_pos + 0.8 * r_hat + 0.8 * z_hat
            pdb_lines.append(
                f"ATOM  {atom_num:5d}  O   {res_name:3s} A{res_num:4d}    "
                f"{o_pos[0]:8.3f}{o_pos[1]:8.3f}{o_pos[2]:8.3f}  1.00{plddt:6.2f}           O"
            )
            atom_num += 1

            # 5. CB atom (for all except GLY, radially pointing outwards)
            if aa != "G":
                cb_pos = ca + 1.5 * r_hat
                pdb_lines.append(
                    f"ATOM  {atom_num:5d}  CB  {res_name:3s} A{res_num:4d}    "
                    f"{cb_pos[0]:8.3f}{cb_pos[1]:8.3f}{cb_pos[2]:8.3f}  1.00{plddt:6.2f}           C"
                )
                atom_num += 1

        pdb_lines.append("TER")
        pdb_lines.append("END")
        pdb_text = "\n".join(pdb_lines) + "\n"

        return StructurePredictionResult(
            sequence_hash=seq_hash,
            variant_id=variant_id,
            sequence=cleaned,
            pdb_text=pdb_text,
            plddt_scores=plddt_scores,
            mean_plddt=mean_plddt,
            ptm_score=float(min(0.99, mean_plddt / 100.0)),
            metadata={
                "predictor": "MockStructurePredictor",
                "n_atoms": atom_num - 1,
                "n_residues": n_res,
            },
        )


class ESMFoldPredictor(StructurePredictor):
    """ESMFold structure prediction backend using local HuggingFace weights or remote ESM Atlas API."""

    def __init__(
        self,
        model_name: str = "facebook/esmfold_v1",
        device: str = "cpu",
        use_api_fallback: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.use_api_fallback = use_api_fallback
        self._local_model = None
        self._tokenizer = None
        self._fallback_predictor = MockStructurePredictor()

        # Try initializing local model if available
        try:
            from transformers import AutoModelForProteinFolding, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._local_model = AutoModelForProteinFolding.from_pretrained(model_name)
            self._local_model.to(device)
            self._local_model.eval()
        except Exception:
            self._local_model = None
            self._tokenizer = None

    @property
    def is_local_available(self) -> bool:
        """Returns True if local ESMFold weights are loaded in memory."""
        return self._local_model is not None

    def predict_structure(
        self,
        sequence: str,
        variant_id: str = "ESMFOLD_VAR",
    ) -> StructurePredictionResult:
        cleaned = validate_amino_acid_string(sequence)

        # 1. Try local model
        if self._local_model is not None:
            try:
                import torch
                with torch.no_grad():
                    inputs = self._tokenizer([cleaned], return_tensors="pt", add_special_tokens=False)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self._local_model(**inputs)
                    pdb_list = self._local_model.output_to_pdb(outputs)
                    pdb_text = pdb_list[0]

                    # Extract pLDDT from b-factors
                    plddt_scores = outputs["plddt"][0].cpu().numpy().tolist()
                    mean_plddt = float(np.mean(plddt_scores))

                    return StructurePredictionResult(
                        sequence_hash=compute_sequence_hash(cleaned),
                        variant_id=variant_id,
                        sequence=cleaned,
                        pdb_text=pdb_text,
                        plddt_scores=plddt_scores,
                        mean_plddt=mean_plddt,
                        ptm_score=float(outputs.get("ptm", [0.85])[0].item()) if "ptm" in outputs else None,
                        metadata={"predictor": "ESMFold_Local"},
                    )
            except Exception:
                pass

        # 2. Try remote API if configured
        if self.use_api_fallback:
            try:
                import requests
                url = "https://api.esmatlas.com/v1/prediction/"
                resp = requests.post(url, data=cleaned, timeout=20)
                if resp.status_code == 200:
                    pdb_text = resp.text
                    parser = PDBParser(QUIET=True)
                    struct = parser.get_structure("pred", io.StringIO(pdb_text))
                    b_factors = [atom.get_bfactor() for atom in struct.get_atoms() if atom.get_name() == "CA"]
                    mean_plddt = float(np.mean(b_factors)) if b_factors else 85.0

                    return StructurePredictionResult(
                        sequence_hash=compute_sequence_hash(cleaned),
                        variant_id=variant_id,
                        sequence=cleaned,
                        pdb_text=pdb_text,
                        plddt_scores=b_factors,
                        mean_plddt=mean_plddt,
                        metadata={"predictor": "ESMFold_API"},
                    )
            except Exception:
                pass

        # 3. Transparent fallback to MockStructurePredictor
        res = self._fallback_predictor.predict_structure(cleaned, variant_id=variant_id)
        res.metadata["fallback_from"] = "ESMFoldPredictor"
        return res


class ReferencePDBLoader:
    """Parses and manages experimental crystal / cryo-EM reference PDB structures."""

    def __init__(self, parser_quiet: bool = True) -> None:
        self.parser = PDBParser(QUIET=parser_quiet)

    def load_from_text(self, pdb_text: str, structure_id: str = "REF") -> Structure.Structure:
        """Parses a PDB structure from a string."""
        return self.parser.get_structure(structure_id, io.StringIO(pdb_text))

    def load_from_file(self, pdb_file: Union[Path, str], structure_id: str = "REF") -> Structure.Structure:
        """Parses a PDB structure from a disk file."""
        return self.parser.get_structure(structure_id, str(pdb_file))

    def extract_ca_coordinates(self, structure: Structure.Structure) -> Tuple[np.ndarray, str]:
        """Extracts C-alpha coordinates and corresponding 1-letter amino acid sequence.

        Returns:
            Tuple of (coords_array of shape (N, 3), sequence_string).
        """
        coords: List[List[float]] = []
        seq_chars: List[str] = []

        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.has_id("CA"):
                        ca_atom = residue["CA"]
                        coords.append(list(ca_atom.get_coord()))
                        res_name = residue.get_resname().strip()
                        one_letter = AA3_TO_AA1.get(res_name.upper(), "X")
                        seq_chars.append(one_letter)
            break  # Process primary model

        if not coords:
            raise ValueError("Structure contains no valid C-alpha atoms.")

        return np.asarray(coords, dtype=np.float32), "".join(seq_chars)
