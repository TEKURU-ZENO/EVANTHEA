"""Physical, biological, and engineering constants for the EVANTHEA framework."""

from typing import Final, Mapping, Set

# Standard 20 canonical amino acids (IUPAC single-letter codes)
STANDARD_AMINO_ACIDS: Final[Set[str]] = {
    "A", "C", "D", "E", "F",
    "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R",
    "S", "T", "V", "W", "Y"
}

# 1-letter to 3-letter amino acid code mapping
AA_1_TO_3: Final[Mapping[str, str]] = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR"
}

# 3-letter to 1-letter amino acid code mapping
AA_3_TO_1: Final[Mapping[str, str]] = {v: k for k, v in AA_1_TO_3.items()}

# Average monoisotopic molecular weights (Da) of amino acid residues in polypeptides
AMINO_ACID_WEIGHTS: Final[Mapping[str, float]] = {
    "A": 71.0788,  "C": 103.1388, "D": 115.0886, "E": 129.1155, "F": 147.1766,
    "G": 57.0519,  "H": 137.1411, "I": 113.1594, "K": 128.1741, "L": 113.1594,
    "M": 131.1926, "N": 114.1038, "P": 97.1167,  "Q": 128.1307, "R": 156.1875,
    "S": 87.0782,  "T": 101.1051, "V": 99.1326,  "W": 186.2132, "Y": 163.1760
}

# Default Pretrained ESM-2 models
DEFAULT_ESM2_150M_CHECKPOINT: Final[str] = "facebook/esm2_t30_150M_UR50D"
DEFAULT_ESM2_35M_CHECKPOINT: Final[str] = "facebook/esm2_t12_35M_UR50D"
DEFAULT_ESM2_650M_CHECKPOINT: Final[str] = "facebook/esm2_t33_650M_UR50D"

# Default maximum context length for ESM-2
ESM2_MAX_SEQUENCE_LENGTH: Final[int] = 1022

# Coordinate conventions
ONE_BASED: Final[str] = "1-based"
ZERO_BASED: Final[str] = "0-based"

# FireProtDB / ProTherm thermodynamic conventions
DDG_SIGN_STABILIZING_NEGATIVE: Final[str] = "ddg_negative_stabilizing"  # dG_mut - dG_wt (where dG is unfolding or folding)
DDG_SIGN_STABILIZING_POSITIVE: Final[str] = "ddg_positive_stabilizing"
