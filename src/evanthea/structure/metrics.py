"""Biophysical and geometric structural metrics: Kabsch RMSD, Steric Clashes, Contact Maps, and Shrake-Rupley SASA."""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from Bio.PDB import Structure

from evanthea.structure.schema import ClashRecord

# Standard van der Waals atomic radii in Angstroms (Bondi 1964)
VDW_RADII: Dict[str, float] = {
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "H": 1.20,
    "P": 1.80,
    "F": 1.47,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
}
DEFAULT_VDW = 1.70

# Hydrophobic canonical amino acid 3-letter codes
HYDROPHOBIC_RESIDUES: Set[str] = {"ALA", "VAL", "LEU", "ILE", "PRO", "PHE", "MET", "TRP"}


def compute_rmsd(coords_ref: np.ndarray, coords_mut: np.ndarray) -> float:
    """Computes coordinate Root Mean Square Deviation (RMSD) after optimal Kabsch rigid-body superposition.

    Args:
        coords_ref: Reference coordinates of shape (N, 3).
        coords_mut: Variant coordinates of shape (N, 3).

    Returns:
        Superposed RMSD in Angstroms.
    """
    p = np.asarray(coords_ref, dtype=np.float64)
    q = np.asarray(coords_mut, dtype=np.float64)

    if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"Coordinate shape mismatch: ref {p.shape} vs mut {q.shape}")
    n = p.shape[0]
    if n == 0:
        return 0.0

    # 1. Center both point sets at origin
    p_center = p - np.mean(p, axis=0)
    q_center = q - np.mean(q, axis=0)

    # 2. Covariance matrix H = P^T * Q
    h = p_center.T @ q_center

    # 3. Singular Value Decomposition
    u, s, vt = np.linalg.svd(h)

    # 4. Optimal rotation matrix R = V * U^T
    v = vt.T
    d = np.linalg.det(v @ u.T)
    # Right-handed coordinate system reflection check
    if d < 0:
        v[:, -1] *= -1.0
    rot = v @ u.T

    # 5. Apply rotation and calculate Euclidean RMSD
    p_rotated = p_center @ rot.T
    diff = p_rotated - q_center
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    return float(rmsd)


def detect_steric_clashes(
    structure: Structure.Structure,
    clash_tolerance: float = 0.4,
    min_sequence_separation: int = 2,
) -> List[ClashRecord]:
    """Detects severe non-bonded atomic steric clashes violating van der Waals clearance.

    Args:
        structure: BioPython Structure object.
        clash_tolerance: Overlap threshold beyond van der Waals sum (default 0.4 A).
        min_sequence_separation: Minimum residue sequence distance for non-bonded
            pairs (default 2, excluding 1-2, 1-3, and 1-4 covalently bonded neighbors).

    Returns:
        List of ClashRecord objects for all non-bonded atom pairs violating tolerance.
    """
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                # Exclude hetero-atoms / water
                if residue.id[0] != " ":
                    continue
                res_id = residue.id[1]
                res_name = residue.get_resname().strip()
                for atom in residue:
                    element = atom.element.strip().upper() if atom.element else atom.get_name()[0].upper()
                    # Skip hydrogens if present
                    if element == "H":
                        continue
                    vdw = VDW_RADII.get(element, DEFAULT_VDW)
                    atoms.append({
                        "name": atom.get_name().strip(),
                        "res_id": res_id,
                        "res_name": res_name,
                        "coord": atom.get_coord(),
                        "vdw": vdw,
                        "element": element,
                    })
        break

    n = len(atoms)
    clashes: List[ClashRecord] = []

    for i in range(n):
        a1 = atoms[i]
        c1 = a1["coord"]
        r1 = a1["vdw"]

        for j in range(i + 1, n):
            a2 = atoms[j]

            # Exclude bonded 1-2, 1-3, and 1-4 pairs within the same residue or adjacent residues
            res_diff = abs(a1["res_id"] - a2["res_id"])
            if res_diff < min_sequence_separation:
                continue

            dist = np.linalg.norm(c1 - a2["coord"])
            cutoff = r1 + a2["vdw"] - clash_tolerance

            if dist < cutoff:
                overlap = float(cutoff - dist)
                clashes.append(
                    ClashRecord(
                        atom1_name=a1["name"],
                        atom2_name=a2["name"],
                        res1_id=a1["res_id"],
                        res2_id=a2["res_id"],
                        res1_name=a1["res_name"],
                        res2_name=a2["res_name"],
                        distance=float(dist),
                        overlap=overlap,
                    )
                )

    return clashes


def compute_contact_map(
    structure: Structure.Structure,
    distance_cutoff: float = 8.0,
    min_sequence_separation: int = 4,
) -> np.ndarray:
    """Constructs binary residue-residue contact map using CB atoms (CA for Glycine).

    Args:
        structure: BioPython Structure object.
        distance_cutoff: Distance threshold in Angstroms (default 8.0 A).
        min_sequence_separation: Minimum sequence distance |i - j| (default 4 residues).

    Returns:
        Boolean contact matrix of shape (L, L).
    """
    rep_coords: List[np.ndarray] = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                # Use CB for all standard residues, CA for GLY
                if residue.has_id("CB"):
                    rep_coords.append(residue["CB"].get_coord())
                elif residue.has_id("CA"):
                    rep_coords.append(residue["CA"].get_coord())
        break

    n = len(rep_coords)
    if n == 0:
        return np.zeros((0, 0), dtype=bool)

    coords_arr = np.asarray(rep_coords, dtype=np.float32)
    # Compute pairwise Euclidean distance matrix
    diff = coords_arr[:, np.newaxis, :] - coords_arr[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    # Construct separation mask
    idx = np.arange(n)
    sep_matrix = np.abs(idx[:, np.newaxis] - idx[np.newaxis, :])

    contacts = (dist_matrix <= distance_cutoff) & (sep_matrix >= min_sequence_separation)
    return contacts


def compute_contact_preservation(ref_contacts: np.ndarray, mut_contacts: np.ndarray) -> float:
    """Calculates the fraction of wild-type tertiary contacts preserved in the mutant structure."""
    if ref_contacts.shape != mut_contacts.shape:
        raise ValueError(f"Contact map shape mismatch: {ref_contacts.shape} vs {mut_contacts.shape}")

    total_ref_contacts = int(np.sum(ref_contacts))
    if total_ref_contacts == 0:
        return 1.0

    # Preserved contacts: contacts active in both reference and mutant
    preserved = int(np.sum(ref_contacts & mut_contacts))
    return float(preserved / total_ref_contacts)


def compute_sasa(
    structure: Structure.Structure,
    probe_radius: float = 1.4,
    n_points: int = 96,
) -> Tuple[float, float]:
    """Calculates total and hydrophobic Solvent Accessible Surface Area using Shrake-Rupley numerical integration.

    Args:
        structure: BioPython Structure object.
        probe_radius: Radius of rolling solvent probe in Angstroms (default 1.4 A water probe).
        n_points: Number of Fibonacci sphere test points per atom (default 96).

    Returns:
        Tuple of (total_sasa_A2, hydrophobic_sasa_A2).
    """
    # 1. Fibonacci golden spiral sphere points on unit sphere
    indices = np.arange(0, n_points, dtype=np.float32) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / n_points)
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    theta = 2.0 * np.pi * indices / golden_ratio
    unit_sphere = np.stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ], axis=-1)

    # 2. Extract atomic coordinates and radii
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                res_name = residue.get_resname().strip()
                is_hydrophobic = res_name in HYDROPHOBIC_RESIDUES
                for atom in residue:
                    elem = atom.element.strip().upper() if atom.element else atom.get_name()[0].upper()
                    if elem == "H":
                        continue
                    r_vdw = VDW_RADII.get(elem, DEFAULT_VDW)
                    atoms.append({
                        "coord": atom.get_coord(),
                        "radius": r_vdw + probe_radius,
                        "is_hydrophobic": is_hydrophobic,
                    })
        break

    n = len(atoms)
    if n == 0:
        return 0.0, 0.0

    coords = np.array([a["coord"] for a in atoms], dtype=np.float32)
    radii = np.array([a["radius"] for a in atoms], dtype=np.float32)
    is_hydro = np.array([a["is_hydrophobic"] for a in atoms], dtype=bool)

    # Area per sphere point: 4 * pi * R^2 / N
    point_areas = 4.0 * np.pi * (radii ** 2) / float(n_points)

    total_sasa = 0.0
    hydrophobic_sasa = 0.0

    # 3. Test accessibility of each sphere point against neighbor atom spheres
    for i in range(n):
        center_i = coords[i]
        r_i = radii[i]
        test_points = center_i + r_i * unit_sphere

        # Find neighboring atoms that could occlude atom i
        dists_to_neighbors = np.linalg.norm(coords - center_i, axis=1)
        # Neighbor can occlude if dist <= r_i + r_neighbor
        neighbor_indices = np.where((dists_to_neighbors > 1e-4) & (dists_to_neighbors < r_i + radii))[0]

        if len(neighbor_indices) == 0:
            # Fully exposed atom
            accessible_count = n_points
        else:
            neighbor_coords = coords[neighbor_indices]
            neighbor_radii = radii[neighbor_indices]

            # Vectorized occlusion test
            # diff shape: (n_points, n_neighbors, 3)
            diff = test_points[:, np.newaxis, :] - neighbor_coords[np.newaxis, :, :]
            sq_dists = np.sum(diff ** 2, axis=-1)
            # Occluded if distance from neighbor center < neighbor radius
            occluded = np.any(sq_dists < (neighbor_radii ** 2)[np.newaxis, :], axis=1)
            accessible_count = int(np.sum(~occluded))

        atom_sasa = float(accessible_count * point_areas[i])
        total_sasa += atom_sasa
        if is_hydro[i]:
            hydrophobic_sasa += atom_sasa

    return float(total_sasa), float(hydrophobic_sasa)
