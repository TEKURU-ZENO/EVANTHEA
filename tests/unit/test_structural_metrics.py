"""Unit tests for biophysical metrics (Kabsch RMSD, Steric Clashes, Contact Maps, SASA)."""

import numpy as np
import pytest

from evanthea.structure.metrics import (
    compute_contact_map,
    compute_contact_preservation,
    compute_rmsd,
    compute_sasa,
    detect_steric_clashes,
)
from evanthea.structure.predictor import MockStructurePredictor, ReferencePDBLoader


def test_kabsch_rmsd():
    # 1. Identical coordinates -> 0.0 RMSD
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float64)

    assert pytest.approx(compute_rmsd(coords, coords), abs=1e-6) == 0.0

    # 2. Translated and rotated coordinates -> should be 0.0 RMSD after Kabsch alignment
    theta = np.radians(45.0)
    rot_z = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    transformed = coords @ rot_z.T + np.array([10.0, -5.0, 2.0])
    assert pytest.approx(compute_rmsd(coords, transformed), abs=1e-5) == 0.0

    # 3. Perturbed coordinate
    perturbed = coords.copy()
    perturbed[0, 0] += 2.0
    rmsd = compute_rmsd(coords, perturbed)
    assert rmsd > 0.0


def test_contact_map_and_preservation():
    seq = "MSKGEELFTG"
    pred = MockStructurePredictor().predict_structure(seq)
    loader = ReferencePDBLoader()
    struct = loader.load_from_text(pred.pdb_text)

    cmap = compute_contact_map(struct, distance_cutoff=10.0, min_sequence_separation=2)
    assert cmap.shape == (len(seq), len(seq))
    # Diagonal must be False (distance = 0 but separation = 0 < min_sep)
    assert not np.any(np.diag(cmap))

    # Self preservation = 1.0
    pres = compute_contact_preservation(cmap, cmap)
    assert pres == 1.0


def test_clash_detection():
    # Standard mock structure has canonical spacing and should have 0 or minimal clashes
    seq = "MSKGEELFTG"
    pred = MockStructurePredictor().predict_structure(seq)
    loader = ReferencePDBLoader()
    struct = loader.load_from_text(pred.pdb_text)

    clashes = detect_steric_clashes(struct, clash_tolerance=0.4)
    assert isinstance(clashes, list)


def test_shrake_rupley_sasa():
    seq = "MSKGEELFTG"
    pred = MockStructurePredictor().predict_structure(seq)
    loader = ReferencePDBLoader()
    struct = loader.load_from_text(pred.pdb_text)

    total_sasa, hydro_sasa = compute_sasa(struct, probe_radius=1.4, n_points=48)
    assert total_sasa > 0.0
    assert hydro_sasa >= 0.0
    assert hydro_sasa <= total_sasa
