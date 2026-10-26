"""Unit tests for fast non-dominated sorting, crowding distance, and Pareto metrics."""

import pytest
import numpy as np

from evanthea.data.schemas import VariantRecord
from evanthea.evolution.individual import Individual
from evanthea.evolution.pareto import (
    calculate_crowding_distance,
    compute_hypervolume,
    compute_spacing_metric,
    crowded_comparison,
    extract_pareto_front,
    fast_non_dominated_sort,
)


def _ind(v_id: str, obj_vec: list[float], rank: int | None = None, cd: float = 0.0) -> Individual:
    v = VariantRecord(variant_id=v_id, protein_id="P", mutations=[], mutated_sequence="A", is_wildtype=True)
    return Individual(variant=v, objectives_vector=obj_vec, rank=rank, crowding_distance=cd)


def test_fast_non_dominated_sort():
    # 2D points to maximize:
    # Front 0 (non-dominated): (1, 5), (3, 3), (5, 1)
    # Front 1 (dominated by front 0): (1, 3), (2, 2)
    # Front 2 (dominated by front 1): (0, 0)
    p1 = _ind("p1", [1.0, 5.0])
    p2 = _ind("p2", [3.0, 3.0])
    p3 = _ind("p3", [5.0, 1.0])
    p4 = _ind("p4", [1.0, 3.0])
    p5 = _ind("p5", [2.0, 2.0])
    p6 = _ind("p6", [0.0, 0.0])

    fronts = fast_non_dominated_sort([p1, p2, p3, p4, p5, p6])
    assert len(fronts) == 3

    f0_ids = {ind.id for ind in fronts[0]}
    assert f0_ids == {"p1", "p2", "p3"}
    for ind in fronts[0]:
        assert ind.rank == 0

    f1_ids = {ind.id for ind in fronts[1]}
    assert f1_ids == {"p4", "p5"}
    for ind in fronts[1]:
        assert ind.rank == 1

    f2_ids = {ind.id for ind in fronts[2]}
    assert f2_ids == {"p6"}
    for ind in fronts[2]:
        assert ind.rank == 2


def test_fast_non_dominated_sort_empty_and_single():
    assert fast_non_dominated_sort([]) == []

    p = _ind("p", [1.0, 2.0])
    fronts = fast_non_dominated_sort([p])
    assert len(fronts) == 1
    assert fronts[0][0].id == "p"
    assert fronts[0][0].rank == 0


def test_calculate_crowding_distance():
    p1 = _ind("p1", [1.0, 5.0])
    p2 = _ind("p2", [2.0, 4.0])
    p3 = _ind("p3", [4.0, 2.0])
    p4 = _ind("p4", [5.0, 1.0])

    front = [p1, p2, p3, p4]
    calculate_crowding_distance(front)

    # Boundaries in obj 0 and obj 1 must be infinite
    assert p1.crowding_distance == float("inf")
    assert p4.crowding_distance == float("inf")

    # Interior points must have finite non-negative crowding distances
    assert 0.0 < p2.crowding_distance < float("inf")
    assert 0.0 < p3.crowding_distance < float("inf")


def test_crowded_comparison():
    # Different ranks: lower rank wins (-1)
    ind_r0 = _ind("r0", [1.0, 2.0], rank=0, cd=1.0)
    ind_r1 = _ind("r1", [3.0, 4.0], rank=1, cd=5.0)
    assert crowded_comparison(ind_r0, ind_r1) == -1
    assert crowded_comparison(ind_r1, ind_r0) == 1

    # Same rank: larger crowding distance wins (-1)
    ind_dense = _ind("dense", [2.0, 2.0], rank=0, cd=0.5)
    ind_sparse = _ind("sparse", [2.0, 2.0], rank=0, cd=2.5)
    assert crowded_comparison(ind_sparse, ind_dense) == -1
    assert crowded_comparison(ind_dense, ind_sparse) == 1

    # Identical rank and cd
    ind_tie = _ind("tie", [2.0, 2.0], rank=0, cd=0.5)
    assert crowded_comparison(ind_dense, ind_tie) == 0


def test_extract_pareto_front():
    p1 = _ind("p1", [2.0, 5.0])
    p2 = _ind("p2", [1.0, 1.0])
    front = extract_pareto_front([p1, p2])
    assert len(front) == 1
    assert front[0].id == "p1"


def test_compute_hypervolume_2d_and_3d():
    # 2D exact test
    # Point at (2, 2), reference point (0, 0) -> area should be 2 * 2 = 4.0
    p = _ind("p", [2.0, 2.0])
    hv = compute_hypervolume([p], reference_point=[0.0, 0.0])
    assert hv == pytest.approx(4.0)

    # Point below reference point -> 0.0
    p_bad = _ind("p_bad", [-1.0, -1.0])
    assert compute_hypervolume([p_bad], reference_point=[0.0, 0.0]) == 0.0

    # 3D Monte Carlo test
    # Point at (2, 2, 2), reference point (0, 0, 0) -> volume should be ~8.0
    p_3d = _ind("p_3d", [2.0, 2.0, 2.0])
    hv_3d = compute_hypervolume([p_3d], reference_point=[0.0, 0.0, 0.0], n_mc_samples=20000)
    assert hv_3d == pytest.approx(8.0, rel=0.05)


def test_compute_spacing_metric():
    # Front with 3 evenly spaced points
    p1 = _ind("p1", [0.0, 2.0])
    p2 = _ind("p2", [1.0, 1.0])
    p3 = _ind("p3", [2.0, 0.0])
    spacing = compute_spacing_metric([p1, p2, p3])
    assert spacing >= 0.0

    # Single point front has 0 spacing
    assert compute_spacing_metric([p1]) == 0.0
