"""Pareto dominance, fast non-dominated sorting, crowding distance, and multi-objective performance metrics."""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from evanthea.evolution.individual import Individual


def fast_non_dominated_sort(individuals: List[Individual]) -> List[List[Individual]]:
    r"""Partitions a population into ranked Pareto non-dominated fronts using the O(M N^2) algorithm.

    Front 0 contains all Pareto-optimal candidates (Rank 0).
    Front 1 contains individuals dominated only by candidates in Front 0 (Rank 1), etc.

    Args:
        individuals: List of individuals with computed objectives_vector.

    Returns:
        List of fronts [F_0, F_1, ..., F_{k-1}], where each front is a list of Individuals.
    """
    if not individuals:
        return []

    # Map each individual to its dominated set S_p and domination counter n_p
    n_p: Dict[int, int] = {}
    s_p: Dict[int, List[int]] = {}
    fronts: List[List[Individual]] = [[]]

    for i, p in enumerate(individuals):
        s_p[i] = []
        n_p[i] = 0

        for j, q in enumerate(individuals):
            if i == j:
                continue
            if p.dominates(q):
                s_p[i].append(j)
            elif q.dominates(p):
                n_p[i] += 1

        if n_p[i] == 0:
            p.rank = 0
            fronts[0].append(p)

    current_rank = 0
    while len(fronts[current_rank]) > 0:
        next_front: List[Individual] = []
        # Look at indices in current front
        for p in fronts[current_rank]:
            p_idx = individuals.index(p)
            for q_idx in s_p[p_idx]:
                n_p[q_idx] -= 1
                if n_p[q_idx] == 0:
                    q = individuals[q_idx]
                    q.rank = current_rank + 1
                    next_front.append(q)

        current_rank += 1
        fronts.append(next_front)

    # Remove the trailing empty front
    if len(fronts) > 1 and len(fronts[-1]) == 0:
        fronts.pop()

    return fronts


def calculate_crowding_distance(front: List[Individual]) -> None:
    """Calculates NSGA-II crowding distance for all individuals in a single Pareto front.

    Boundary individuals in each objective receive distance = inf.
    Interior individuals receive the normalized distance between their neighboring individuals.
    """
    n = len(front)
    if n == 0:
        return
    if n <= 2:
        for ind in front:
            ind.crowding_distance = float("inf")
        return

    # Reset crowding distances
    for ind in front:
        ind.crowding_distance = 0.0

    if front[0].objectives_vector is None:
        raise ValueError("Cannot compute crowding distance for unevaluated individuals.")

    m_objs = len(front[0].objectives_vector)

    for m in range(m_objs):
        # Sort front ascending by objective m
        front.sort(key=lambda ind: ind.objectives_vector[m])  # type: ignore[index]

        # Boundary individuals receive infinite crowding distance
        front[0].crowding_distance = float("inf")
        front[-1].crowding_distance = float("inf")

        f_min = front[0].objectives_vector[m]  # type: ignore[index]
        f_max = front[-1].objectives_vector[m]  # type: ignore[index]
        obj_range = f_max - f_min

        if obj_range < 1e-9:
            continue

        for i in range(1, n - 1):
            if front[i].crowding_distance != float("inf"):
                prev_val = front[i - 1].objectives_vector[m]  # type: ignore[index]
                next_val = front[i + 1].objectives_vector[m]  # type: ignore[index]
                front[i].crowding_distance += (next_val - prev_val) / obj_range


def crowded_comparison(ind1: Individual, ind2: Individual) -> int:
    """Crowded comparison operator (prec_n) for tournament selection and survival.

    Returns:
        -1 if ind1 is preferred over ind2
        +1 if ind2 is preferred over ind1
         0 if ind1 and ind2 are equivalent
    """
    r1 = ind1.rank if ind1.rank is not None else 999999
    r2 = ind2.rank if ind2.rank is not None else 999999

    # Better (lower) rank wins
    if r1 < r2:
        return -1
    elif r1 > r2:
        return 1

    # Same rank: larger crowding distance wins (less crowded region)
    cd1 = ind1.crowding_distance
    cd2 = ind2.crowding_distance

    if cd1 > cd2 + 1e-9:
        return -1
    elif cd2 > cd1 + 1e-9:
        return 1
    return 0


def extract_pareto_front(individuals: List[Individual]) -> List[Individual]:
    """Returns the non-dominated Pareto front (Rank 0) from an arbitrary list of candidates."""
    fronts = fast_non_dominated_sort(individuals)
    if not fronts or not fronts[0]:
        return []
    calculate_crowding_distance(fronts[0])
    return fronts[0]


def compute_hypervolume(
    front: List[Individual],
    reference_point: Sequence[float],
    n_mc_samples: int = 15000,
    seed: Optional[int] = 42,
) -> float:
    """Computes hypervolume indicator dominated by the Pareto front relative to reference point.

    For 2D, computes exact geometric hypervolume.
    For >= 3D, computes high-precision Monte Carlo volume estimation.
    """
    if not front:
        return 0.0

    ref = np.asarray(reference_point, dtype=np.float64)
    points = np.array([ind.objectives_vector for ind in front], dtype=np.float64)

    # Filter points that strictly dominate the reference point in at least one dimension
    # and are not dominated by the reference point in all dimensions
    valid_points = points[np.all(points >= ref, axis=1)]
    if len(valid_points) == 0:
        return 0.0

    m_dim = valid_points.shape[1]

    if m_dim == 2:
        # Exact 2D hypervolume
        # Sort points by objective 0 ascending
        sorted_idx = np.argsort(valid_points[:, 0])
        sorted_pts = valid_points[sorted_idx]

        # Prune dominated points within 2D set
        filtered = []
        max_y = -np.inf
        for pt in reversed(sorted_pts):
            if pt[1] > max_y:
                filtered.append(pt)
                max_y = pt[1]
        filtered.reverse()
        pts_2d = np.array(filtered)

        hv = 0.0
        for i in range(len(pts_2d)):
            width = pts_2d[i, 0] - (ref[0] if i == 0 else pts_2d[i - 1, 0])
            height = pts_2d[i, 1] - ref[1]
            if width > 0 and height > 0:
                hv += width * height
        return float(hv)

    # Monte Carlo estimation for m >= 3
    upper_bounds = np.max(valid_points, axis=0)
    box_lengths = upper_bounds - ref
    if np.any(box_lengths <= 0):
        return 0.0

    bounding_box_volume = float(np.prod(box_lengths))
    rng = np.random.default_rng(seed)

    # Sample points uniformly in the bounding box [ref, upper_bounds]
    samples = rng.uniform(low=ref, high=upper_bounds, size=(n_mc_samples, m_dim))

    # Check which samples are dominated by at least one point in the front
    # Sample S is dominated by front point P if P >= S in all dimensions
    dominated_count = 0
    for s in samples:
        if np.any(np.all(valid_points >= s, axis=1)):
            dominated_count += 1

    fraction_dominated = dominated_count / n_mc_samples
    return float(fraction_dominated * bounding_box_volume)


def compute_spacing_metric(front: List[Individual]) -> float:
    """Computes the spacing/spread metric measuring uniformity of points across the Pareto front.

    Formula: S = sqrt( 1/(n-1) * sum_i (d_i - d_mean)^2 )
    where d_i = min_{j != i} sum_m |f_m^i - f_m^j|. Lower value indicates more uniform coverage.
    """
    n = len(front)
    if n <= 1:
        return 0.0

    pts = np.array([ind.objectives_vector for ind in front], dtype=np.float64)
    min_dists = []

    for i in range(n):
        diffs = np.abs(pts - pts[i])
        manhattan_dists = np.sum(diffs, axis=1)
        # Exclude self distance (which is 0)
        manhattan_dists[i] = np.inf
        min_dists.append(np.min(manhattan_dists))

    d_arr = np.array(min_dists, dtype=np.float64)
    d_mean = np.mean(d_arr)
    variance = np.sum((d_arr - d_mean) ** 2) / (n - 1)
    return float(np.sqrt(variance))
