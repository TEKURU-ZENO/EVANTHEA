"""Leakage-resistant dataset splitting strategies with explicit scientific question documentation."""

import random
from typing import Dict, List, Set, Tuple
from evanthea.data.schemas import DatasetSplitManifest, VariantRecord
from evanthea.exceptions import DataLeakageError


def verify_split_disjointness(train_ids: List[str], val_ids: List[str], test_ids: List[str]) -> None:
    """Verifies that no variant ID appears in more than one partition."""
    set_train = set(train_ids)
    set_val = set(val_ids)
    set_test = set(test_ids)

    train_val_overlap = set_train.intersection(set_val)
    if train_val_overlap:
        raise DataLeakageError(
            f"Leakage detected between train and val splits ({len(train_val_overlap)} variants)",
            leaking_identifiers=list(train_val_overlap)[:10],
        )

    train_test_overlap = set_train.intersection(set_test)
    if train_test_overlap:
        raise DataLeakageError(
            f"Leakage detected between train and test splits ({len(train_test_overlap)} variants)",
            leaking_identifiers=list(train_test_overlap)[:10],
        )

    val_test_overlap = set_val.intersection(set_test)
    if val_test_overlap:
        raise DataLeakageError(
            f"Leakage detected between val and test splits ({len(val_test_overlap)} variants)",
            leaking_identifiers=list(val_test_overlap)[:10],
        )


def create_random_split(
    dataset_id: str,
    variants: List[VariantRecord],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetSplitManifest:
    """Standard random partition across the entire variant distribution.

    Scientific Question:
        Can the model accurately predict variant fitness when drawn from the identical
        underlying assay distribution? (Measures in-distribution interpolation).
    """
    if not (0.999 <= train_ratio + val_ratio + test_ratio <= 1.001):
        raise ValueError("Split ratios must sum to 1.0.")

    rng = random.Random(seed)
    shuffled = list(variants)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_vars = shuffled[:n_train]
    val_vars = shuffled[n_train : n_train + n_val]
    test_vars = shuffled[n_train + n_val :]

    train_ids = [v.variant_id for v in train_vars]
    val_ids = [v.variant_id for v in val_vars]
    test_ids = [v.variant_id for v in test_vars]

    verify_split_disjointness(train_ids, val_ids, test_ids)

    return DatasetSplitManifest(
        dataset_id=dataset_id,
        split_strategy="random",
        scientific_question="In-distribution generalization across the same assay distribution.",
        seed=seed,
        train_variant_ids=train_ids,
        val_variant_ids=val_ids,
        test_variant_ids=test_ids,
        leakage_verification={
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "test_count": len(test_ids),
            "cross_split_id_overlap": 0,
        },
    )


def create_position_held_out_split(
    dataset_id: str,
    variants: List[VariantRecord],
    sequence_length: int,
    test_position_fraction: float = 0.2,
    val_position_fraction: float = 0.1,
    seed: int = 42,
) -> DatasetSplitManifest:
    """Partitions variants such that test residue positions are strictly never seen in training.

    Scientific Question:
        Can the model generalize to mutations involving residue positions absent from training?
        (Tests spatial / positional extrapolation across the protein structure).
    """
    rng = random.Random(seed)
    all_positions = list(range(1, sequence_length + 1))
    rng.shuffle(all_positions)

    n_test_pos = max(1, int(sequence_length * test_position_fraction))
    n_val_pos = max(1, int(sequence_length * val_position_fraction))

    test_positions: Set[int] = set(all_positions[:n_test_pos])
    val_positions: Set[int] = set(all_positions[n_test_pos : n_test_pos + n_val_pos])
    train_positions: Set[int] = set(all_positions[n_test_pos + n_val_pos :])

    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []

    for v in variants:
        if v.is_wildtype or not v.mutations:
            train_ids.append(v.variant_id)
            continue

        var_positions = {m.position_1based for m in v.mutations}

        # If any mutation is in the held-out test positions, assign to test
        if var_positions.intersection(test_positions):
            test_ids.append(v.variant_id)
        # Else if any mutation is in validation positions, assign to val
        elif var_positions.intersection(val_positions):
            val_ids.append(v.variant_id)
        # Else all mutations belong strictly to train positions
        else:
            train_ids.append(v.variant_id)

    verify_split_disjointness(train_ids, val_ids, test_ids)

    # Verify zero positional leakage between train and test
    id_to_var = {v.variant_id: v for v in variants}
    train_used_positions = {m.position_1based for vid in train_ids for m in id_to_var[vid].mutations}
    test_held_out_overlap = train_used_positions.intersection(test_positions)
    if test_held_out_overlap:
        raise DataLeakageError(
            f"Position leakage: train set contains mutations at held-out test positions {sorted(test_held_out_overlap)}"
        )

    return DatasetSplitManifest(
        dataset_id=dataset_id,
        split_strategy="position_held_out",
        scientific_question="Generalization to mutations involving residue positions absent from training.",
        seed=seed,
        train_variant_ids=train_ids,
        val_variant_ids=val_ids,
        test_variant_ids=test_ids,
        leakage_verification={
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "test_count": len(test_ids),
            "held_out_test_positions_count": len(test_positions),
            "held_out_val_positions_count": len(val_positions),
            "train_positions_count": len(train_positions),
            "positional_leakage_overlap": 0,
        },
    )


def create_mutation_order_split(
    dataset_id: str,
    variants: List[VariantRecord],
    train_max_order: int = 1,
    val_order: int = 2,
    test_min_order: int = 3,
    seed: int = 42,
) -> DatasetSplitManifest:
    """Partitions variants by mutation count / combinatorial degree.

    Scientific Question:
        Can the model generalize from lower-order (e.g. single/double) to higher-order (e.g. triple+)
        mutational combinations? (Tests epistatic extrapolation).
    """
    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []

    for v in variants:
        k = v.num_mutations
        if k <= train_max_order:
            train_ids.append(v.variant_id)
        elif k == val_order:
            val_ids.append(v.variant_id)
        elif k >= test_min_order:
            test_ids.append(v.variant_id)

    verify_split_disjointness(train_ids, val_ids, test_ids)

    return DatasetSplitManifest(
        dataset_id=dataset_id,
        split_strategy="mutation_order",
        scientific_question="Generalization from lower-order to higher-order combinations (epistatic extrapolation).",
        seed=seed,
        train_variant_ids=train_ids,
        val_variant_ids=val_ids,
        test_variant_ids=test_ids,
        leakage_verification={
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "test_count": len(test_ids),
            "train_max_order": train_max_order,
            "val_order": val_order,
            "test_min_order": test_min_order,
            "cross_split_id_overlap": 0,
        },
    )
