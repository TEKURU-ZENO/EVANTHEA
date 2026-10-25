"""Unit tests for leakage-resistant dataset splits and verification assertions."""

import pytest
from evanthea.data.schemas import Mutation, VariantRecord
from evanthea.data.splits import (
    create_mutation_order_split,
    create_position_held_out_split,
    create_random_split,
    verify_split_disjointness,
)
from evanthea.exceptions import DataLeakageError


def _mock_variant(vid: str, mutations_list: list[tuple[str, int, str]], seq_len: int = 10) -> VariantRecord:
    mutations = [Mutation(ref_residue=r, position_1based=p, mut_residue=m) for r, p, m in mutations_list]
    seq = list("A" * seq_len)
    for m in mutations:
        seq[m.position_0based] = m.mut_residue
    return VariantRecord(
        variant_id=vid,
        protein_id="P_TEST",
        mutations=mutations,
        mutated_sequence="".join(seq),
        is_wildtype=len(mutations) == 0,
        num_mutations=len(mutations),
    )


def test_verify_split_disjointness_catches_leakage():
    """Throws DataLeakageError if an ID is present in multiple splits."""
    with pytest.raises(DataLeakageError):
        verify_split_disjointness(["v1", "v2"], ["v2", "v3"], ["v4"])
    with pytest.raises(DataLeakageError):
        verify_split_disjointness(["v1", "v2"], ["v3"], ["v1", "v4"])


def test_create_random_split():
    """Generates disjoint random partitions."""
    variants = [_mock_variant(f"v_{i}", [("A", 1, "C")]) for i in range(100)]
    manifest = create_random_split("TEST_DMS", variants, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)

    assert manifest.split_strategy == "random"
    assert len(manifest.train_variant_ids) == 80
    assert len(manifest.val_variant_ids) == 10
    assert len(manifest.test_variant_ids) == 10
    assert manifest.leakage_verification["cross_split_id_overlap"] == 0


def test_create_position_held_out_split_zero_position_leak():
    """Guarantees test mutation positions never appear in train mutations."""
    # 10 positions total.
    # Variant v1 at pos 1, v2 at pos 2, v3 at pos 3, v4 at pos 8, v5 at pos 9, v6 at pos 10
    variants = [
        _mock_variant(f"v_{pos}", [("A", pos, "C")]) for pos in range(1, 11)
    ]
    manifest = create_position_held_out_split(
        "TEST_DMS", variants, sequence_length=10, test_position_fraction=0.3, val_position_fraction=0.2, seed=42
    )

    assert manifest.split_strategy == "position_held_out"
    assert manifest.leakage_verification["positional_leakage_overlap"] == 0

    # Ensure all test variants only mutate held-out positions
    id_map = {v.variant_id: v for v in variants}
    train_positions = {m.position_1based for vid in manifest.train_variant_ids for m in id_map[vid].mutations}
    test_positions = {m.position_1based for vid in manifest.test_variant_ids for m in id_map[vid].mutations}

    assert train_positions.intersection(test_positions) == set()


def test_create_mutation_order_split():
    """Partitions strictly by mutation degree (single vs double vs triple+)."""
    v_wt = _mock_variant("wt", [])
    v_s1 = _mock_variant("s1", [("A", 1, "C")])
    v_s2 = _mock_variant("s2", [("A", 2, "C")])
    v_d1 = _mock_variant("d1", [("A", 1, "C"), ("A", 2, "D")])
    v_t1 = _mock_variant("t1", [("A", 1, "C"), ("A", 2, "D"), ("A", 3, "E")])

    variants = [v_wt, v_s1, v_s2, v_d1, v_t1]
    manifest = create_mutation_order_split("TEST_DMS", variants, train_max_order=1, val_order=2, test_min_order=3)

    assert manifest.split_strategy == "mutation_order"
    assert set(manifest.train_variant_ids) == {"wt", "s1", "s2"}
    assert set(manifest.val_variant_ids) == {"d1"}
    assert set(manifest.test_variant_ids) == {"t1"}
