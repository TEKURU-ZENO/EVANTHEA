"""Unit tests for PersistentMemoryStore SQLite persistence layer."""

from pathlib import Path
import pytest

from evanthea.data.schemas import Mutation
from evanthea.memory.schema import MemoryRecord
from evanthea.memory.storage import PersistentMemoryStore


def test_persistent_memory_store_roundtrip(tmp_path: Path):
    db_file = tmp_path / "memory.db"
    store = PersistentMemoryStore(db_file)

    assert store.count_variants() == 0
    assert store.has_sequence("h_test") is False

    m = Mutation(ref_residue="M", position_1based=1, mut_residue="A")
    rec = MemoryRecord(
        sequence_hash="h_test",
        variant_id="var_1",
        sequence="ASKGE",
        mutations=[m],
        fitness_values={"fit1": 3.14, "fit2": -1.5},
        objectives_vector=[3.14, -1.5],
        generation=1,
        parent_ids=["h_wt"],
        metadata={"operator": "esm2_point"},
    )

    store.save_record(rec)

    assert store.count_variants() == 1
    assert store.has_sequence("h_test") is True

    loaded = store.get_record("h_test")
    assert loaded is not None
    assert loaded.sequence_hash == "h_test"
    assert loaded.variant_id == "var_1"
    assert loaded.sequence == "ASKGE"
    assert len(loaded.mutations) == 1
    assert loaded.mutations[0].to_string() == "M1A"
    assert loaded.fitness_values["fit1"] == pytest.approx(3.14)
    assert loaded.fitness_values["fit2"] == pytest.approx(-1.5)
    assert loaded.objectives_vector == [3.14, -1.5]
    assert loaded.generation == 1
    assert loaded.parent_ids == ["h_wt"]

    # Lineage queries
    assert store.get_parents("h_test") == ["h_wt"]
    assert store.get_children("h_wt") == ["h_test"]

    store.close()


def test_persistent_memory_store_update_existing(tmp_path: Path):
    db_file = tmp_path / "memory_update.db"
    with PersistentMemoryStore(db_file) as store:
        rec1 = MemoryRecord(
            sequence_hash="h1",
            variant_id="v1",
            sequence="MSKGE",
            fitness_values={"score": 1.0},
        )
        store.save_record(rec1)
        assert store.count_variants() == 1

        # Update with new fitness values and generation
        rec2 = MemoryRecord(
            sequence_hash="h1",
            variant_id="v1_updated",
            sequence="MSKGE",
            fitness_values={"score": 2.5, "extra": 4.0},
            generation=3,
        )
        store.save_record(rec2)
        assert store.count_variants() == 1

        loaded = store.get_record("h1")
        assert loaded.fitness_values["score"] == 2.5
        assert loaded.fitness_values["extra"] == 4.0
        assert loaded.generation == 3


def test_persistent_memory_store_get_all_records(tmp_path: Path):
    db_file = tmp_path / "memory_all.db"
    with PersistentMemoryStore(db_file) as store:
        r1 = MemoryRecord(sequence_hash="h1", variant_id="v1", sequence="A")
        r2 = MemoryRecord(sequence_hash="h2", variant_id="v2", sequence="C")
        store.save_record(r1)
        store.save_record(r2)

        all_recs = store.get_all_records()
        assert len(all_recs) == 2
        assert {r.sequence_hash for r in all_recs} == {"h1", "h2"}
