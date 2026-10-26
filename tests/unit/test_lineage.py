"""Unit tests for LineageTracker and trajectory tracing."""

import json
from pathlib import Path
import pytest

from evanthea.data.schemas import Mutation, VariantRecord
from evanthea.evolution.individual import Individual
from evanthea.evolution.lineage import LineageTracker


def _make_ind(v_id: str, parent_ids: list[str], fit: float = 1.0) -> Individual:
    v = VariantRecord(variant_id=v_id, protein_id="P", mutations=[], mutated_sequence="A", is_wildtype=True)
    return Individual(
        variant=v,
        parent_ids=parent_ids,
        fitness_values={"fit": fit},
        is_evaluated=True,
    )


def test_lineage_tracker_record_and_trajectory(tmp_path: Path):
    tracker = LineageTracker()

    root = _make_ind("root", [])
    child1 = _make_ind("c1", ["root"])
    child2 = _make_ind("c2", ["c1"])

    tracker.record_individual(root, operator="seed")
    tracker.record_individual(child1, operator="mutation")
    tracker.record_individual(child2, operator="mutation")

    # Trajectory from child2
    traj = tracker.get_trajectory("c2")
    assert len(traj) == 3
    assert [node.variant_id for node in traj] == ["root", "c1", "c2"]
    assert traj[0].operator_applied == "seed"

    # Export graph
    graph = tracker.export_graph()
    assert graph["total_nodes"] == 3
    assert graph["total_edges"] == 2

    # Save to disk
    json_file = tmp_path / "lineage.json"
    tracker.save_json(json_file)
    assert json_file.exists()
    with open(json_file, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["total_nodes"] == 3
