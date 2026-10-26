"""Lineage DAG tracker recording ancestry, genetic operations, and evolutionary trajectories."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from evanthea.evolution.individual import Individual


class LineageNode(BaseModel):
    """Represents a node in the evolutionary ancestry DAG."""
    variant_id: str
    parent_ids: List[str] = Field(default_factory=list)
    generation: int
    mutation_string: str
    fitness_values: Dict[str, float] = Field(default_factory=dict)
    operator_applied: Optional[str] = None


class LineageTracker:
    """Maintains a directed acyclic graph (DAG) of variant ancestry throughout evolutionary search."""

    def __init__(self) -> None:
        self.nodes: Dict[str, LineageNode] = {}

    def record_individual(
        self,
        individual: Individual,
        operator: Optional[str] = None,
    ) -> None:
        """Records an individual in the lineage graph."""
        v_id = individual.id
        if v_id in self.nodes:
            # Already recorded, update fitness if newly evaluated
            if individual.is_evaluated and not self.nodes[v_id].fitness_values:
                self.nodes[v_id].fitness_values = dict(individual.fitness_values)
            return

        node = LineageNode(
            variant_id=v_id,
            parent_ids=list(individual.parent_ids),
            generation=individual.generation,
            mutation_string=individual.mutation_string,
            fitness_values=dict(individual.fitness_values) if individual.is_evaluated else {},
            operator_applied=operator or individual.variant.metadata.get("operator"),
        )
        self.nodes[v_id] = node

    def get_trajectory(self, variant_id: str) -> List[LineageNode]:
        """Walks backwards along primary parental lineage to root ancestor."""
        trajectory: List[LineageNode] = []
        curr_id: Optional[str] = variant_id

        visited = set()
        while curr_id and curr_id in self.nodes and curr_id not in visited:
            visited.add(curr_id)
            node = self.nodes[curr_id]
            trajectory.append(node)
            curr_id = node.parent_ids[0] if node.parent_ids else None

        trajectory.reverse()  # Root ancestor -> target variant
        return trajectory

    def export_graph(self) -> Dict[str, Any]:
        """Exports graph in node-link format for network analysis and visualization."""
        nodes_list = []
        edges_list = []

        for node in self.nodes.values():
            nodes_list.append(node.model_dump())
            for p_id in node.parent_ids:
                if p_id in self.nodes:
                    edges_list.append({"source": p_id, "target": node.variant_id})

        return {
            "total_nodes": len(nodes_list),
            "total_edges": len(edges_list),
            "nodes": nodes_list,
            "edges": edges_list,
        }

    def save_json(self, path: Path | str) -> None:
        """Serializes lineage graph to disk as JSON."""
        data = self.export_graph()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
