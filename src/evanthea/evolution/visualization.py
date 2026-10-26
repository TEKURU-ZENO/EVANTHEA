"""Visualization utilities and trajectory renderers for evolutionary experiments."""

from typing import Any, Dict, List, Optional
import numpy as np

from evanthea.evolution.history import EvolutionHistory
from evanthea.evolution.individual import Individual


def render_generation_trajectory_table(
    history: EvolutionHistory,
    objective_name: str,
) -> str:
    """Renders a GitHub-flavored Markdown table showing generation-by-generation progress for an objective."""
    lines = [
        f"| Generation | Mean {objective_name} | Median | Min | Max | Std | Diversity | Pareto Size | Hypervolume |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for g in history.generations:
        metrics = g.objective_metrics.get(objective_name, {})
        mean_v = f"{metrics.get('mean', 0.0):.3f}" if metrics else "N/A"
        med_v = f"{metrics.get('median', 0.0):.3f}" if metrics else "N/A"
        min_v = f"{metrics.get('min', 0.0):.3f}" if metrics else "N/A"
        max_v = f"{metrics.get('max', 0.0):.3f}" if metrics else "N/A"
        std_v = f"{metrics.get('std', 0.0):.3f}" if metrics else "N/A"
        div_v = f"{g.pairwise_diversity:.2f}"
        p_size = str(g.pareto_front_size)
        hv_v = f"{g.hypervolume:.3f}" if g.hypervolume is not None else "N/A"

        lines.append(
            f"| **Gen {g.generation}** | {mean_v} | {med_v} | {min_v} | {max_v} | {std_v} | {div_v} | {p_size} | {hv_v} |"
        )

    return "\n".join(lines)


def render_pareto_summary_table(front: List[Individual]) -> str:
    """Renders a Markdown summary of individuals in the final Pareto front."""
    if not front:
        return "No Pareto candidates discovered."

    obj_names = list(front[0].fitness_values.keys())
    headers = ["Variant ID", "Mutations", "Rank", "Crowding Dist"] + [f"Score: {name}" for name in obj_names]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for ind in front:
        cd_str = "inf" if ind.crowding_distance == float("inf") else f"{ind.crowding_distance:.3f}"
        row = [
            f"`{ind.id[:12]}`",
            ind.mutation_string,
            str(ind.rank),
            cd_str,
        ]
        for name in obj_names:
            score = ind.fitness_values.get(name, 0.0)
            row.append(f"{score:.3f}")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def extract_plot_data(history: EvolutionHistory) -> Dict[str, Any]:
    """Extracts numerical arrays for external charting libraries (Matplotlib, Plotly, Dashboard)."""
    gens = [g.generation for g in history.generations]
    data: Dict[str, Any] = {
        "generations": gens,
        "pairwise_diversity": [g.pairwise_diversity for g in history.generations],
        "hypervolume": [g.hypervolume for g in history.generations],
        "pareto_front_size": [g.pareto_front_size for g in history.generations],
        "objectives": {},
    }

    if history.generations and history.generations[0].objective_metrics:
        for obj_name in history.generations[0].objective_metrics.keys():
            data["objectives"][obj_name] = {
                "mean": [g.objective_metrics.get(obj_name, {}).get("mean", 0.0) for g in history.generations],
                "max": [g.objective_metrics.get(obj_name, {}).get("max", 0.0) for g in history.generations],
                "min": [g.objective_metrics.get(obj_name, {}).get("min", 0.0) for g in history.generations],
            }

    return data
