"""EVANTHEA closed-loop evolutionary optimization framework (Single-Objective GA, Multi-Objective NSGA-II, Baselines)."""

from evanthea.evolution.algorithms.base import EvolutionaryAlgorithm
from evanthea.evolution.algorithms.baselines import (
    ESMGuidedSearch,
    GreedyHillClimber,
    RandomSearchOptimizer,
)
from evanthea.evolution.algorithms.genetic_algorithm import SingleObjectiveGA
from evanthea.evolution.algorithms.nsga2 import NSGA2Optimizer
from evanthea.evolution.crossover import (
    CrossoverOperator,
    HomologousMutationCrossover,
    SinglePointCrossover,
)
from evanthea.evolution.history import EvolutionHistory, GenerationRecord
from evanthea.evolution.individual import Individual
from evanthea.evolution.lineage import LineageNode, LineageTracker
from evanthea.evolution.objectives import (
    MultiObjectiveEvaluator,
    Objective,
    ObjectiveDirection,
)
from evanthea.evolution.pareto import (
    calculate_crowding_distance,
    compute_hypervolume,
    compute_spacing_metric,
    crowded_comparison,
    extract_pareto_front,
    fast_non_dominated_sort,
)
from evanthea.evolution.population import Population
from evanthea.evolution.selection import (
    ParentSelector,
    RandomSelector,
    RouletteWheelSelector,
    TournamentSelector,
)
from evanthea.evolution.visualization import (
    extract_plot_data,
    render_generation_trajectory_table,
    render_pareto_summary_table,
)

__all__ = [
    "Individual",
    "Population",
    "Objective",
    "ObjectiveDirection",
    "MultiObjectiveEvaluator",
    "ParentSelector",
    "TournamentSelector",
    "RouletteWheelSelector",
    "RandomSelector",
    "CrossoverOperator",
    "SinglePointCrossover",
    "HomologousMutationCrossover",
    "LineageNode",
    "LineageTracker",
    "GenerationRecord",
    "EvolutionHistory",
    "fast_non_dominated_sort",
    "calculate_crowding_distance",
    "crowded_comparison",
    "extract_pareto_front",
    "compute_hypervolume",
    "compute_spacing_metric",
    "EvolutionaryAlgorithm",
    "SingleObjectiveGA",
    "NSGA2Optimizer",
    "RandomSearchOptimizer",
    "GreedyHillClimber",
    "ESMGuidedSearch",
    "render_generation_trajectory_table",
    "render_pareto_summary_table",
    "extract_plot_data",
]
