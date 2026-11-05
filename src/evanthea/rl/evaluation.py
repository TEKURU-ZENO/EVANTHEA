"""Policy evaluation and trajectory analysis for protein design RL agents."""

from typing import List, Optional
import numpy as np

from evanthea.evolution.individual import Individual
from evanthea.evolution.pareto import fast_non_dominated_sort
from evanthea.rl.agent import RLAgent
from evanthea.rl.environment import ProteinDesignEnv
from evanthea.rl.schema import PolicyEvaluationSummary, Trajectory


def evaluate_policy(
    agent: RLAgent,
    env: ProteinDesignEnv,
    n_episodes: int = 10,
    deterministic: bool = True,
) -> PolicyEvaluationSummary:
    """Evaluates an RL agent over multiple independent episodes and aggregates metrics."""
    returns: List[float] = []
    final_fitnesses: List[float] = []
    mutation_counts: List[int] = []
    unique_sequences = set()
    individuals: List[Individual] = []

    for _ in range(n_episodes):
        traj = agent.rollout_episode(env, deterministic=deterministic)
        returns.append(traj.total_reward)
        final_fitnesses.append(traj.final_predicted_fitness)
        mutation_counts.append(len(traj.mutations))
        unique_sequences.add(traj.final_sequence)

        # Build Individual for Pareto sorting
        ind = Individual(
            variant=agent.propose_candidates(env, n_candidates=1, deterministic=deterministic)[0],
            fitness_values={
                "predicted_fitness": traj.final_predicted_fitness,
                "esm2_plausibility": traj.final_esm_score,
            },
            objectives_vector=[traj.final_predicted_fitness, traj.final_esm_score],
            is_evaluated=True,
        )
        individuals.append(ind)

    # Compute Pareto front size
    fronts = fast_non_dominated_sort(individuals)
    p_size = len(fronts[0]) if fronts else 0

    return PolicyEvaluationSummary(
        episodes=n_episodes,
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        best_fitness=float(np.max(final_fitnesses)),
        mean_fitness=float(np.mean(final_fitnesses)),
        mean_mutations=float(np.mean(mutation_counts)),
        unique_variants=len(unique_sequences),
        pareto_front_size=p_size,
    )
