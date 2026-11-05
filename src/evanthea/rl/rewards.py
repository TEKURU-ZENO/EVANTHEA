"""Reward formulation and multi-objective scalarization for protein design RL."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field

from evanthea.data.schemas import Mutation, VariantRecord
from evanthea.evolution.individual import Individual
from evanthea.evolution.objectives import MultiObjectiveEvaluator
from evanthea.memory.manager import EvolutionaryMemory


class RewardWeights(BaseModel):
    """Weight coefficients for balancing multi-objective components in scalar RL reward."""
    fitness_weight: float = Field(1.0, ge=0.0, description="Weight on surrogate predicted fitness progression")
    esm_plausibility_weight: float = Field(0.5, ge=0.0, description="Weight on ESM-2 zero-shot log-odds score")
    novelty_weight: float = Field(0.2, ge=0.0, description="Weight on evolutionary memory representational novelty")
    conservation_penalty: float = Field(0.05, ge=0.0, description="Penalty per step to favor parsimonious modifications")
    violation_penalty: float = Field(5.0, ge=0.0, description="Penalty applied if candidate violates biophysical constraints")


class CompositeRewardFunction:
    """Computes scalar reward balancing fitness improvement, PLM plausibility, memory novelty, and parsimony."""

    def __init__(
        self,
        evaluator: MultiObjectiveEvaluator,
        weights: Optional[RewardWeights] = None,
        memory: Optional[EvolutionaryMemory] = None,
        clip_reward: Optional[Tuple[float, float]] = (-10.0, 10.0),
    ) -> None:
        self.evaluator = evaluator
        self.weights = weights or RewardWeights()
        self.memory = memory
        self.clip_reward = clip_reward

    def compute_reward(
        self,
        current_sequence: str,
        next_sequence: str,
        mutations: List[Mutation],
        prev_fitness: float,
        step: int,
        is_feasible: bool = True,
        violations: Optional[List[str]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Calculates scalar transition reward and returns detailed component breakdown.

        Args:
            current_sequence: Protein sequence before action.
            next_sequence: Protein sequence after mutation.
            mutations: Full cumulative mutation list of next variant.
            prev_fitness: Predicted fitness of current sequence.
            step: Current episode step index.
            is_feasible: Whether biophysical constraints are satisfied.
            violations: List of violation descriptions if any.

        Returns:
            Tuple of (total_scalar_reward, component_breakdown_dict).
        """
        # Form temporary individual for evaluation
        v = VariantRecord(
            variant_id=f"RL_STEP_{step}",
            protein_id="DESIGN",
            mutations=mutations,
            mutated_sequence=next_sequence,
            is_feasible=is_feasible,
            feasibility_violations=violations or [],
        )
        ind = Individual(variant=v)
        self.evaluator.evaluate_individual(ind)

        # 1. Fitness progression reward: Delta Fitness = y_{t+1} - y_t
        curr_fitness = float(ind.fitness_values.get("predicted_fitness", 0.0))
        delta_fitness = curr_fitness - prev_fitness
        fitness_reward = self.weights.fitness_weight * delta_fitness

        # 2. ESM-2 Evolutionary Plausibility reward
        esm_score = float(ind.fitness_values.get("esm2_plausibility", ind.fitness_values.get("esm2_log_plausibility", 0.0)))
        esm_reward = self.weights.esm_plausibility_weight * esm_score

        # 3. Novelty bonus from EvolutionaryMemory
        novelty_val = 0.0
        if self.memory is not None and self.weights.novelty_weight > 0.0:
            try:
                novelty_val = self.memory.get_novelty_score(next_sequence)
            except Exception:
                novelty_val = 0.0
        novelty_reward = self.weights.novelty_weight * novelty_val

        # 4. Conservation penalty (favors minimal mutations)
        cons_penalty = self.weights.conservation_penalty

        # 5. Feasibility violation penalty
        viol_penalty = 0.0
        if not is_feasible:
            viol_penalty = self.weights.violation_penalty * len(violations or [1])

        raw_total = fitness_reward + esm_reward + novelty_reward - cons_penalty - viol_penalty

        if self.clip_reward is not None:
            clipped_total = float(np.clip(raw_total, self.clip_reward[0], self.clip_reward[1]))
        else:
            clipped_total = float(raw_total)

        breakdown = {
            "total_reward": clipped_total,
            "delta_fitness": delta_fitness,
            "current_fitness": curr_fitness,
            "esm_score": esm_score,
            "novelty_score": novelty_val,
            "fitness_reward": fitness_reward,
            "esm_reward": esm_reward,
            "novelty_reward": novelty_reward,
            "conservation_penalty": cons_penalty,
            "violation_penalty": viol_penalty,
        }

        return clipped_total, breakdown
