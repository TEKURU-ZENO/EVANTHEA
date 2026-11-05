"""Data schemas and result containers for Reinforcement Learning protein design."""

from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field

from evanthea.data.schemas import Mutation, VariantRecord


class ActionDefinition(BaseModel):
    """Mapping of a discrete action index to a concrete single-point substitution."""
    action_id: int = Field(..., ge=0, description="Discrete action integer index")
    position_1based: int = Field(..., gt=0, description="1-indexed residue position in sequence")
    position_0based: int = Field(..., ge=0, description="0-indexed residue position in sequence")
    ref_residue: str = Field(..., min_length=1, max_length=1, description="Wild-type/current reference residue")
    mut_residue: str = Field(..., min_length=1, max_length=1, description="Proposed mutant substitution residue")

    def to_mutation(self) -> Mutation:
        """Converts action definition to EVANTHEA Mutation model."""
        return Mutation(
            ref_residue=self.ref_residue,
            position_1based=self.position_1based,
            mut_residue=self.mut_residue,
        )


class RLStepResult(BaseModel):
    """Observation and outcome container following an environment transition step."""
    observation: List[float] = Field(..., description="Vectorized environment state observation")
    reward: float = Field(..., description="Scalar multi-objective transition reward")
    terminated: bool = Field(False, description="True if episode concluded naturally (e.g. goal reached or stop action)")
    truncated: bool = Field(False, description="True if episode was truncated by max step horizon")
    info: Dict[str, Any] = Field(default_factory=dict, description="Metadata including variant record and sub-rewards")

    model_config = {
        "arbitrary_types_allowed": True
    }


class TrajectoryStep(BaseModel):
    """Single transition record collected during policy rollout."""
    state: List[float] = Field(..., description="Observation vector at step t")
    action: int = Field(..., ge=0, description="Action taken at step t")
    reward: float = Field(..., description="Reward received after step t")
    value: float = Field(0.0, description="Critic state-value estimate V(s_t)")
    log_prob: float = Field(0.0, description="Policy log-probability log pi(a_t | s_t)")
    action_mask: Optional[List[bool]] = Field(None, description="Action mask applied at step t")
    terminated: bool = Field(False, description="Termination flag")
    truncated: bool = Field(False, description="Truncation flag")


class Trajectory(BaseModel):
    """Complete episodic trajectory of an RL protein design agent."""
    trajectory_id: str = Field(..., description="Unique trajectory identifier")
    steps: List[TrajectoryStep] = Field(default_factory=list, description="Sequence of step transitions")
    initial_sequence: str = Field(..., description="Starting sequence (wild-type)")
    final_sequence: str = Field(..., description="Final optimized variant sequence")
    mutations: List[Mutation] = Field(default_factory=list, description="All substitutions applied along trajectory")
    total_reward: float = Field(0.0, description="Cumulative episodic return sum(R_t)")
    final_predicted_fitness: float = Field(0.0, description="Fitness score of final sequence")
    final_esm_score: float = Field(0.0, description="ESM-2 evolutionary plausibility of final sequence")
    novelty_score: float = Field(0.0, description="Memory novelty score of final variant")


class PolicyEvaluationSummary(BaseModel):
    """Aggregated evaluation metrics across multiple policy rollouts."""
    episodes: int = Field(..., ge=1, description="Number of evaluation episodes evaluated")
    mean_return: float = Field(..., description="Mean cumulative episodic reward")
    std_return: float = Field(..., description="Standard deviation of episodic reward")
    best_fitness: float = Field(..., description="Highest predicted fitness achieved across rollouts")
    mean_fitness: float = Field(..., description="Average final predicted fitness")
    mean_mutations: float = Field(..., description="Average number of mutations introduced per trajectory")
    unique_variants: int = Field(..., ge=0, description="Count of distinct unique sequences generated")
    pareto_front_size: int = Field(0, ge=0, description="Number of non-dominated variants discovered")
