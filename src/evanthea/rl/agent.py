"""RLAgent interface connecting trained ActorCriticNetwork to EVANTHEA variant generation."""

import uuid
from typing import Any, Dict, List, Optional
import numpy as np

from evanthea.data.schemas import Mutation, VariantRecord
from evanthea.rl.environment import ProteinDesignEnv
from evanthea.rl.models import ActorCriticNetwork
from evanthea.rl.schema import Trajectory, TrajectoryStep


class RLAgent:
    """Trained reinforcement learning agent for directed protein variant generation."""

    def __init__(
        self,
        network: ActorCriticNetwork,
        protein_id: str = "DESIGN_PROT",
        deterministic: bool = False,
    ) -> None:
        self.network = network
        self.protein_id = protein_id
        self.deterministic = deterministic

    def rollout_episode(
        self,
        env: ProteinDesignEnv,
        deterministic: Optional[bool] = None,
    ) -> Trajectory:
        """Executes a complete design trajectory in the environment and returns Trajectory object."""
        is_det = deterministic if deterministic is not None else self.deterministic
        traj_id = f"traj_{uuid.uuid4().hex[:8]}"

        obs, info = env.reset()
        initial_seq = env.wildtype_sequence
        steps: List[TrajectoryStep] = []
        total_reward = 0.0

        mask = info["action_mask"]
        done = False

        while not done:
            action, log_prob, value = self.network.get_action(obs, action_mask=mask, deterministic=is_det)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            step_rec = TrajectoryStep(
                state=obs.tolist(),
                action=action,
                reward=float(reward),
                value=float(value),
                log_prob=float(log_prob),
                action_mask=mask.tolist() if mask is not None else None,
                terminated=terminated,
                truncated=truncated,
            )
            steps.append(step_rec)
            total_reward += reward

            obs = next_obs
            mask = step_info.get("action_mask", None)

        final_seq = env.current_sequence
        mutations = list(env.accumulated_mutations)

        return Trajectory(
            trajectory_id=traj_id,
            steps=steps,
            initial_sequence=initial_seq,
            final_sequence=final_seq,
            mutations=mutations,
            total_reward=float(total_reward),
            final_predicted_fitness=float(env.current_fitness),
            final_esm_score=float(env.current_esm_score),
            novelty_score=float(step_info.get("reward_breakdown", {}).get("novelty_score", 0.0)),
        )

    def propose_candidates(
        self,
        env: ProteinDesignEnv,
        n_candidates: int = 10,
        deterministic: bool = False,
    ) -> List[VariantRecord]:
        """Generates a batch of designed protein variants using the RL policy."""
        variants: List[VariantRecord] = []

        for i in range(n_candidates):
            traj = self.rollout_episode(env, deterministic=deterministic)
            var = VariantRecord(
                variant_id=f"RL_VAR_{traj.trajectory_id}",
                protein_id=self.protein_id,
                mutations=traj.mutations,
                mutated_sequence=traj.final_sequence,
                raw_score=traj.final_predicted_fitness,
                is_wildtype=len(traj.mutations) == 0,
                metadata={
                    "origin": "rl_agent",
                    "total_reward": traj.total_reward,
                    "final_esm_score": traj.final_esm_score,
                    "novelty_score": traj.novelty_score,
                    "steps": len(traj.steps),
                },
            )
            variants.append(var)

        return variants
