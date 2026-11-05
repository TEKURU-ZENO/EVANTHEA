"""Proximal Policy Optimization (PPO) training engine for protein sequence optimization."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from evanthea.rl.environment import ProteinDesignEnv
from evanthea.rl.models import ActorCriticNetwork
from evanthea.rl.schema import Trajectory, TrajectoryStep


class PPORolloutBuffer:
    """Stores episodic transitions and calculates Generalized Advantage Estimation (GAE)."""

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clear()

    def clear(self) -> None:
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.masks: List[np.ndarray] = []
        self.dones: List[bool] = []

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        action_mask: np.ndarray,
        done: bool,
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.masks.append(action_mask)
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates GAE advantages and target returns."""
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_val = last_value
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_val = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])

            delta = self.rewards[t] + self.gamma * next_val * next_non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values, dtype=np.float32)

        # Advantage normalization
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        norm_advantages = (advantages - adv_mean) / adv_std

        return (
            torch.as_tensor(norm_advantages, dtype=torch.float32),
            torch.as_tensor(returns, dtype=torch.float32),
        )


class PPOOptimizer:
    """Proximal Policy Optimization (PPO-Clip) with Generalized Advantage Estimation and action masking."""

    def __init__(
        self,
        network: ActorCriticNetwork,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        self.network = network.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def collect_rollouts(
        self,
        env: ProteinDesignEnv,
        target_steps: int = 64,
        max_evaluations: Optional[int] = None,
    ) -> PPORolloutBuffer:
        """Executes current policy in the environment to collect transition rollouts."""
        buffer = PPORolloutBuffer(gamma=self.gamma, gae_lambda=self.gae_lambda)
        steps_collected = 0

        obs, info = env.reset()
        mask = info["action_mask"]

        while steps_collected < target_steps:
            if max_evaluations is not None and env.evaluations_count >= max_evaluations:
                break

            action, log_prob, value = self.network.get_action(obs, action_mask=mask, deterministic=False)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            buffer.add(
                state=obs,
                action=action,
                reward=reward,
                value=value,
                log_prob=log_prob,
                action_mask=mask,
                done=done,
            )
            steps_collected += 1

            if done:
                obs, info = env.reset()
                mask = info["action_mask"]
            else:
                obs = next_obs
                mask = step_info.get("action_mask", env.get_action_mask())

        return buffer

    def train_step(self, buffer: PPORolloutBuffer) -> Dict[str, float]:
        """Performs PPO mini-batch updates on collected rollout buffer."""
        n_samples = len(buffer.states)
        if n_samples == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}

        advantages, returns = buffer.compute_returns_and_advantages()

        states_t = torch.as_tensor(np.array(buffer.states), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(np.array(buffer.actions), dtype=torch.int64, device=self.device)
        old_log_probs_t = torch.as_tensor(np.array(buffer.log_probs), dtype=torch.float32, device=self.device)
        masks_t = torch.as_tensor(np.array(buffer.masks), dtype=torch.bool, device=self.device)
        advantages_t = advantages.to(self.device)
        returns_t = returns.to(self.device)

        self.network.train()
        indices = np.arange(n_samples)

        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_entropies = []
        epoch_kls = []

        actual_batch_size = min(self.batch_size, n_samples)

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, actual_batch_size):
                end = start + actual_batch_size
                b_idx = indices[start:end]

                b_states = states_t[b_idx]
                b_actions = actions_t[b_idx]
                b_old_log_probs = old_log_probs_t[b_idx]
                b_masks = masks_t[b_idx]
                b_advantages = advantages_t[b_idx]
                b_returns = returns_t[b_idx]

                new_log_probs, entropy, new_values = self.network.evaluate_actions(
                    obs=b_states,
                    actions=b_actions,
                    action_mask=b_masks,
                )

                # Policy ratio & clipped objective
                log_ratio = new_log_probs - b_old_log_probs
                ratio = torch.exp(log_ratio)

                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * ((new_values - b_returns) ** 2).mean()

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()

                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                epoch_entropies.append(entropy.mean().item())
                epoch_kls.append(approx_kl)

        return {
            "policy_loss": float(np.mean(epoch_policy_losses)),
            "value_loss": float(np.mean(epoch_value_losses)),
            "entropy": float(np.mean(epoch_entropies)),
            "approx_kl": float(np.mean(epoch_kls)),
            "samples_trained": n_samples,
        }
