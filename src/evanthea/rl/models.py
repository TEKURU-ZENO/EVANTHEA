"""Actor-Critic neural network architectures with dynamic action masking for protein sequence optimization."""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCriticNetwork(nn.Module):
    """PyTorch Actor-Critic policy and value network with dynamic action masking."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Shared representation trunk
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Actor head (policy logits)
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Critic head (state value V(s))
        self.critic_head = nn.Linear(hidden_dim, 1)

        # Orthogonal weight initialization
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Small gain on policy head for near-uniform initial exploration
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through trunk, returning masked policy logits and critic state value."""
        features = self.trunk(obs)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)

        if action_mask is not None:
            # Apply -1e9 mask to invalid actions
            mask_bool = action_mask.to(dtype=torch.bool)
            logits = torch.where(mask_bool, logits, torch.tensor(-1e9, device=logits.device, dtype=logits.dtype))

        return logits, value

    def get_action(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        action_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """Samples an action from the policy for a single environment state.

        Args:
            obs: Observation array of shape (obs_dim,).
            action_mask: Boolean mask array of shape (action_dim,).
            deterministic: If True, selects argmax (greedy exploitation).

        Returns:
            Tuple of (action_int, log_prob_float, value_float).
        """
        self.eval()
        with torch.no_grad():
            if not isinstance(obs, torch.Tensor):
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            else:
                obs_t = obs.unsqueeze(0) if obs.ndim == 1 else obs

            mask_t = None
            if action_mask is not None:
                if not isinstance(action_mask, torch.Tensor):
                    mask_t = torch.as_tensor(action_mask, dtype=torch.bool).unsqueeze(0)
                else:
                    mask_t = action_mask.unsqueeze(0) if action_mask.ndim == 1 else action_mask

            logits, value = self.forward(obs_t, mask_t)

            if deterministic:
                action = int(torch.argmax(logits, dim=-1).item())
                dist = Categorical(logits=logits)
                log_prob = float(dist.log_prob(torch.tensor(action)).item())
            else:
                dist = Categorical(logits=logits)
                act_tensor = dist.sample()
                action = int(act_tensor.item())
                log_prob = float(dist.log_prob(act_tensor).item())

            val_float = float(value.item())

        return action, log_prob, val_float

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluates batch of state-action transitions during PPO training updates.

        Returns:
            Tuple of (log_probs, entropy, values).
        """
        logits, values = self.forward(obs, action_mask)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values

    def save(self, path: Union[Path, str]) -> None:
        """Saves network weights to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "state_dict": self.state_dict(),
        }, p)

    @classmethod
    def load(cls, path: Union[Path, str], device: str = "cpu") -> "ActorCriticNetwork":
        """Loads network weights from disk."""
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model = cls(
            obs_dim=checkpoint["obs_dim"],
            action_dim=checkpoint["action_dim"],
            hidden_dim=checkpoint["hidden_dim"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        return model
