import numpy as np
from numba import njit
from dataclasses import dataclass
import torch
from torch import nn
import gymnasium as gym
from copy import deepcopy
from quantrl.agents.actor_critic import ActorCritic, ActorContinuous, ActorDiscrete
from quantrl.utils.buffer import RolloutBuffer
from quantrl.policy.base import BasePolicy
from typing import Tuple, Any

@dataclass
class VMPO(BasePolicy):
    actor_critic: ActorCritic
    eps_eta: float = 0.01
    eps_nu: float = 0.1
    learning_rate: float = 0.001
    betas: Tuple[float, float] = (0.9, 0.999)
    gamma: float = 0.95
    gae_lambda: float | None = None

    def __post_init__(self):
        assert not self.actor_critic.critic.include_action
        assert 0 <= self.gamma <= 1.0
        assert self.gae_lambda is None or 0 <= self.gae_lambda <= 1.0
        assert isinstance(self.actor_critic.actor, (ActorContinuous, ActorDiscrete))
        if isinstance(self.actor_critic.actor, ActorContinuous):
            self._discrete = False
        else:
            self._discrete = True
        self.eta = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.nu = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self._actor_critic_old = deepcopy(self.actor_critic)


        self.optim = torch.optim.Adam(
            [
                {'params': self.actor_critic.parameters()},
                {'params': self.eta},
                {'params': self.nu},
            ], 
            lr=self.learning_rate,
            betas=self.betas
        )

    def learn(self, epochs: int, buffer: RolloutBuffer) -> None:
        actions, states, rewards, terminal_states = buffer[:]
        if self.gae_lambda is None:
            monte_carlo_returns = self._monte_carlo_returns(
                terminal_states=terminal_states,
                rewards=rewards,
                gamma=self.gamma,
            )
            monte_carlo_returns = torch.from_numpy(monte_carlo_returns)
        with torch.no_grad():
            action_dist_old, _, _ = self._actor_critic_old.forward(states, actions)
        
        for _ in range(epochs):
            action_dist_new, log_probs_new, state_values = self.actor_critic.forward(states, actions)
            # Critic loss based on GAE or MC returns
            if self.gae_lambda is None:
                advantages = monte_carlo_returns - state_values.detach()
                critic_loss = torch.pow(monte_carlo_returns - state_values, 2).mean()
            else:
                advantages = self._gae_advantages(
                    terminal_states=terminal_states,
                    values=state_values.detach().numpy(),
                    rewards=rewards,
                    gamma=self.gamma,
                    gae_lambda=self.gae_lambda
                )
                advantages = torch.from_numpy(advantages)
                critic_loss = torch.pow(advantages + state_values.detach() - state_values, 2).mean()

            # KL loss
            if self._discrete:
                # In case of a discrete distribution, the KL divergence is just
                # \pi_old(a|s)*log(\pi_old(a|s)/\pi_new(a|s))
                kl_divergence = torch.mean(action_dist_old * (action_dist_old.log() - action_dist_new.log()), dim=1)
            else:
                # In case of a continuous action distribution, we assume the distribution is parameterised as a Gaussian with mean
                # mu and variance sigma. Then the KL divergence is computed as
                # 1/2 * (log(sigma_new ** 2/sigma_old ** 2) - 1 + (mu_old - mu_new) ** 2 / sigma_new ** 2 + sigma_old ** 2 / sigma_new ** 2
                action_dim = action_dist_old.shape[1] // 2
                mu_old, sigma_old = action_dist_old[:, :action_dim], action_dist_old[:, action_dim:] + 1e-8
                mu_new, sigma_new = action_dist_new[:, :action_dim], action_dist_new[:, action_dim:] + 1e-8
                variance_ratio = torch.pow(sigma_old / sigma_new, 2)
                kl_divergence = torch.mean(
                    (
                        torch.log(1 / variance_ratio)
                        - 1 + torch.pow((mu_old -  mu_new) / sigma_new, 2)
                        + variance_ratio
                    ) / 2
                )
            kl_loss = (
                self.nu * (self.eps_nu - kl_divergence.detach()) + 
                self.nu.detach() * kl_divergence
            ).mean()

            # Policy loss
            top_indices = torch.sort(advantages, descending=True).indices[:len(advantages) // 2]
            top_advantages = advantages[top_indices].detach()
            weights = torch.exp((top_advantages - top_advantages.max()) / self.eta.detach())
            weights = weights / weights.sum()
            policy_loss = -(
                 weights * log_probs_new[top_indices]
            ).sum()

            # Temperature loss
            temperature_loss = (
                self.eta * self.eps_eta + self.eta * (advantages.detach() - advantages.detach().max()).mean()
            )

            total_loss = policy_loss + kl_loss + critic_loss + temperature_loss

            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()

        self._actor_critic_old.load_state_dict(deepcopy(self.actor_critic.state_dict()))

    def act(self, state: torch.Tensor | np.ndarray, evaluation: bool = False) -> torch.Tensor:
        return self.actor_critic.actor.act(state, evaluation=evaluation)
