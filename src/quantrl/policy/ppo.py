import numpy as np
from numba import njit
from dataclasses import dataclass
import torch
from torch import nn
import gymnasium as gym
from copy import deepcopy
from quantrl.agents.actor_critic import ActorCritic
from quantrl.utils.buffer import RolloutBuffer
from quantrl.policy.base import BasePolicy
from typing import Tuple, Any


@dataclass
class PPO(BasePolicy):
    actor_critic: ActorCritic
    learning_rate: float = 0.001
    betas: Tuple[float, float] = (0.9, 0.999)
    gamma: float = 0.97
    gae_lambda: float = 1.0
    epsilon: float = 0.1
    normalise_rewards: bool = False

    def __post_init__(self):
        assert not self.actor_critic.critic.include_action
        assert 0 <= self.gae_lambda <= 1
        assert 0 <= self.gamma <= 1
        self._actor_critic_old = deepcopy(self.actor_critic)


        self.optim = torch.optim.Adam(
            [
                {'params': self.actor_critic.parameters()},
            ], 
            lr=self.learning_rate,
            betas=self.betas
        )

    def learn(self, epochs: int, buffer: RolloutBuffer) -> None:
        actions, states, rewards, terminal_states = buffer[:]
        if self.normalise_rewards:
            raise NotImplementedError()

        with torch.no_grad():
            _, old_logprobs, _ = self._actor_critic_old.forward(states, actions)
        
        for _ in range(epochs):
            _, log_probs_new, state_values = self.actor_critic.forward(states, actions)
            advantages = self._gae_advantages(
                terminal_states=terminal_states,
                values=state_values.detach().numpy(),
                rewards=rewards,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda
            )
            advantages = torch.from_numpy(advantages)
            
            # Policy loss
            ratio = torch.exp(log_probs_new - old_logprobs)
            clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            policy_loss = (-torch.minimum(ratio, clipped_ratio) * advantages).mean()
            
            # Critic loss
            critic_loss = torch.pow(advantages + state_values.detach() - state_values, 2).mean()

            total_loss = policy_loss + critic_loss
            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()

        self._actor_critic_old.load_state_dict(deepcopy(self.actor_critic.state_dict()))

    def act(self, state: torch.Tensor | np.ndarray, evaluation: bool = False) -> torch.Tensor:
        return self.actor_critic.actor.act(state, evaluation=evaluation)