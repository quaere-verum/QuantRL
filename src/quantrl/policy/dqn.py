from quantrl.agents.actor_critic import QNet
from quantrl.policy.base import BasePolicy
from quantrl.utils.buffer import BaseBuffer
from dataclasses import dataclass
from copy import deepcopy
from typing import Tuple
import torch
import numpy as np
import gymnasium as gym

@dataclass 
class DQN(BasePolicy):
    q_network: QNet
    sample_size: int
    action_space: gym.spaces.Discrete
    target_update_frequency: int = 0
    learning_rate: float = 0.001
    betas: Tuple[float, float] = (0.9, 0.999)
    gamma: float = 0.99
    epsilon: float = 0.1
    double_dqn: bool = False
    n_step: int = 1

    def __post_init__(self):
        assert isinstance(self.action_space, gym.spaces.Discrete)
        self._use_target = self.target_update_frequency > 0
        if self.double_dqn:
            assert self._use_target
        if self._use_target:
            self._target_q_network = deepcopy(self.q_network)
        else:
            self._target_q_network = None
        self._rounds = 0
        self.optim = torch.optim.Adam(
            [
                {'params': self.q_network.parameters()},
            ], 
            lr=self.learning_rate,
            betas=self.betas
        )

    def _get_target_q_values(self, states_next: np.ndarray) -> torch.Tensor:
        q_values = self.q_network.value(states_next)
        actions = q_values.argmax(dim=-1)
        if self._use_target:
            target_q_values = self._target_q_network.value(states_next).max(dim=-1)[0]
        else:
            target_q_values = q_values
        if self.double_dqn:
            return target_q_values[:, actions]
        else:
            return target_q_values.max(dim=-1)[0]
        
    def _get_target_q_values(self, next_states: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            if self._use_target:
                target_q_values = self._target_q_network.value(next_states)
                if self.double_dqn:
                    q_values = self.q_network.value(next_states)
                    actions = q_values.argmax(dim=-1)
                    return target_q_values[:, actions].detach().numpy()
                else:
                    return target_q_values.max(dim=-1)[0].detach().numpy()
            else:
                q_values = self.q_network.value(next_states)
                return q_values.max(dim=-1)[0].detach().numpy()



    def act(self, state: torch.Tensor | np.ndarray, evaluation = False):
        action_values = self.q_network.value(state=state)
        actions = action_values.argmax(dim=-1) + self.action_space.start
        if not evaluation:
            random_exploration = np.random.choice(
                [False, True], 
                actions.shape[0], 
                replace=True, 
                p=[1 - self.epsilon, self.epsilon]
            )
            actions[random_exploration] = np.random.choice(
                self.action_space.n,
                random_exploration.sum(),
                replace=True,
            ) + self.action_space.start
        return actions


    def learn(self, epochs: int, buffer: BaseBuffer):
        for _ in range(epochs):
            self._rounds += 1
            if self._use_target and self._rounds % self.target_update_frequency:
                self._target_q_network.load_state_dict(
                    deepcopy(
                        self.q_network.state_dict()
                    )
                )
            sample_indices = np.random.choice(buffer.steps_collected, self.sample_size, replace=True)
            actions, states, _, _ = buffer[sample_indices]
            returns = buffer.calculate_n_step_return(
                indices=sample_indices,
                model=self._get_target_q_values,
                gamma=self.gamma,
                n_step=self.n_step
            )
            q_values = self.q_network.value(states)[:, actions]
            td_error = torch.from_numpy(returns) - q_values
            loss = td_error.pow(2).mean()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()