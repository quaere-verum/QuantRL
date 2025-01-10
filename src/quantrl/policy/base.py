from dataclasses import dataclass
from abc import ABC, abstractmethod
from quantrl.utils.buffer import RolloutBuffer
import numpy as np
from numba import njit
from typing import Any
import torch


@dataclass
class BasePolicy(ABC):
    
    @abstractmethod
    def act(self, action, evaluation: bool = False) -> torch.Tensor:
        pass

    @abstractmethod
    def learn(self, epochs: int, buffer: RolloutBuffer) -> None:
        pass

    @staticmethod
    @njit
    def _monte_carlo_returns(
        terminal_states: np.ndarray[Any, bool],
        rewards: np.ndarray[Any, float],
        gamma: float
    ) -> np.ndarray[Any, float]:
        discounted_returns = np.zeros_like(rewards, dtype=np.float64)
        discounted_reward = 0
        for k in range(len(terminal_states) - 1, -1, -1):
            if terminal_states[k]:
                discounted_reward = 0
            discounted_reward = rewards[k] + (gamma * discounted_reward)
            discounted_returns[k] = discounted_reward
        return discounted_returns

    @staticmethod
    @njit
    def _gae_advantages(
        terminal_states: np.ndarray[Any, bool],
        values: np.ndarray[Any, float],
        rewards: np.ndarray[Any, float],
        gamma: float,
        gae_lambda: float,
    ) -> np.ndarray[Any, float]:
        advantages = np.zeros_like(rewards)
        values_next = np.roll(values, -1)
        values_next[-1] = 0
        decay_factor = (1 - terminal_states) * gamma
        temporal_difference_error = rewards + values_next * decay_factor - values
        decay_factor *= gae_lambda
        gae = 0.0
        for k in range(len(rewards) - 1, -1, -1):
            gae = temporal_difference_error[k] + decay_factor[k] * gae
            advantages[k] = gae
        return advantages

    @staticmethod
    @njit
    def _n_step_return(
        terminal_states: np.ndarray[Any, bool],
        rewards: np.ndarray[Any, float],
        target_q: np.ndarray[Any, float],
        indices: np.ndarray[Any, int],
        gamma: float,
        n_step: int,
        buffer_size: int,
        mean: float,
        std: float,
    ) -> np.ndarray[Any, float]:
        target_shape = target_q.shape
        batch_size = target_shape[0]
        target_q = target_q.reshape(batch_size, -1)
        returns = np.zeros(target_q.shape)
        trajectory_lengths = np.full(indices.shape, n_step)
        for n in range(n_step - 1, -1, -1):
            now = (indices + n) % buffer_size
            trajectory_lengths[terminal_states[now]] = n
            returns[terminal_states[now]] = 0.0
            returns = (rewards[now].reshape(-1, 1) - mean) / std + gamma * returns
        target_q[trajectory_lengths != n_step] = 0.0
        trajectory_lengths = trajectory_lengths.reshape(-1, 1)
        target_q = target_q * (gamma ** trajectory_lengths) + returns
        return target_q.reshape(target_shape)
