from dataclasses import dataclass
from typing import Tuple, Any
import numpy as np
from numba import njit
from abc import ABC, abstractmethod
from typing import Callable


class BaseBuffer(ABC):
    buffer_size: int

    @property
    @abstractmethod
    def steps_collected(self) -> int:
        pass

    @abstractmethod
    def __post_init__(self):
        self._filled: int
        self.num_envs: int

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def add(
        self, 
        action: float | np.ndarray[Any, float],
        state: np.ndarray[Any, float],
        reward: float,
        terminal_state: bool
    ) -> None:
        pass

    @abstractmethod
    def __getitem__(
        self, 
        indices: np.ndarray[Any, int] | int | slice | None
    ) -> Tuple[np.ndarray[Any, float], np.ndarray[Any, float], np.ndarray[Any, float], np.ndarray[Any, bool]]:
        pass

    def sample(
        self,
        sample_size: int, 
        with_replacement: bool = True
    ) -> Tuple[np.ndarray[Any, float], np.ndarray[Any, float], np.ndarray[Any, float], np.ndarray[Any, float], np.ndarray[Any, bool]]:
        indices = np.random.choice(self._filled, sample_size, with_replacement)
        return self[indices]
    
    @staticmethod
    @njit
    def calculate_monte_carlo_returns(
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
    def calculate_gae_advantages(
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
            returns = rewards[now].reshape(-1, 1) + gamma * returns
        target_q[trajectory_lengths != n_step] = 0.0
        trajectory_lengths = trajectory_lengths.reshape(-1, 1)
        target_q = target_q * (gamma ** trajectory_lengths) + returns
        return target_q.reshape(target_shape)
    
    def calculate_n_step_return(
        self,
        indices: np.ndarray[Any, int],
        target_q_function: Callable[[np.ndarray], np.ndarray],
        gamma: float,
        n_step: int,
    ) -> np.ndarray[Any, float]:
        assert indices.ndim == 1
        _, all_states, all_rewards, all_terminal_states = self[:self._filled]
        next_indices = (indices + n_step) % self._filled
        next_states = all_states[next_indices]
        target_q = target_q_function(next_states)
        return self._n_step_return(
            terminal_states=all_terminal_states,
            rewards=all_rewards,
            target_q=target_q,
            indices=indices,
            gamma=gamma,
            n_step=n_step,
            buffer_size=self._filled,
        )

@dataclass
class RolloutBuffer(BaseBuffer):
    action_shape: int | Tuple[int, ...]
    state_shape: Tuple[int, ...]
    buffer_size: int

    def __post_init__(self):
        self.num_envs = 1
        if isinstance(self.action_shape, (int, np.int_)):
            self.action_shape = (self.action_shape,)
        else:
            assert isinstance(self.action_shape, tuple)
        self._ptr = 0
        self._filled = 0
        self._action_buffer = np.empty((self.buffer_size,) + self.action_shape, dtype=float)
        self._state_buffer = np.empty((self.buffer_size,) + self.state_shape, dtype=float)
        self._reward_buffer = np.empty(self.buffer_size, dtype=float)
        self._terminal_state_buffer = np.empty(self.buffer_size, dtype=bool)
    
    @property
    def steps_collected(self) -> int:
        return self._filled

    def reset(self):
        self._ptr = 0
        self._filled = 0
        self._action_buffer = np.empty((self.buffer_size,) + self.action_shape, dtype=float)
        self._state_buffer = np.empty((self.buffer_size,) + self.state_shape, dtype=float)
        self._reward_buffer = np.empty(self.buffer_size, dtype=float)
        self._terminal_state_buffer = np.empty(self.buffer_size, dtype=bool)

    def add(
        self, 
        action: float | np.ndarray[Any, float],
        state: np.ndarray[Any, float],
        reward: float,
        terminal_state: bool
    ) -> None:
        self._action_buffer[self._ptr] = action
        self._state_buffer[self._ptr] = state
        self._reward_buffer[self._ptr] = reward
        self._terminal_state_buffer[self._ptr] = terminal_state
        self._ptr += 1
        if self._ptr >= self.buffer_size:
            self._ptr = 0
        if self._filled < self.buffer_size:
            self._filled += 1

    def __getitem__(
        self, 
        indices: np.ndarray[Any, int] | int | slice | None
    ) -> Tuple[np.ndarray[Any, float], np.ndarray[Any, float], np.ndarray[Any, float], np.ndarray[Any, bool]]:
        if isinstance(indices, (np.ndarray)):
            assert indices.max() <= self._filled
        elif isinstance(indices, int):
            assert indices <= self._filled
        elif indices is None:
            assert self._filled == self.buffer_size
        elif isinstance(indices, slice):
            if indices.stop is None:
                assert self._filled == self.buffer_size
            else:
                assert indices.stop <= self._filled
        return (
            self._action_buffer[indices],
            self._state_buffer[indices],
            self._reward_buffer[indices],
            self._terminal_state_buffer[indices]
        )
    

@dataclass
class VectorRolloutBuffer(BaseBuffer):
    action_shape: Tuple[int, ...]
    state_shape: Tuple[int, ...]
    buffer_size: int
    num_envs: int

    def __post_init__(self):
        assert self.num_envs > 1
        assert isinstance(self.action_shape, tuple)
        self._ptr = 0
        self._filled = 0
        self._action_buffer = np.empty((self.buffer_size, self.num_envs) + self.action_shape, dtype=float)
        self._state_buffer = np.empty((self.buffer_size, self.num_envs) + self.state_shape, dtype=float)
        self._reward_buffer = np.empty((self.buffer_size, self.num_envs), dtype=float)
        self._terminal_state_buffer = np.empty((self.buffer_size, self.num_envs), dtype=bool)

    @property
    def steps_collected(self) -> int:
        return self.num_envs * self._filled

    def reset(self):
        self._ptr = 0
        self._filled = 0
        self._action_buffer = np.empty((self.buffer_size, self.num_envs) + self.action_shape, dtype=float)
        self._state_buffer = np.empty((self.buffer_size, self.num_envs) + self.state_shape, dtype=float)
        self._reward_buffer = np.empty((self.buffer_size, self.num_envs), dtype=float)
        self._terminal_state_buffer = np.empty((self.buffer_size, self.num_envs), dtype=bool)

    def add(
        self, 
        action: np.ndarray[Any, float],
        state: np.ndarray[Any, float],
        reward: np.ndarray[Any, float],
        terminal_state: np.ndarray[Any, bool]
    ) -> None:
        if len(self.action_shape) == 1 and self.action_shape[0] == 1:
            action = action.reshape(-1, 1)
        if len(self.state_shape) == 1 and self.state_shape[0] == 1:
            state = state.reshape(-1, 1)
        self._action_buffer[self._ptr] = action
        self._state_buffer[self._ptr] = state
        self._reward_buffer[self._ptr] = reward
        self._terminal_state_buffer[self._ptr] = terminal_state
        self._ptr += 1
        if self._ptr >= self.buffer_size:
            self._ptr = 0
        if self._filled < self.buffer_size:
            self._filled += 1

    def __getitem__(
        self, 
        indices: np.ndarray[Any, int] | int | slice | None
    ) -> Tuple[np.ndarray[Any, float], np.ndarray[Any, float], np.ndarray[Any, float], np.ndarray[Any, bool]]:
        if isinstance(indices, (np.ndarray)):
            assert indices.max() <= self._filled * self.num_envs
        elif isinstance(indices, int):
            assert indices <= self._filled * self.num_envs
        elif indices is None:
            assert self._filled == self.buffer_size
        elif isinstance(indices, slice):
            if indices.stop is None:
                assert self._filled == self.buffer_size
            else:
                assert indices.stop <= self._filled * self.num_envs
        actions = np.transpose(self._action_buffer, (1, 0) + tuple(range(2, len(self.action_shape) + 2))).reshape((-1,) + self.action_shape)
        states = np.transpose(self._state_buffer, (1, 0) + tuple(range(2, len(self.state_shape) + 2))).reshape((-1,) + self.state_shape)
        rewards = np.transpose(self._reward_buffer, (1, 0)).reshape(-1)
        terminal_states = self._terminal_state_buffer.copy()
        # Terminal states were gathered from vectorised envs, and may have been terminated early.
        # These trajectories will now be stacked, so we need to set the final 'terminal_state'
        # in each env's buffer to True, to indicate this trajectory was truncated
        terminal_states[self._filled - 1] = True
        terminal_states = np.transpose(terminal_states, (1, 0)).reshape(-1)
        return (
            actions[indices],
            states[indices],
            rewards[indices],
            terminal_states[indices]
        )
    