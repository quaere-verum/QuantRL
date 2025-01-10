from dataclasses import dataclass
from typing import Tuple, Any
import numpy as np
from abc import ABC, abstractmethod


class BaseBuffer(ABC):
    buffer_size: int

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

@dataclass
class RolloutBuffer(BaseBuffer):
    action_shape: int
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