from dataclasses import dataclass
from typing import Tuple, Any
import numpy as np

@dataclass
class ReplayBuffer:
    action_shape: int
    state_shape: Tuple[int, ...]
    buffer_size: int

    def __post_init__(self):
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

    def sample(
        self,
        sample_size: int, 
        with_replacement: bool = True
    ) -> Tuple[np.ndarray[Any, float], np.ndarray[Any, float], np.ndarray[Any, float], np.ndarray[Any, float], np.ndarray[Any, bool]]:
        indices = np.random.choice(self._filled, sample_size, with_replacement)
        return self[indices]