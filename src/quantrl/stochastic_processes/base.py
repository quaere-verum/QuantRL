from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Any


@dataclass
class StochasticProcessBase(ABC):
    n_paths: int
    seed: int | None

    def __post_init__(self):
        self._rng = np.random.RandomState(seed=self.seed)

    @abstractmethod
    def reset(self, timestep: int | None, **kwargs) -> np.ndarray[Any, float]:
        pass

    @abstractmethod
    def step(self, **kwargs) -> np.ndarray[Any, float]:
        pass

    @abstractmethod
    def generate_trajectories(
        self,
        T: float,
        *,
        n: int | None = None,
        dt: float | None = None,
    ) -> np.ndarray[Any, float]:
        pass