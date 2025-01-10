from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import math
from functools import partial
from typing import Any, Callable, Tuple
from enum import Enum

class NoiseType(Enum):
    GAUSSIAN  = 1
    T_DISTRIBUTION = 2
    LAPLACE = 3
    LOGISTIC = 4

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

@dataclass
class Noise(StochasticProcessBase):
    mean: float
    standard_deviation: float
    dof: float | None
    noise_type: NoiseType

    def __post_init__(self):
        super().__post_init__()
        if self.noise_type == NoiseType.T_DISTRIBUTION:
            assert self.dof is not None
        else:
            assert self.dof is None
        match self.noise_type:
            case NoiseType.GAUSSIAN:
                scale = self.standard_deviation
                self._dist = partial(self._rng.normal, loc=0, scale=scale)
                self._scale_factor = 1
            case NoiseType.T_DISTRIBUTION:
                self._dist = partial(self._rng.standard_t, df=self.dof)
                self._scale_factor = np.sqrt((self.dof - 2) / self.dof) * self.standard_deviation
            case NoiseType.LOGISTIC:
                scale = self.standard_deviation * math.sqrt(3) / np.pi
                self._dist = partial(self._rng.logistic, loc=0, scale=scale)
                self._scale_factor = 1
            case NoiseType.LAPLACE:
                scale = self.standard_deviation / math.sqrt(2)
                self._dist = partial(self._rng.laplace, loc=0, scale=scale)
                self._scale_factor = 1
            

    def reset(self, timestep):
        return self._dist(size=self.n_paths) * self._scale_factor + self.mean

    def step(self, **kwargs):
        return self._dist(size=self.n_paths) * self._scale_factor + self.mean
    
    def generate_trajectories(self, T: float, *, n: int | None = None, dt: float | None = None):
        if n is None:
            assert dt is not None
            n = math.floor(T / dt)
        else:
            assert dt is None
        return self._dist(size=(self.n_paths, n)) * self._scale_factor + self.mean
