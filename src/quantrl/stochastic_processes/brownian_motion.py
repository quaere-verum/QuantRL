from quantrl.stochastic_processes.base import StochasticProcessBase
import numpy as np
from dataclasses import dataclass
import math
from typing import Any


@dataclass
class GaussianNoise(StochasticProcessBase):
    mean: float
    standard_deviation: float

    def __post_init__(self):
        return super().__post_init__()

    def reset(self, timestep):
        return self._rng.normal(loc=self.mean, scale=self.standard_deviation, size=self.n_paths)

    def step(self, **kwargs):
        return self._rng.normal(loc=self.mean, scale=self.standard_deviation, size=self.n_paths)
    
    def generate_trajectories(self, T: float, *, n: int | None = None, dt: float | None = None):
        if n is None:
            assert dt is not None
            n = math.floor(T / dt)
        else:
            assert dt is None
        return self._rng.normal(loc=self.mean, scale=self.standard_deviation, size=(self.n_paths, n))
    

@dataclass
class BrownianMotion(GaussianNoise):
    initial_value: float

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.initial_value, (float, int))
        self._state: np.ndarray[Any, float] | None = None

    def reset(self, timestep):
        super().reset(timestep)
        self._state = np.full((self.n_paths,), self.initial_value, dtype=float)
        return self._state
    
    def step(self, **kwargs):
        self._state += super().step(**kwargs)
        return self._state
    
    def generate_trajectories(self, T: float, *, n: int | None = None, dt: float | None = None):
        increments = super().generate_trajectories(T, n=n, dt=dt)
        dt = dt or T / n
        increments = (increments - self.mean * (1 - dt)) * math.sqrt(dt)
        increments[:, 0] = 0
        return np.cumsum(increments, axis=-1) + self.initial_value
    

@dataclass 
class GeometricBrownianMotion(BrownianMotion):
    
    def __post_init__(self):
        super().__post_init__()

    def reset(self, timestep):
        return super().reset(timestep)
    
    def step(self, **kwargs):
        log_step = super().step(**kwargs)
        self._state *= np.exp(log_step)
        return self._state
    
    def generate_trajectories(self, T, *, n=None, dt=None):
        log_trajectories = super().generate_trajectories(T, n=n, dt=dt)
        return np.exp(log_trajectories - self.initial_value) * self.initial_value
