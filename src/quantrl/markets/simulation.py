from abc import ABC, abstractmethod
from dataclasses import dataclass
import polars as pl
import numpy as np
import math

@dataclass
class MarketSimulator(ABC):
    
    @abstractmethod
    def reset(self) -> pl.DataFrame:
        pass


class StochasticProcesses:
    def __init__(self, seed: int | None):
        self.rng = np.random.RandomState(seed=seed)


    def generate_geometric_brownian_motion(
        self,
        n_paths: int,
        initial_value: float,
        mu: float,
        sigma: float,
        T: float,
        *,
        n: int | None,
        dt: float | None,
    ) -> np.ndarray:
        assert n is None or dt is None, (
            "Specify either dt or n, but not both."
        )
        dt = dt or T / n
        n = n or int(math.floor(T / dt))
        drift = (mu - 0.5 * sigma ** 2) * np.linspace(0, T, n)
        increments = self.rng.standard_normal(size=(n_paths, n))
        increments[:, 0] = 0
        return initial_value * np.exp(drift + sigma * increments.cumsum(axis=-1) * math.sqrt(dt))


if __name__ == "__main__":
    test = StochasticProcesses(123)
    import matplotlib.pyplot as plt
    plt.plot(
        test.generate_geometric_brownian_motion(
            100,
            1,
            0.005,
            0.05,
            1,
            n=200,
            dt=None,
        ).T
    )
    plt.show()
