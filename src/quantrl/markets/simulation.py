from abc import ABC, abstractmethod
from dataclasses import dataclass
import polars as pl
import numpy as np
import scipy.stats
import math
from numba import njit

@dataclass
class MarketSimulator(ABC):
    
    @abstractmethod
    def reset(self) -> pl.DataFrame:
        pass


class StochasticProcesses:
    def __init__(self, seed: int | None):
        self.seed = seed
        self.rng = np.random.RandomState(seed=seed)
        self._compile()

    def _compile(self) -> None:
        self.generate_ornstein_uhlenbeck_process(
            n_paths=1,
            initial_value=0,
            mu=np.array([0]),
            theta=np.array([1]),
            cov=None,
            sigma=np.array([1]),
            n=2,
            T=1,
            dt=None
        )

    def generate_geometric_brownian_motion(
        self,
        *,
        n_paths: int | None,
        initial_value: float,
        mu: float | np.ndarray,
        cov: np.ndarray | None,
        sigma: float | np.ndarray,
        n: int,
        T: float | None,
        dt: float | None,
    ) -> np.ndarray:
        if cov is not None:
            assert isinstance(mu, np.ndarray) 
            assert isinstance(sigma, np.ndarray)
            assert mu.ndim == 1 and cov.shape == (mu.size, mu.size) and mu.shape == sigma.shape
            assert n_paths is None
        else:
            assert n_paths is not None
            cov = np.eye(n_paths)
            if isinstance(mu, float):
                mu = np.full(n_paths, mu)
            if isinstance(sigma, float):
                sigma = np.full(n_paths, sigma)
        assert T is None or dt is None, (
            "Specify either T or dt, but not both."
        )
        dt = dt or T / n
        n = n or int(math.floor(T / dt))
        drift = ((mu - 0.5 * sigma ** 2) * np.linspace(0, T, n).reshape(-1, 1)).T
        increments = self.rng.multivariate_normal(mean=mu, cov=cov, size=n).T
        increments[:, 0] = 0
        return initial_value * np.exp(drift + sigma.reshape(-1, 1) * increments.cumsum(axis=-1) * math.sqrt(dt))
    
    @staticmethod
    @njit
    def _ou_process(
        *,
        n_paths: int,
        n: int,
        initial_value: float,
        mu: np.ndarray,
        theta: np.ndarray,
        sigma: np.ndarray,
        dt: float,
        increments: np.ndarray,
    ) -> np.ndarray:
        x = np.empty((n_paths, n), dtype=float)
        x[:, 0] = initial_value
        for t in range(1, n):
            x[:, t] = (
                x[:, t - 1] + theta * (mu - x[:, t - 1]) * dt
                + sigma * np.sqrt(dt) * increments[:, t - 1]
            )
        return x
    
    def generate_ornstein_uhlenbeck_process(
        self,
        *,
        n_paths: int | None,
        n: int,
        initial_value: float,
        mu: np.ndarray | float,
        theta: np.ndarray | float,
        cov: np.ndarray | None,
        sigma: np.ndarray | float,
        T: float | None,
        dt: float | None,
    ) -> np.ndarray:
        if cov is not None:
            assert isinstance(mu, np.ndarray) 
            assert isinstance(sigma, np.ndarray)
            assert mu.ndim == 1 and cov.shape == (mu.size, mu.size) and mu.shape == sigma.shape
            assert n_paths is None
            n_paths = mu.size
        else:
            assert n_paths is not None
            cov = np.eye(n_paths)
            if isinstance(mu, float):
                mu = np.full(n_paths, mu)
            if isinstance(sigma, float):
                sigma = np.full(n_paths, sigma)
        assert T is None or dt is None, (
            "Specify either T or dt, but not both."
        )
        dt = dt or T / n
        increments = self.rng.multivariate_normal(mean=mu, cov=cov, size=n-1).T
        x = self._ou_process(
            n_paths=n_paths,
            n=n,
            initial_value=initial_value,
            mu=mu,
            theta=theta,
            sigma=sigma,
            dt=dt, 
            increments=increments,
        )
        return x

    def generate_brownian_motion(
        self,
        *,
        n_paths: int | None,
        initial_value: float,
        mu: np.ndarray | float,
        cov: np.ndarray | None,
        sigma: np.ndarray | float,
        n: int,
        T: float | None,
        dt: float | None,
    ) -> np.ndarray:
        if cov is not None:
            assert isinstance(mu, np.ndarray) 
            assert isinstance(sigma, np.ndarray)
            assert mu.ndim == 1 and cov.shape == (mu.size, mu.size) and mu.shape == sigma.shape
            assert n_paths is None
        else:
            assert n_paths is not None
            cov = np.eye(n_paths)
            if isinstance(mu, float):
                mu = np.full(n_paths, mu)
            if isinstance(sigma, float):
                sigma = np.full(n_paths, sigma)
        assert T is None or dt is None, (
            "Specify either T or dt, but not both."
        )
        dt = dt or T / n
        increments = math.sqrt(dt) * self.rng.multivariate_normal(mean=mu, cov=cov, size=n).T
        increments[:, 0] = 0
        return increments.cumsum(axis=-1) + initial_value
        
    def generate_random_covariance_matrix(
        self,
        principal_components: np.ndarray,
        normalise: bool = True,
    ) -> np.ndarray:
        assert np.all(principal_components > 0) and principal_components.ndim == 1
        o = scipy.stats.ortho_group(
            dim=principal_components.size,
        ).rvs(random_state=self.rng)
        cov = o.T @ np.diag(principal_components) @ o
        if normalise:
            d = np.diag(1/np.sqrt(np.diag(cov)))
            return d @ cov @ d
        else:
            return cov 

if __name__ == "__main__":
    test = StochasticProcesses(123)
    import matplotlib.pyplot as plt
    plt.plot(
       test.generate_geometric_brownian_motion(
        n_paths=None,
        initial_value=100,
        mu=np.array([0.0, 0.0, -0.01]),
        cov=test.generate_random_covariance_matrix(np.array([1, 0.2, 0.2])),
        sigma=np.array([0.05, 0.1, 0.05]),
        n=200,
        dt=None,
        T=10,
       ).T
    )
    plt.show()
    plt.plot(
        test.generate_ornstein_uhlenbeck_process(
            n_paths=None,
            initial_value=0,
            mu=np.array([0.0, 0.0, -0.01]),
            cov=test.generate_random_covariance_matrix(np.array([1, 0.2, 0.2])),
            sigma=np.array([0.05, 0.1, 0.05]),
            theta=np.array([0.1, 0.1, 0.5]),
            n=200,
            dt=None,
            T=10,
        ).T
    )
    plt.show()