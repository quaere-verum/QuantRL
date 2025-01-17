from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple
import polars as pl
import numpy as np
import pandas as pd
import scipy.stats
import math
from numba import njit


class MarketSimulator(ABC):
    
    @abstractmethod
    def reset(self) -> pl.DataFrame:
        pass


@dataclass
class HestonProcessMarketSimulator(MarketSimulator):
    process_generator: "StochasticProcesses"
    """
    The object generating the prices and volumes for each asset.
    """
    initial_value: np.ndarray
    """
    The initial price for each of the assets.
    """
    mu: np.ndarray
    """
    The drift in the Brownian motion driving the prices for each of the assets.
    """
    cov: np.ndarray
    """
    The covariance matrix specifying the correlations between the Brownian motion
    driving the prices.
    """
    initial_value_vol: np.ndarray
    """
    The initial volatility.
    """
    cov_vol: np.ndarray
    """
    The covariance matrix specifying the correlations between the Brownian motion
    driving the volatilities.
    """
    sigma_vol: np.ndarray
    """
    Volatility of volatility for each of the assets.
    """
    theta_vol: np.ndarray
    """
    Mean reversion speed of the volatility process for each of the assets.
    """
    mean_vol: np.ndarray
    """
    Mean value that the volatility reverts to for each of the assets.
    """
    rho: np.ndarray
    """
    The correlation between the Brownian motion driving the price, and the Brownian motion
    driving the volatility for each of the assets.
    """
    volume_vol_sensitivity_window: int
    """
    The size of the moving average window for volatility, as input for the volume model.
    """
    volume_vol_sensitivity: np.ndarray
    """
    Regression coefficient (regressing volume on moving average of volatility).
    """
    volume_price_change_sensitivity_window: int
    """
    The size of the moving average window for the size of price changes, as input for the volume model.
    """
    volume_price_change_sensitivity: np.ndarray
    """
    Regression coefficient (regressing volume on moving average of absolute value of (logarithmic) returns).
    """
    volume_noise_std: np.ndarray
    """
    Standard deviation for the noise (normally distributed) that is added to the volume.
    """
    n: int
    """
    Number of timesteps to generate.
    """
    T: float
    """
    Ending time (in days).
    """

    def __post_init__(self) -> None:
        self._date_ids = np.floor(np.linspace(0, self.T * (self.n - 1) / self.n, self.n)).astype(int)
        self._time_ids = np.arange(self.n) % int(math.floor(self.n / self.T))
        self._market_ids = np.arange(self.n)
        self._schema = {
            "market_id": pl.Int32,
            "date_id": pl.Int32,
            "time_id": pl.Int32,
            "symbol_id": pl.Int32,
            "midprice": pl.Float64,
            "volume": pl.Float64,
        }

    def reset(self) -> None:
        prices, volumes = self.process_generator.generate_heston_process_with_volume(
            n_paths=None, # n_paths is None because it is determined by the size of the covariance matrix
            initial_value=self.initial_value,
            mu=self.mu,
            cov=self.cov,
            initial_value_vol=self.initial_value_vol,
            cov_vol=self.cov_vol,
            sigma_vol=self.sigma_vol,
            theta_vol=self.theta_vol,
            mean_vol=self.mean_vol,
            rho=self.rho,
            volume_vol_sensitivity_window=self.volume_vol_sensitivity_window,
            volume_vol_sensitivity=self.volume_vol_sensitivity,
            volume_price_change_sensitivity_window=self.volume_price_change_sensitivity_window,
            volume_price_change_sensitivity=self.volume_price_change_sensitivity,
            volume_noise_std=self.volume_noise_std,
            n=self.n,
            T=self.T,
            dt=None, # dt is None because it is already determined by n and T
        )
        data = (
                pd.DataFrame(
                columns=pd.MultiIndex.from_product((np.arange(len(prices)), ("midprice", "volume")), names=["symbol_id", "value"]),
                index=pd.MultiIndex.from_arrays((self._market_ids, self._date_ids, self._time_ids), names=["market_id", "date_id", "time_id"]),
                data=np.concatenate((prices.T, volumes.T), axis=1)
            )
            .swaplevel(axis=1)
            .stack(future_stack=True)
            .reset_index()
        )
        return pl.DataFrame(data=data, schema=self._schema)


class StochasticProcesses:
    def __init__(self, seed: int | None):
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self._compile()

    def _compile(self) -> None:
        self._ou_process(
            n_paths=1,
            n=2,
            initial_value=0,
            mu=np.array([0]),
            theta=np.array([1]),
            sigma=np.array([1]),
            dt=1,
            increments=np.array([[0.1, 0.1]])
        )
        self._cir_process(
            n_paths=1,
            n=2,
            initial_value=1,
            mu=np.array([0]),
            theta=np.array([1]),
            sigma=np.array([1]),
            dt=1,
            increments=np.array([[0.1, 0.1]])
        )

    def generate_geometric_brownian_motion(
        self,
        *,
        n_paths: int | None,
        initial_value: np.ndarray | float,
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
            if isinstance(initial_value, float):
                initial_value = np.full(mu.size, initial_value)
            else:
                assert initial_value.shape == mu.shape
        else:
            assert n_paths is not None
            cov = np.eye(n_paths)
            if isinstance(mu, float):
                mu = np.full(n_paths, mu)
            if isinstance(sigma, float):
                sigma = np.full(n_paths, sigma)
            if isinstance(initial_value, float):
                initial_value = np.full(n_paths, initial_value)
        assert (T is None or dt is None) and (T is not None or dt is not None), (
            "Specify either T or dt, but not both."
        )
        dt = dt or T / n
        n = n or int(math.floor(T / dt))
        drift = ((mu - 0.5 * sigma ** 2) * np.linspace(0, T, n).reshape(-1, 1)).T
        increments = self.rng.multivariate_normal(mean=mu, cov=cov, size=n).T
        increments[:, 0] = 0
        return initial_value.reshape(-1, 1) * np.exp(drift + sigma.reshape(-1, 1) * increments.cumsum(axis=-1) * math.sqrt(dt))
    
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
        initial_value: np.ndarray | float,
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
            if isinstance(initial_value, float):
                initial_value = np.full(mu.size, initial_value)
            else:
                assert initial_value.shape == mu.shape
            n_paths = mu.size
        else:
            assert n_paths is not None
            cov = np.eye(n_paths)
            if isinstance(mu, float):
                mu = np.full(n_paths, mu)
            if isinstance(sigma, float):
                sigma = np.full(n_paths, sigma)
            if isinstance(initial_value, float):
                initial_value = np.full(n_paths, initial_value)
        assert (T is None or dt is None) and (T is not None or dt is not None), (
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
        initial_value: np.ndarray | float,
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
            if isinstance(initial_value, float):
                initial_value = np.full(mu.size, initial_value)
            else:
                assert initial_value.shape == mu.shape
        else:
            assert n_paths is not None
            cov = np.eye(n_paths)
            if isinstance(mu, float):
                mu = np.full(n_paths, mu)
            if isinstance(sigma, float):
                sigma = np.full(n_paths, sigma)
            if isinstance(initial_value, float):
                initial_value = np.full(n_paths, initial_value)
        assert (T is None or dt is None) and (T is not None or dt is not None), (
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
        # Strictly speaking should be o.T @ ... @ o, but o is symmetric
        cov = o.T @ np.diag(principal_components) @ o
        if normalise:
            d = np.diag(1/np.sqrt(np.diag(cov)))
            # Strictly speaking should be d.T @ cov @ d, but d is symmetric
            return d @ cov @ d
        else:
            return cov 

    @staticmethod
    @njit
    def _cir_process(
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
                + sigma * np.sqrt(np.clip(x[:, t - 1], 1e-8, np.inf)) * np.sqrt(dt) * increments[:, t - 1]
            )
        return x

    def generate_cir_process(
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
        assert (T is None or dt is None) and (T is not None or dt is not None), (
            "Specify either T or dt, but not both."
        )
        dt = dt or T / n
        increments = self.rng.multivariate_normal(mean=mu, cov=cov, size=n-1).T
        x = self._cir_process(
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

    def generate_heston_process(
        self,
        *,
        n_paths: int | None,
        initial_value: np.ndarray | float,
        mu: np.ndarray | float,
        cov: np.ndarray | None,
        initial_value_vol: float,
        cov_vol: np.ndarray | None,
        sigma_vol: np.ndarray | float,
        theta_vol: np.ndarray | float,
        mean_vol: np.ndarray | float,
        rho: np.ndarray | float,
        n: int,
        T: float | None,
        dt: float | None,
    ) -> np.ndarray:
        if cov is not None:
            assert isinstance(mu, np.ndarray) 
            assert mu.ndim == 1 and cov.shape == (mu.size, mu.size) and mu.shape
            assert n_paths is None
            if cov_vol is not None:
                assert cov_vol.shape == cov.shape
            else:
                cov_vol = np.eye(mu.size)
            assert isinstance(sigma_vol, np.ndarray)
            assert isinstance(theta_vol, np.ndarray)
            assert isinstance(rho, np.ndarray)
            assert isinstance(mean_vol, np.ndarray)
            if isinstance(initial_value, float):
                initial_value = np.full(mu.size, initial_value)
            else:
                assert initial_value.shape == mu.shape
        else:
            assert n_paths is not None
            assert cov_vol is None
            cov = np.eye(n_paths)
            cov_vol = np.eye(n_paths)
            if isinstance(mu, float):
                mu = np.full(n_paths, mu)
            if isinstance(sigma_vol, float):
                sigma_vol = np.full(n_paths, sigma_vol)
            if isinstance(rho, float):
                rho = np.full(n_paths, rho)
            if isinstance(theta_vol, float):
                theta_vol = np.full(n_paths, theta_vol)
            if isinstance(mean_vol, float):
                mean_vol = np.full(n_paths, mean_vol)
            if isinstance(initial_value, float):
                initial_value = np.full(n_paths, initial_value)
        assert np.all(2 * theta_vol * mean_vol > sigma_vol ** 2)
        assert (T is None or dt is None) and (T is not None or dt is not None), (
            "Specify either T or dt, but not both."
        )
        dt = dt or T / n
        T = T or n / dt
        cov_total = np.block(
            [
                [cov, np.diag(rho) * np.sqrt(np.diag(cov))],
                [np.diag(rho) * np.sqrt(np.diag(cov)), cov_vol]
            ]
        )
        assert np.all(np.linalg.eigvals(cov_total) > 0)
        mu_total = np.concatenate((mu, np.zeros(mu.size)), axis=0)
        increments = self.rng.multivariate_normal(mu_total, cov_total, size=n).T
        increments_vol = increments[mu.size:, :]
        increments_asset = increments[:mu.size, :]
        volatility_squared = self._cir_process(
            n_paths=mu.size,
            n=n,
            initial_value=initial_value_vol,
            mu=mean_vol,
            theta=theta_vol,
            sigma=sigma_vol,
            dt=dt,
            increments=increments_vol,
        )
        time = np.linspace(0, T, n)

        return (
            initial_value.reshape(-1, 1) 
            * np.exp(
                (mu.reshape(-1, 1) - volatility_squared / 2) * time 
                + np.cumsum(np.sqrt(volatility_squared) * increments_asset * math.sqrt(dt), axis=-1)
            )
        )
    
    def generate_volumes_from_prices(
        self,
        *,
        prices: np.ndarray,
        volatilities: np.ndarray,
        noise_std: np.ndarray,
        price_change_sensitivity: np.ndarray,
        volatility_sensitivity: np.ndarray,
        price_change_window: int,
        volatility_window: int,
        n: int,
    ) -> np.ndarray:
        volume_noise = self.rng.normal(0, noise_std, size=n)
        price_returns = np.log(prices[..., 1:] / prices[..., :-1])
        price_returns = np.lib.stride_tricks.sliding_window_view(
            price_returns,
            window_shape=price_change_window,
            axis=-1,
        ).mean(axis=-1)
        price_returns = np.concatenate(
            (
                np.full((prices.shape[0], price_change_window), price_returns[:, [0]]),
                price_returns,
            ),
            axis=-1,
        )
        volume_vol = np.lib.stride_tricks.sliding_window_view(
            volatilities,
            window_shape=volatility_window,
            axis=-1,
        ).mean(axis=-1)
        volume_vol = np.concatenate(
            (
                np.full((prices.shape[0], volatility_window - 1), volume_vol[:, [0]]),
                volume_vol
            ),
            axis=-1,
        )
        volumes = np.exp(
            volatility_sensitivity * volume_vol
            + price_change_sensitivity * np.abs(price_returns)
            + volume_noise
        )
        return volumes

    
    def generate_heston_process_with_volume(
        self,
        *,
        n_paths: int | None,
        initial_value: np.ndarray | float,
        mu: np.ndarray | float,
        cov: np.ndarray | None,
        initial_value_vol: float,
        cov_vol: np.ndarray | None,
        sigma_vol: np.ndarray | float,
        theta_vol: np.ndarray | float,
        mean_vol: np.ndarray | float,
        rho: np.ndarray | float,
        volume_vol_sensitivity_window: int,
        volume_vol_sensitivity: np.ndarray | float,
        volume_price_change_sensitivity_window: int,
        volume_price_change_sensitivity: np.ndarray | float,
        volume_noise_std: np.ndarray | float,
        n: int,
        T: float | None,
        dt: float | None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if cov is not None:
            assert isinstance(mu, np.ndarray) 
            assert mu.ndim == 1 and cov.shape == (mu.size, mu.size) and mu.shape
            assert n_paths is None
            if cov_vol is not None:
                assert cov_vol.shape == cov.shape
            else:
                cov_vol = np.eye(mu.size)
            assert isinstance(sigma_vol, np.ndarray)
            assert isinstance(theta_vol, np.ndarray)
            assert isinstance(rho, np.ndarray)
            assert isinstance(mean_vol, np.ndarray)
            if isinstance(initial_value, float):
                initial_value = np.full(mu.size, initial_value)
            else:
                assert initial_value.shape == mu.shape
        else:
            assert n_paths is not None
            assert cov_vol is None
            cov = np.eye(n_paths)
            cov_vol = np.eye(n_paths)
            if isinstance(mu, float):
                mu = np.full(n_paths, mu)
            if isinstance(sigma_vol, float):
                sigma_vol = np.full(n_paths, sigma_vol)
            if isinstance(rho, float):
                rho = np.full(n_paths, rho)
            if isinstance(theta_vol, float):
                theta_vol = np.full(n_paths, theta_vol)
            if isinstance(mean_vol, float):
                mean_vol = np.full(n_paths, mean_vol)
            if isinstance(initial_value, float):
                initial_value = np.full(n_paths, initial_value)
        assert np.all(2 * theta_vol * mean_vol > sigma_vol ** 2)
        assert (T is None or dt is None) and (T is not None or dt is not None), (
            "Specify either T or dt, but not both."
        )
        dt = dt or T / n
        T = T or n / dt
        cov_total = np.block(
            [
                [cov, np.diag(rho) * np.sqrt(np.diag(cov))],
                [np.diag(rho) * np.sqrt(np.diag(cov)), cov_vol]
            ]
        )
        assert np.all(np.linalg.eigvals(cov_total) > 0)
        mu_total = np.concatenate((mu, np.zeros(mu.size)), axis=0)
        increments = self.rng.multivariate_normal(mu_total, cov_total, size=n).T
        increments_vol = increments[mu.size:, :]
        increments_asset = increments[:mu.size, :]
        volatility_squared = self._cir_process(
            n_paths=mu.size,
            n=n,
            initial_value=initial_value_vol,
            mu=mean_vol,
            theta=theta_vol,
            sigma=sigma_vol,
            dt=dt,
            increments=increments_vol,
        )
        volatility = np.sqrt(volatility_squared)
        time = np.linspace(0, T, n)
        prices = (
            initial_value.reshape(-1, 1) 
            * np.exp(
                (mu.reshape(-1, 1) - volatility_squared / 2) * time 
                + np.cumsum(volatility * increments_asset * math.sqrt(dt), axis=-1)
            )
        )
        volumes = self.generate_volumes_from_prices(
            prices=prices,
            volatilities=volatility,
            noise_std=volume_noise_std,
            price_change_sensitivity=volume_price_change_sensitivity,
            volatility_sensitivity=volume_vol_sensitivity,
            price_change_window=volume_price_change_sensitivity_window,
            volatility_window=volume_vol_sensitivity_window,
            n=n,
        )
        return prices, volumes

    @staticmethod
    @njit
    def _hawkes_process(
        rng_generator: np.random.Generator,
        initial_intensity: np.ndarray,
        T: float,
    ) -> np.ndarray:
        raise NotImplementedError()


    def generate_hawkes_process(
        self,
        *,
        n_paths: int | None,
        intensity_lambda: np.ndarray | float,
        n: int,
        dt: float | None,
        T: float | None,
    ) -> np.ndarray:
        raise NotImplementedError()
