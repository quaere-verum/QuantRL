from __future__ import annotations
import gymnasium as gym
import polars as pl
import numpy as np
from typing import Any, Dict, Iterable, List
from dataclasses import dataclass
import quantrl as qrl
from abc import abstractmethod

@dataclass
class BaseEnv(gym.Env):
    market: qrl.Market
    cash_account: qrl.CashAccount
    portfolio: qrl.Portfolio
    predictive_model: qrl.PredictiveModel
    lags: int
    stride: int
    market_observation_columns: List[str] | None

    def __post_init__(self) -> None:
        self._t: int | None = None
        if self.market_observation_columns is None:
            self.market_observation_columns = self.market.market_data.columns
        self.observation_space = gym.spaces.Dict(
            {
                "market": gym.spaces.Box(0, np.inf, (self.lags + 1, len(self.market_observation_columns))),
                "portfolio": gym.spaces.Box(-np.inf, np.inf, self.portfolio.summary_shape),
                "cash_account": gym.spaces.Box(-np.inf, np.inf, (1,)),
                "predictive_model": gym.spaces.Box(-np.inf, np.inf, (len(self.predictive_model.symbols) + 1,)) # Predictions + performance
            }
        )

    @property
    @abstractmethod
    def state(self) -> Dict[str, np.ndarray[Any, float]]:
        pass

    @property
    @abstractmethod
    def reward(self) -> float:
        pass

    @property
    @abstractmethod
    def done(self) -> bool:
        pass

    @property
    @abstractmethod
    def truncated(self) -> bool:
        pass

    @property
    @abstractmethod
    def info(self) -> Dict[str, Any]:
        pass

    
    @abstractmethod
    def step(self, action: np.ndarray[Any, float | int]):
        pass

    

    def _process_action(self, action: np.ndarray[Any, float | int]) -> Iterable[Dict[str, float | int]]:
        pass

    def closing_positions(self, action: np.ndarray[Any, float | int]) -> pl.Series | None:
        pass

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        self._t = options["initial_timestep"]
        self.market.reset(self._t)
        self.cash_account.reset(self._t)
        self.portfolio.reset()
        self.predictive_model.reset(self._t)


    def render(self):
        pass