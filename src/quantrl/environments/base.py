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
    """
    Market object used to simulate a market evolving over time.
    """
    cash_account: qrl.CashAccount
    """
    Cash account object used to manage transactions within the environment.
    """
    portfolio: qrl.Portfolio
    """
    Portfolio object used to open and close positions within the environment.
    """
    predictive_model: qrl.PredictiveModel
    """
    PredictiveModel object used to generate signals within the environment.
    """
    lags: int
    """
    How many lagged values of market data to return for each observation.
    """
    stride: int
    """
    How many timesteps between the lagged values of market data.
    """
    market_observation_columns: List[str]
    """
    Which columns of the market data to use for observations.
    """

    def __post_init__(self) -> None:
        self._t: int | None = None
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

    

    def _process_action(self, action: np.ndarray[Any, float | int]) -> Iterable[qrl.PositionData]:
        """
        Given the agent's action, specify the positions to be opened.

        Parameters
        ----------
        action : np.ndarray[Any, float  |  int]
            The action taken by the reinforcement learning agent.

        Returns
        -------
        Iterable[qrl.PositionData]
            The positions to be opened.
        """
        pass

    def closing_positions(self, action: np.ndarray[Any, float | int]) -> pl.Series | None:
        """
        Some logic (possibly based on the agent's chosen action) that specifies which of the
        open positions in self.portfolio.open_positions to close.

        Parameters
        ----------
        action : np.ndarray[Any, float  |  int]
            The action taken by the reinforcement learning agent.

        Returns
        -------
        pl.Series | None
            Polars series containing booleans, of the same length as self.portfolio.open_positions. If true, the corresponding position will be closed.
        """
        pass

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        self._t = options.get("initial_timestep", 0)
        self.market.reset(self._t)
        self.cash_account.reset(self._t)
        self.portfolio.reset()
        self.predictive_model.reset(self._t)


    def render(self):
        pass