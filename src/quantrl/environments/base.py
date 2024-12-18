from __future__ import annotations
import gymnasium as gym
import polars as pl
import numpy as np
from typing import Any, Dict, Iterable
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

    def __post_init__(self) -> None:
        self._t: int | None = None
        self.observation_space = gym.spaces.Dict(
            {
                "market": gym.spaces.Box(0, np.inf, (self.lags + 1, len(self.market.market_data.columns))),
                "portfolio": gym.spaces.Box(-np.inf, np.inf, self.portfolio.summary_shape),
                "cash_account": gym.spaces.Box(-np.inf, np.inf, (1,)),
                "predictive_model": gym.spaces.Box(-np.inf, np.inf, (len(self.predictive_model.symbols),))
            }
        )
        self.action_space: gym.spaces.Space = None

    @property
    def state(self) -> Dict[str, np.ndarray[Any, float]]:
        return {
            "market": self.market.get_data(self.lags, self.stride).to_numpy(),
            "portfolio": self.portfolio.summarise_positions(market=self.market),
            "cash_account": np.array([self.cash_account.current_capital]),
            "predictive_model": np.array(
                [
                    self.predictive_model.predict(market=self.market, symbol_id=symbol_id)
                    for symbol_id in self.predictive_model.symbols
                ]
            ),
        }

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


    def step(self, action: np.ndarray[Any, float | int]):
        self._t += 1
        self.market.step()
        self.cash_account.step()
        self.portfolio.step()
        self.predictive_model.step()

        self.portfolio.close_positions(
            self.market.get_prices(),
            self.cash_account,
            closing_mask=self.closing_positions(action=action),
        )

        positions_to_open = self._process_action(action, self.cash_account)
        for position in positions_to_open:
            self.portfolio.open_position(
                **position
            )

        return self.state, self.reward, self.done, self.truncated, self.info

    

    def _process_action(self, action: np.ndarray[Any, float | int], cash_account: qrl.CashAccount) -> Iterable[Dict[str, float | int]]:
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