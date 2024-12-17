from abc import ABC
import gymnasium as gym
import polars as pl
import numpy as np
from typing import Any, Dict, Iterable
from dataclasses import dataclass
import quantrl as qrl

@dataclass
class BaseEnv(gym.Env):
    market: qrl.Market
    cash_account: qrl.CashAccount
    portfolio: qrl.PortfolioBase
    lags: int
    stride: int

    def __post_init__(self) -> None:
        self._t: int | None = None

    def step(self, action: np.ndarray[Any, float | int]):
        self._t += 1
        self.market.step()
        self.cash_account.step()
        self.portfolio.step()

        self.portfolio.close_positions(
            self.market.get_prices(),
            self.cash_account,
            closing_mask=self.closing_positions(action),
        )

        positions_to_open = self._process_action(action)
        for position in positions_to_open:
            self.portfolio.open_position(
                **position
            )
            

    def _process_action(self, action: np.ndarray[Any, float | int]) -> Iterable[Dict[str, float | int]]:
        pass

    def closing_positions(self, action: np.ndarray[Any, float | int]) -> pl.Series | None:
        pass

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        self._t = self.lags * self.stride
        self.market.reset(self._t)
        self.cash_account.reset(self._t)
        self.portfolio.reset()


    def render(self):
        pass