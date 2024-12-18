from __future__ import annotations
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
    portfolio: qrl.Portfolio
    predictive_model: qrl.PredictiveModel
    lags: int
    stride: int

    def __post_init__(self) -> None:
        self._t: int | None = None

    @property
    def state(self) -> Dict[str, pl.DataFrame | float]:
        model_state = pl.DataFrame(
            {
                f"{symbol_id}": self.predictive_model.predict(market_data=self.market.get_data(), symbol_id=symbol_id)
                for symbol_id in self.predictive_model.symbols
            }
        )
        return {
            "market": self.market.get_data(self.lags, self.stride),
            "portfolio": self.portfolio.open_positions,
            "cash_account": self.cash_account.current_capital,
            "predictive_model": model_state,
        }



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

        return self.state, 0, False, False, None

    

    def _process_action(self, action: np.ndarray[Any, float | int], cash_account: qrl.CashAccount) -> Iterable[Dict[str, float | int]]:
        pass

    def closing_positions(self, action: np.ndarray[Any, float | int]) -> pl.Series | None:
        pass

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        self._t = self.lags * self.stride
        self.market.reset(self._t)
        self.cash_account.reset(self._t)
        self.portfolio.reset()
        self.predictive_model.reset(self._t)


    def render(self):
        pass