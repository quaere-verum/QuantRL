from __future__ import annotations
import quantrl as qrl
import numpy as np
from typing import Any, Dict, Iterable, Tuple
import polars as pl
import gymnasium as gym
from dataclasses import dataclass

@dataclass
class TradingEnv(qrl.BaseEnv):
    action_shape: Tuple[int, ...]
    action_size: int | None
    episode_length: int

    def __post_init__(self):
        super().__post_init__()
        self._step: int | None = None
        self._previous_portfolio_value: float | None = None
        self._current_portfolio_value: float | None = None
        if self.action_size is not None:
            assert len(self.action_shape) == 1
            self.action_space = gym.spaces.Discrete(self.action_size)
        else:
            self.action_space = gym.spaces.Box(-np.inf, np.inf, self.action_shape)

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)
        self._step = 0
        self._current_portfolio_value = self.cash_account.current_capital

    def step(self, action: np.ndarray[Any, float | int]) -> Tuple[Dict[str, np.ndarray[Any, float]], float, bool, bool, Dict[str, Any]]:
        self._t += 1
        self._step += 1
        self.market.step()
        self.cash_account.step()
        self.portfolio.step()
        self.predictive_model.step()

        self.portfolio.close_positions(
            self.market,
            self.cash_account,
            closing_mask=self.closing_positions(action=action),
        )

        positions_to_open = self._process_action(action)
        for position in positions_to_open:
            self.portfolio.open_position(
                **position
            )
        self._previous_portfolio_value = self._current_portfolio_value
        self._current_portfolio_value = qrl.value_portfolio(
            self.portfolio.open_positions,
            self.market,
            None,
            True,
        ) + self.cash_account.current_capital

        return self.state, self.reward, self.done, self.truncated, self.info

    @property
    def state(self) -> Dict[str, np.ndarray[Any, float]]:
        return {
            "market": self.market.get_data(self.lags, self.stride, columns=self.market_observation_columns).to_numpy(),
            "portfolio": self.portfolio.summarise_positions(self.market),
            "cash_account": np.array([self.cash_account.current_capital]),
            "predictive_model": np.array(
                [
                    self.predictive_model.predict(self.market, symbol_id)
                    for symbol_id in self.predictive_model.symbols
                ] + [self.predictive_model.performance]
            )
        }

    @property
    def reward(self) -> float:
        return self._current_portfolio_value - self._previous_portfolio_value

    @property
    def done(self) -> bool:
        return (
            True 
            if qrl.value_portfolio(self.portfolio.open_positions, self.market, None, True)
            + self.cash_account.current_capital < 0
            else False
        )

    @property
    def truncated(self) -> bool:
        return (
            True
            if self._step >= self.episode_length
            else False
        )

    @property
    def info(self) -> Dict[str, Any]:
        return dict()
    

    def _process_action(self, action: np.ndarray[Any, float | int]) -> Iterable[Dict[str, float | int]]:
        buy_prices = self.market.get_prices(side="BUY").to_dict()
        sell_prices = self.market.get_prices(side="SELL").to_dict()
        return [
                {
                "symbol_id": 0,
                "position": action.item(),
                "entry_price": buy_prices["price"].item() if action >= 0 else sell_prices["price"].item(),
                "strike_price": None,
                "contract": "SPOT",
                "maturity": None,
            }
        ]

    def closing_positions(self, action: np.ndarray[Any, float | int]) -> pl.Series | None:
        return None