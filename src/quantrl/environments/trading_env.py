from __future__ import annotations
import quantrl as qrl
import numpy as np
from typing import Any, Dict, Iterable, Tuple
import polars as pl
import gymnasium as gym
from dataclasses import dataclass

@dataclass
class SingleAssetTradingEnv(qrl.BaseEnv):
    episode_length: int
    take_profit: float
    stop_loss: float
    horizon: int | None

    def __post_init__(self):
        super().__post_init__()
        assert self.predictive_model.symbols.size == 1
        symbol_id = self.market.market_data.select("symbol_id").unique().to_numpy()
        assert symbol_id.size == 1
        self._symbol_id = symbol_id.item()
        self._step: int | None = None
        self._previous_portfolio_value: float | None = None
        self._current_portfolio_value: float | None = None
        self.action_space = gym.spaces.Box(-1, 1, (1,))

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
                cash_account=self.cash_account,
                market=self.market,
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
        return np.log(self._current_portfolio_value / self._previous_portfolio_value)

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
        if action.item() >= 0:
            price = self.market.get_prices(side="BUY").to_dict()["price"]
        else:
            price = self.market.get_prices(side="SELL").to_dict()["price"]
        investment = action.item() * self.cash_account.current_capital
        self.cash_account.withdraw(investment)
        return [
                {
                "symbol_id": self._symbol_id,
                "position": investment / price,
                "entry_price": price,
                "strike_price": None,
                "contract": qrl.ContractType.SPOT,
                "maturity": self.horizon,
            }
        ]

    def closing_positions(self, action: np.ndarray[Any, float | int]) -> pl.Series | None:
        buy_prices = self.market.get_prices(side="BUY").rename({"price": "buy_price"})
        sell_prices = self.market.get_prices(side="SELL").rename({"price": "sell_price"})
    
        portfolio = (
            self.portfolio.open_positions.join(buy_prices, on="symbol_id").join(sell_prices, on="symbol_id")
        ).with_columns(pl.when(pl.col("position") < 0).then(pl.col("buy_price")).otherwise(pl.col("sell_price")).alias("price"))
        portfolio = portfolio.with_columns(
            (
                (pl.col("price") / pl.col("entry_price") - 1) 
                * pl.when(pl.col("position") >= 0).then(pl.lit(1.0)).otherwise(pl.lit(-1.0))
            ).alias("pnl")   
        )
        return (
            (portfolio.select("pnl").to_series() > self.take_profit) 
            | (portfolio.select("pnl").to_series() < self.stop_loss)
            | (portfolio.select("time_remaining") == 0)
        )