from __future__ import annotations
import quantrl as qrl
import numpy as np
from typing import Any, Dict, Iterable, Tuple
import polars as pl
import gymnasium as gym
from dataclasses import dataclass

@dataclass
class TradingEnv(qrl.BaseEnv):
    episode_length: int
    take_profit: float
    stop_loss: float
    horizon: int | None

    def __post_init__(self):
        super().__post_init__()
        self._symbol_ids = self.market.get_all_data().select("symbol_id").unique().to_numpy().flatten()
        self._symbol_ids.sort()
        self._step: int | None = None
        self._previous_portfolio_value: float | None = None
        self._current_portfolio_value: float | None = None
        # Action space dim is the number of assets plus the risk free asset (cash)
        self.action_space = gym.spaces.Box(-1, 1, (self._symbol_ids.size + 1,))

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)
        timstep_id_end = self._t + self.episode_length
        assert timstep_id_end <= self.market.get_all_data(columns=["timestep_id"]).to_series().max(), (
            f"Not enough data to start at timestep_id={self._t} with episode length {self.episode_length}."
        )
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
                position=position,
            )
        self._previous_portfolio_value = self._current_portfolio_value
        self._current_portfolio_value = qrl.value_portfolio(
            portfolio=self.portfolio.open_positions,
            market=self.market,
            contract_type=None,
            apply_bid_ask_spread=True,
        ) + self.cash_account.current_capital

        return self.state, self.reward, self.done, self.truncated, self.info

    @property
    def state(self) -> Dict[str, np.ndarray[Any, float]]:
        return {
            "market": self.market.get_current_data(
                self.lags, 
                self.stride, 
                columns=self.market_observation_columns + ["market_id", "symbol_id"]
            )
            .to_pandas()
            .pivot(index="market_id", columns="symbol_id", values=self.market_observation_columns)
            .values,
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
            if self._current_portfolio_value <= 0
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
    

    def _process_action(self, action: np.ndarray[Any, float | int]) -> Iterable[qrl.PositionData]:
        positions = []
        buy_prices = price = self.market.get_prices(side=qrl.OrderType.BUY)
        sell_prices = self.market.get_prices(side=qrl.OrderType.SELL)
        for position, symbol_id in zip(action[1:], self._symbol_ids):
            if position >= 0:
                price = buy_prices.filter(pl.col("symbol_id") == symbol_id).select("price").item()
            else:
                price = sell_prices.filter(pl.col("symbol_id") == symbol_id).select("price").item()
            investment = position * self.cash_account.current_capital
            positions.append(
                qrl.PositionData(
                    symbol_id=symbol_id,
                    position=investment / price,
                    entry_price=price,
                    strike_price=None,
                    contract_type=qrl.ContractType.SPOT,
                    maturity=None,
                )
            )
        return positions

    def closing_positions(self, action: np.ndarray[Any, float | int]) -> pl.Series | None:
        if len(self.portfolio.open_positions) == 0:
            return None
        else:
            buy_prices = self.market.get_prices(side=qrl.OrderType.BUY).rename({"price": "buy_price"})
            sell_prices = self.market.get_prices(side=qrl.OrderType.SELL).rename({"price": "sell_price"})
        
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
                | (portfolio.select("time_remaining").to_series() == 0).fill_null(False)
            )