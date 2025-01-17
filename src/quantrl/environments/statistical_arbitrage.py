import quantrl as qrl
import numpy as np
from dataclasses import dataclass, field
import gymnasium as gym
from typing import Tuple, Dict, Any, Iterable
import polars as pl


@dataclass
class StatArbEnv(qrl.BaseEnv):
    episode_length: int
    take_profit: float
    stop_loss: float
    horizon: int | None
    margin_percent: float

    def __post_init__(self):
        super().__post_init__()
        self._symbol_ids = self.market.get_all_data().select("symbol_id").unique().to_numpy().flatten()
        self._symbol_ids.sort()
        self._previous_portfolio_value: float | None = None
        self._current_portfolio_value: float | None = None
        # Action space dim is the number of assets plus the risk free asset
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(len(self._symbol_ids),), dtype=float
        )

    def reset(self) -> tuple[dict[str, np.ndarray[float]]]:
        self._current_portfolio_value = self.cash_account.current_balance(qrl.AccountType.CASH)
        return super().reset()

    def step(self, action: np.ndarray[float]) -> tuple[dict[str, np.ndarray[float]], float, bool, bool, dict[str, Any]]:
        self._t += 1
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
        self._current_portfolio_value = (
            qrl.value_portfolio(
                portfolio=self.portfolio.open_positions,
                market=self.market,
                contract_type=None,
                apply_bid_ask_spread=True,
            ) 
            + self.cash_account.current_balance(qrl.AccountType.CASH)
            + self.cash_account.current_balance(qrl.AccountType.SHORT)
            + self.cash_account.current_balance(qrl.AccountType.MARGIN)
        )
        return self.state, self.reward, self.done, self.truncated, self.info
    
    @property
    def state(self) -> Dict[str, np.ndarray[Any, float]]:
        market_data_pivot: pl.DataFrame = self.market.get_current_data(
            self.lags, 
            self.stride, 
            columns=self.market_observation_columns + ["market_id", "symbol_id"]
        ).pivot(index="market_id", columns="symbol_id", values=self.market_observation_columns)
        return {
            "market": market_data_pivot.to_numpy(),
            "portfolio": self.portfolio.summarise_positions(self.market),
            "cash_account": np.array(
                [
                    self.cash_account.current_balance(qrl.AccountType.CASH),
                    self.cash_account.current_balance(qrl.AccountType.SHORT),
                    self.cash_account.current_balance(qrl.AccountType.MARGIN),
                ]
            ),
            "predictive_model": np.array(
                [
                    self.predictive_model.predict(self.market, symbol_id)
                    for symbol_id in self.predictive_model.symbols
                ] 
                + [self.predictive_model.performance]
            )
        }

    @property
    def reward(self) -> float:
        # TODO: Significantly refine the reward to specialise to the present context
        return np.log(self._current_portfolio_value / self._previous_portfolio_value)

    @property
    def done(self) -> bool:
        return (
            True
            if self._t >= self.episode_length
            else False
        )

    @property
    def truncated(self) -> bool:
        return (
            True 
            if self._current_portfolio_value <= 0
            else False
        )
        

    @property
    def info(self) -> Dict[str, Any]:
        return dict()
    

    def _process_action(self, action: np.ndarray[float]) -> Iterable[qrl.PositionData]:
        positions = []
        buy_prices = price = self.market.get_prices(side=qrl.OrderType.BUY)
        sell_prices = self.market.get_prices(side=qrl.OrderType.SELL)
        cash_distribution = (
            self.cash_account.current_balance(qrl.AccountType.CASH)
            * np.abs(action) / max(np.abs(action).sum(), 1e-8)
            * np.sign(action)
        )
        for investment, symbol_id in zip(cash_distribution, self._symbol_ids):
            if np.isclose(investment, 0):
                continue
            elif investment > 0:
                price = buy_prices.filter(pl.col("symbol_id") == symbol_id).select("price").item()
            else:
                price = sell_prices.filter(pl.col("symbol_id") == symbol_id).select("price").item()
            positions.append(
                qrl.PositionData(
                    symbol_id=symbol_id,
                    position=investment / price,
                    entry_price=price,
                    strike_price=None,
                    contract_type=qrl.ContractType.SPOT,
                    maturity=None,
                    margin_percent=self.margin_percent if investment < 0 else None,
                    market_id=self.market.market_id
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
            ).with_columns(
                pl.when(pl.col("position") < 0)
                .then(pl.col("buy_price"))
                .otherwise(pl.col("sell_price"))
                .alias("price")
            )
            portfolio = (
                portfolio
                .with_columns(
                    (
                        (pl.col("price") / pl.col("entry_price") - 1) 
                        * pl.when(pl.col("position") >= 0).then(pl.lit(1.0)).otherwise(pl.lit(-1.0))
                    )
                    .alias("pnl")   
                )
                .with_columns(
                    (
                        pl.col("pnl") * pl.col("position").abs()
                    )
                    .alias("weighted_pnl")
                )
            )

            trade_pnl = (
                portfolio
                .group_by("market_id")
                .agg(
                    (pl.col("weighted_pnl").sum() / pl.col("position").abs().sum()).alias("vw_pnl")
                )
                .with_columns(
                    (~pl.col("vw_pnl").is_between(self.stop_loss, self.take_profit)).alias("closing_mask")
                )
            )

            portfolio = portfolio.join(trade_pnl.select(["market_id", "closing_mask"]), on="market_id")

            return portfolio.select("closing_mask").to_series()