import quantrl as qrl
import numpy as np
from dataclasses import dataclass
import gymnasium as gym
from typing import Any, Iterable
import polars as pl


@dataclass
class StatArbEnv(qrl.BaseEnv):
    """
    Reinforcement learning environment for statistical arbitrage. Positions are aggregated by the market_id on which they
    were opened, and the PnL of the aggregated position is used to determine whether or not to close the positions. E.g. in
    case of pairs trading, the PnL of both legs of the trade are aggregated, and then the corresponding positions are 
    closed if a stop loss or take profit threshold is reached.
    """
    max_episode_length: int
    take_profit: float
    stop_loss: float
    horizon: int | None
    margin_percent: float

    def __post_init__(self):
        super().__post_init__()
        self._symbol_ids = self.market.get_all_data().select("symbol_id").unique().to_numpy().flatten()
        self._symbol_ids.sort()
        # Action space dim is the number of assets plus the risk free asset
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(len(self._symbol_ids),), dtype=float
        )
        self._final_timestep_id = self.market.get_all_data().select("timestep_id").to_numpy().max()
        self._previous_market_id: int | None = None
        self._previous_value: float | None = None
        self._current_value: float | None = None

    def reset(self) -> tuple[dict[str, np.ndarray[float]]]:
        obs, info = super().reset()
        self._current_value = self.cash_account.current_balance(qrl.AccountType.CASH)
        return obs, info

    def step(self, action: np.ndarray[float]) -> tuple[dict[str, np.ndarray[float]], float, bool, bool, dict[str, Any]]:
        self._t += 1
        self._previous_market_id = self.market.market_id
        self.cash_account.step()
        self.market.step()
        self.portfolio.step()
        self.predictive_model.step()

        self.portfolio.close_positions(
            closing_positions=self.closing_positions(action),
            cash_account=self.cash_account,
        )

        positions_to_open = self._process_action(action)
        for position in positions_to_open:
            self.portfolio.open_position(
                cash_account=self.cash_account,
                position=position,
            )

        portfolio_value = qrl.value_portfolio(
                portfolio=(
                    self.portfolio.open_positions
                    .join(
                        self.market.get_prices().rename({"price": "closing_price"}),
                        on="symbol_id"
                    )
                ),

            )
        self.portfolio.set_value(self._t, portfolio_value)
        self._previous_value = self._current_value
        self._current_value = (
            portfolio_value
            + self.cash_account.current_balance(qrl.AccountType.CASH)
            + self.cash_account.current_balance(qrl.AccountType.MARGIN)
        )
        
        return self.state, self.reward, self.done, self.truncated, self.info
    
    @property
    def state(self) -> dict[str, np.ndarray[Any, float]]:
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
        return np.log(self._current_value / self._previous_value)

    @property
    def done(self) -> bool:
        return (
            True
            if self._t >= min(self.max_episode_length, self._final_timestep_id)
            else False
        )

    @property
    def truncated(self) -> bool:
        return (
            True 
            if self._current_value <= 0
            else False
        )

    @property
    def info(self) -> dict[str, Any]:
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

    def closing_positions(self, action: np.ndarray[Any, float | int]) -> pl.DataFrame | None:
        if len(self.portfolio.open_positions) == 0:
            return None
        else:
            # Get the prices between the previous environment step and the current step
            price_evolution = (
                self.market
                .get_current_data(
                    lags=self.market.market_id - self._previous_market_id, 
                    columns=["market_id", "symbol_id", "midprice"]
                )
                .rename({"market_id": "trajectory_market_id"})
            )
            # For each open position, calculate the volume-weighted pnl
            portfolio = (
                self.portfolio.open_positions
                .join(price_evolution, on="symbol_id")
                .with_columns(
                    pl.when(pl.col("position") > 0).then(1.0).otherwise(-1.0).alias("side")
                )
                .with_columns(
                    (pl.col("midprice") * (1 - pl.col("side") * self.market.bid_ask_spread * 1e-4)).alias("closing_price")
                )
                .with_columns(
                    (pl.col("side") * (pl.col("closing_price") - pl.col("entry_price")) / pl.col("entry_price")).alias("pnl")
                )
                .with_columns(
                    (pl.col("pnl") * pl.col("position").abs()).alias("weighted_pnl")
                )
            )
            # Aggregate the volume-weighted pnl to market_id level (i.e. aggregate each leg of the trade)
            trade_pnl = (
                portfolio
                .group_by("market_id", "trajectory_market_id")
                .agg((pl.col("weighted_pnl").sum() / (pl.col("position").abs().sum())).alias("trade_pnl"))
            )
            # Determine which trades exceed the thresholds
            profit_or_loss = trade_pnl.filter((pl.col("trade_pnl") < self.stop_loss) | (pl.col("trade_pnl") > self.take_profit))
            # Determine the times at which the trades were closed, and at what price
            closing_trades = (
                profit_or_loss
                .group_by("market_id")
                .agg(pl.min("trajectory_market_id"))
            )
            # Determine which positions were closed, and at what price
            closing_positions = (
                closing_trades
                .join(
                    self.portfolio.open_positions.select("position_id", "market_id", "symbol_id"),
                    on="market_id",
                    how="left",
                )
                .drop("market_id")
                .join(
                    portfolio.select("symbol_id", "trajectory_market_id", "closing_price").unique(),
                    on=["symbol_id", "trajectory_market_id"],
                    how="left",
                )
                .drop("trajectory_market_id", "symbol_id")
            )

            return closing_positions