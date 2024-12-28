from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
import quantrl as qrl
import polars as pl
from enum import Enum
from typing import Literal, Any, Tuple

class ContractType(Enum):
    SPOT = 1
    FUTURE = 2
    OPTION = 3

portfolio_schema = {
    "symbol_id": pl.Int8,
    "position": pl.Float32,
    "entry_price": pl.Float32,
    "strike_price": pl.Float32,
    "contract_id": pl.Int8,
    "maturity": pl.Int8,
    "time_remaining": pl.Int8
}

@dataclass
class PositionData:
    symbol_id: int
    position: float
    entry_price: float
    strike_price: float | None
    contract_type: Literal["SPOT", "FUTURE", "OPTION"]
    maturity: int | None

def get_contract_id(
    name: Literal["SPOT", "FUTURE", "OPTION"]
) -> int:
    match name:
        case "SPOT":
            return ContractType.SPOT.value
        case "FUTURE":
            return ContractType.FUTURE.value
        case "OPTION":
            return ContractType.OPTION.value
        case _:
            raise ValueError(f"Contract name '{name}' unknown.")

class Portfolio(ABC):
    def __init__(self) -> None:
        self.open_positions = pl.DataFrame(
            schema=portfolio_schema
        )

    def reset(self) -> None:
        """
        Reset the portfolio, to be used when resetting the reinforcement learning environment.
        """
        self.open_positions = pl.DataFrame(
            schema=portfolio_schema
        )        

    def open_position(
        self,
        cash_account: qrl.CashAccount,
        position: PositionData,
    ) -> None:
        """
        Opens the specified position, if possible.

        Parameters
        ----------
        cash_account : qrl.CashAccount
            Cash account which manages the funding.
        position : PositionData
            Dataclass which contains the information required to open the position.
        """
        contract_id = get_contract_id(position.contract_type)
        if position.contract_type in ["FUTURE", "OPTION"]:
            assert position.strike_price is not None and position.maturity is not None
        self.open_positions.extend(
            pl.DataFrame(
                {
                    "symbol_id": position.symbol_id,
                    "position": position.position,
                    "entry_price": position.entry_price,
                    "strike_price": position.strike_price,
                    "contract_id": contract_id,
                    "maturity": position.maturity,
                    "time_remaining": position.maturity,
                },
                schema=portfolio_schema
            )
        )

    def step(self) -> None:
        """
        Evolve the portfolio by one timestep, to be used when the reinforcement learning environment takes a step.
        """
        self.open_positions._replace("time_remaining", self.open_positions.select("time_remaining").to_series() - 1)

    def close_positions(
        self,
        market: qrl.Market,
        cash_account: qrl.CashAccount,
        *,
        closing_mask: pl.Series | None = None,
    ) -> None:
        """
        Close the specified positions.

        Parameters
        ----------
        market : qrl.Market
            Market object that can be used to derive closing criteria.
        cash_account : qrl.CashAccount
            Cash account which manages funding.
        closing_mask : pl.Series | None, optional
            Boolean mask that specifies which positions to close, by default None. If None, use self.closing_positions.
        """
        if len(self.open_positions) == 0:
            return
        else:
            closing_mask = closing_mask or self.closing_mask(market)
            closing_positions = self.open_positions.filter(closing_mask)
            closing_value = value_portfolio(closing_positions, market, contract_type=None)
            self.open_positions = self.open_positions.filter(~closing_mask)
            cash_account.deposit(closing_value)

    @abstractmethod
    def closing_mask(self, market: qrl.Market) -> pl.Series:
        """
        Determines which positions to close, possibly based on market data.

        Parameters
        ----------
        market : qrl.Market
            Market object from which to derive closing conditions.

        Returns
        -------
        pl.Series
            Boolean mask that specifies which conditions to close.
        """
        pass

    @abstractmethod
    def summarise_positions(self, market: qrl.Market) -> np.ndarray[Any, float]:
        """
        Summarise the current open positions and return a numpy array, to be used as part of
        the observation for the reinforcement learning agent.

        Parameters
        ----------
        market : qrl.Market
            Market object to be used for summarising the position data.

        Returns
        -------
        np.ndarray[Any, float]
            The summary of the position data.
        """
        pass

    @property
    @abstractmethod
    def summary_shape(self) -> Tuple[int, ...]:
        """
        The shape of the self.summarise_positions functions. Used to create the observation space 
        of the reinforcement learning environment.

        Returns
        -------
        Tuple[int, ...]
            The shape of the position summary.
        """
        pass


def value_portfolio(
    portfolio: pl.DataFrame, 
    market: qrl.Market,
    contract_type: Literal["SPOT", "FUTURE", "OPTION"] | None = None,
    apply_bid_ask_spread: bool = True,
) -> float:
    """
    Calculate the value of the portfolio in the given market.

    Parameters
    ----------
    portfolio : pl.DataFrame
        Portfolio object to be valuated.
    market : qrl.Market
        Market object to be used for deriving the portfolio value.
    contract_type : Literal[&quot;SPOT&quot;, &quot;FUTURE&quot;, &quot;OPTION&quot;] | None, optional
        The contract type contained in the portfolio. If None, assumes mixed contract type.
    apply_bid_ask_spread : bool, optional
        Whether to apply the bid ask spread for valuation, by default True.

    Returns
    -------
    float
        The value of the portfolio.
    """
    if len(portfolio) == 0:
        return 0
    if contract_type is None:
        contract_ids = portfolio.select("contract_id").unique().to_numpy()
        return sum(
            [
                value_portfolio(
                    portfolio.filter(pl.col("contract_id") == contract_id),
                    market,
                    contract_type=ContractType(contract_id)
                )
                for contract_id in contract_ids
            ]
        )
    else:
        buy_prices = market.get_prices(side="BUY" if apply_bid_ask_spread else None).rename({"price": "buy_price"})
        sell_prices = market.get_prices(side="SELL" if apply_bid_ask_spread else None).rename({"price": "sell_price"})
    
        portfolio = (
            portfolio.join(buy_prices, on="symbol_id").join(sell_prices, on="symbol_id")
        ).with_columns(pl.when(pl.col("position") < 0).then(pl.col("buy_price")).otherwise(pl.col("sell_price")).alias("price"))
        match contract_type:
            case "SPOT":
                return (
                    portfolio.select("position").to_series()
                    * portfolio.select("price").to_series()
                ).sum()
            case "FUTURE":
                # TODO: Implement valuation for futures contract
                raise NotImplementedError()
            case "OPTION":
                # TODO: Implement valuation for options contract
                raise NotImplementedError()

class TripleBarrierPortfolio(Portfolio):
    def __init__(self, model: qrl.TripleBarrierClassifier):
        super().__init__()
        self.model = model

    # TODO: add Greeks/risk metrics to position summary
    def summarise_positions(self, market: qrl.Market):
        return np.array(
            [
                value_portfolio(self.open_positions, market, None, True)
            ]
        )

    @property
    def summary_shape(self) -> Tuple[int, ...]:
        return (1,)

    def _position_returns(self, market: qrl.Market) -> pl.Series:
        pass

    def closing_mask(self, market: qrl.Market) -> pl.Series:
        buy_prices = market.get_prices(side="BUY").rename({"price": "buy_price"})
        sell_prices = market.get_prices(side="SELL").rename({"price": "sell_price"})
    
        portfolio = (
            self.open_positions.join(buy_prices, on="symbol_id").join(sell_prices, on="symbol_id")
        ).with_columns(pl.when(pl.col("position") < 0).then(pl.col("buy_price")).otherwise(pl.col("sell_price")).alias("price"))
        portfolio = portfolio.with_columns(
            pl.when(
                pl.col("contract_id") == get_contract_id("SPOT")
            ).then(
                (pl.col("price") / pl.col("entry_price") - 1) 
                * pl.when(pl.col("position") >= 0).then(pl.lit(1.0)).otherwise(pl.lit(-1.0))
            ).when(
                pl.col("contract_id") == get_contract_id("FUTURE")
            ).then(
                None
            ).when(
                pl.col("contract_id") == get_contract_id("OPTION")
            ).then(
                None
            ).alias("pnl")
        )
        return (portfolio.select("pnl").to_series() > self.model.take_profit) | (portfolio.select("pnl").to_series() < self.model.stop_loss)
