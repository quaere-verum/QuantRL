import numpy as np
from abc import ABC, abstractmethod
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
    "contract_id": pl.Int8,
    "maturity": pl.Int8,
    "time_remaining": pl.Int8
}

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
        self.open_positions = pl.DataFrame(
            schema=portfolio_schema
        )        

    def open_position(
        self,
        symbol_id: int,
        position: float,
        entry_price: float,
        contract: Literal["SPOT", "FUTURE", "OPTION"],
        maturity: int,
    ) -> None:
        self.open_positions.extend(
            pl.DataFrame(
                {
                    "symbol_id": symbol_id,
                    "position": position,
                    "entry_price": entry_price,
                    "contract_id": get_contract_id(contract),
                    "maturity": maturity,
                    "time_remaining": maturity
                },
                schema=portfolio_schema
            )
        )

    def step(self) -> None:
        self.open_positions._replace("time_remaining", self.open_positions.select("time_remaining").to_series() - 1)

    def close_positions(
        self,
        market: qrl.Market,
        cash_account: qrl.CashAccount,
        *,
        closing_mask: pl.Series | None = None,
    ) -> None:
        if len(self.open_positions) == 0:
            return
        else:
            closing_positions = self.open_positions.filter(closing_mask or self.closing_mask)
            closing_value = value_portfolio(closing_positions, market, contract_type=None)
            self.open_positions = self.open_positions.filter(~(closing_mask or self.closing_mask))
            cash_account.deposit(closing_value)

    @property
    @abstractmethod
    def closing_mask(self) -> pl.Series:
        pass

    @abstractmethod
    def summarise_positions(self, market: qrl.Market) -> np.ndarray[Any, float]:
        pass

    @property
    @abstractmethod
    def summary_shape(self) -> Tuple[int, ...]:
        pass


def value_portfolio(
    portfolio: pl.DataFrame, 
    market: qrl.Market,
    contract_type: Literal["SPOT", "FUTURE", "OPTION"] | None = None,
    apply_bid_ask_spread: bool = True,
) -> float:
    if len(portfolio) == 0:
        return 0
    if contract_type is None:
        contract_ids = portfolio.select("contract_id").unique()
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
                return (
                    portfolio.select("position").to_series()
                    * portfolio.select("price").to_series()
                ).sum()
            case "OPTION":
                raise NotImplementedError()

class InvestmentPortfolio(Portfolio):

    @property
    def closing_mask(self) -> pl.Series:
        return pl.Series(name="closing_mask", values=[False for _ in range(len(self.open_positions))])
    
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