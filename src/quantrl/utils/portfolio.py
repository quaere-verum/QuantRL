from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import quantrl as qrl
import polars as pl
from enum import Enum
from typing import Literal, Any, Tuple

class ContractType(Enum):
    SPOT = 1
    FUTURE = 2
    OPTION = 3

portfolio_schema = {
    "symbol_id": pl.Int16,
    "position": pl.Float64,
    "entry_price": pl.Float64,
    "strike_price": pl.Float64,
    "contract_id": pl.Int8,
    "maturity": pl.Int16,
    "margin_percent": pl.Float64,
    "market_id": pl.Int16,
}

@dataclass(kw_only=True)
class PositionData:
    symbol_id: int
    """
    The symbol_id for the position to open.
    """
    position: float
    """
    The position to take. Negative values are interpreted as shorting.
    """
    entry_price: float
    """
    The value of the contract per unit at the time the position is opened.
    """
    strike_price: float | None
    """
    The strike price in case of a futures contract or option.
    """
    contract_type: ContractType
    """
    The type of the contract.
    """
    maturity: int | None
    """
    The time to maturity for the contract in case of a futures contract or option.
    """
    margin_percent: float | None
    """
    In case of shortselling, specifies the required margin.
    """
    market_id: int
    """
    The market_id at which the position was entered.
    """

class Portfolio(ABC): 
    def __init__(self) -> None:
        self.open_positions = pl.DataFrame(
            schema=portfolio_schema
        )
        self._position_id: int | None = None

    def reset(self) -> None:
        """
        Reset the portfolio, to be used when resetting the reinforcement learning environment.
        """
        self.open_positions = pl.DataFrame(
            schema=portfolio_schema
        )
        self._position_id = 0 

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
        if position.contract_type in [ContractType.FUTURE, ContractType.OPTION]:
            assert position.strike_price is not None and position.maturity is not None
        if position.position >= 0:
            cash_account.withdraw(position.position * position.entry_price, account=qrl.AccountType.CASH)
        else:
            assert position.margin_percent is not None
            proceeds = -position.position * position.entry_price
            margin = -position.position * position.entry_price * position.margin_percent
            cash_account.withdraw(margin, account=qrl.AccountType.CASH)
            cash_account.deposit(margin + proceeds, account=qrl.AccountType.MARGIN)

        self.open_positions.extend(
            pl.DataFrame(
                {
                    "position_id": self._position_id,
                    "symbol_id": position.symbol_id,
                    "position": position.position,
                    "entry_price": position.entry_price,
                    "strike_price": position.strike_price,
                    "contract_id": position.contract_type.value,
                    "maturity": position.maturity,
                    "margin_percent": position.margin_percent,
                    "market_id": position.market_id,
                },
                schema=portfolio_schema
            )
        )
        self._position_id += 1

    def close_positions(
        self,
        closing_positions: pl.DataFrame | None,
        cash_account: qrl.CashAccount,
    ) -> None:
        """
        Close the specified positions.

        Parameters
        ----------
        closing_positions : pl.DataFrame
            Dataframe with the columns `position_id` and `closing_price`, specifying which positions to close and
            at what price.
        cash_account : qrl.CashAccount
            Cash account which manages funding.
        """
        if len(self.open_positions) == 0 or len(closing_positions) == 0 or closing_positions is None:
            return
        else:
            assert "position_id" in closing_positions.columns and "closing_price" in closing_positions.columns
            # Get the data from the closing positions
            closing_positions = closing_positions.join(self.open_positions, on="position_id", how="left")
            
            # All long positions are closed by simply selling the assets
            closing_long_positions = closing_positions.filter(pl.col("position") >= 0)
            closing_long_value = value_portfolio(closing_long_positions, contract_type=None)
            cash_account.deposit(closing_long_value, account=qrl.AccountType.CASH)
            
            # Short positions are closed by buying back the asset with cash from the margin
            # account. The remainder is the P&L which is added back into the cash account.
            closing_short_positions = closing_positions.filter(pl.col("position") < 0)

            # TODO: Maybe make margin_percent update each step based on current price, to accomodate margin calls.
            # Current setup only works if no additional margin needs to be deposited.
            closing_margin_account_balance = (
                -closing_short_positions.select("position").to_series()
                * closing_short_positions.select("entry_price").to_series()
                * (closing_short_positions.select("margin_percent").to_series() + 1)
            ).sum()
            closing_short_asset_buyback_value = -value_portfolio(closing_short_positions, contract_type=None)
            pnl_plus_margin = closing_margin_account_balance - closing_short_asset_buyback_value
            cash_account.withdraw(closing_margin_account_balance, account=qrl.AccountType.MARGIN)
            cash_account.deposit(pnl_plus_margin, account=qrl.AccountType.CASH)

            self.open_positions = self.open_positions.filter(~pl.col("position_id").is_in(closing_positions.select("position_id")))

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
    contract_type: ContractType | None = None,
) -> float:
    """
    Calculate the value of the portfolio given the current market prices.

    Parameters
    ----------
    portfolio : pl.DataFrame
        Portfolio object to be valuated. Should contain the columns `closing_price` in addition to the 
        standard portfolio dataframe columns (see qrl.utils.portfolio.portfolio_schema).
    contract_type : Literal[&quot;SPOT&quot;, &quot;FUTURE&quot;, &quot;OPTION&quot;] | None, optional
        The contract type contained in the portfolio. If None, assumes mixed contract type.

    Returns
    -------
    float
        The value of the portfolio.
    """
    assert "closing_price" in portfolio.columns
    if len(portfolio) == 0:
        return 0
    if contract_type is None:
        contract_ids = portfolio.select("contract_id").unique().to_numpy()
        return sum(
            [
                value_portfolio(
                    portfolio.filter(pl.col("contract_id") == contract_id),
                    contract_type=ContractType(contract_id)
                )
                for contract_id in contract_ids
            ]
        )
    else:
        match contract_type:
            case ContractType.SPOT:
                return (
                    portfolio.select("position").to_series()
                    * portfolio.select("closing_price").to_series()
                ).sum()
            case ContractType.FUTURE:
                # TODO: Implement valuation for futures contract
                raise NotImplementedError()
            case ContractType.OPTION:
                # TODO: Implement valuation for options contract
                raise NotImplementedError()
