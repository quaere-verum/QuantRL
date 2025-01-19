from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
import polars as pl
import numpy as np
from typing import Any, List
from enum import Enum
from quantrl.markets.simulation import MarketSimulator

class OrderType(Enum):
    BUY = 1
    SELL = 2

@dataclass
class Market(ABC):
    bid_ask_spread: int
    """
    The market's bid ask spread in basis points.
    """

    def __post_init__(self) -> None:
        all_market_data = self.get_all_data()
        assert np.isin(
            ["timestep_id", "market_id", "symbol_id", "midprice"],
            all_market_data.columns
        ).all(), (
            "spot_price dataframe must contain columns with timestep_id, market_id, symbol_id"
        )
        assert (
            all_market_data
            .group_by("market_id", "symbol_id")
            .agg(pl.col("market_id").count().alias("pk_count"))
            .select("pk_count")
            .to_series()
            .max() == 1
        ), ("The combination (market_id, symbol_id) should be a primary key.")
        timestep_id = all_market_data.select("timestep_id").drop_nulls().unique().to_numpy().flatten()
        timestep_id.sort()
        assert timestep_id.min() == 0, "timestep_id should start at 0."
        assert np.all(np.diff(timestep_id) == 1), "timestep_id should contain consecutive integer values."
        market_id = all_market_data.select("market_id").unique().to_numpy().flatten()
        market_id.sort()
        assert market_id.min() == 0, "market_id should start at 0."
        assert np.all(np.diff(market_id) == 1), "market_id should contain consecutive integer values."
        assert all_market_data.schema["market_id"].is_integer()
        assert all_market_data.schema["timestep_id"].is_integer()
        assert all_market_data.schema["symbol_id"].is_integer()
        for col in all_market_data.columns:
            if col not in ["timestep_id", "market_id", "symbol_id"]:
                assert all_market_data.schema[col].is_numeric()

        
        self._all_symbols = all_market_data.select("symbol_id").unique().to_numpy().flatten()
        self._t: int | None = None
        self.market_id: int | None = None

    def reset(self) -> None:
        """
        Reset the market, to be used when resetting the reinforcement learning environment.
        """
        self._t = 0
        self.market_id = int(self.market_data.filter(pl.col("timestep_id") == self._t).select("market_id").max().item())

    def step(self) -> None:
        """
        Evolve the market by one timestep, to be used when the reinforcement learning environment takes a step.
        """
        self._t += 1
        self.market_id = int(self.market_data.filter(pl.col("timestep_id") == self._t).select("market_id").max().item())

    def get_all_data(
        self, 
        symbol_id: int | np.ndarray[Any, int] | None = None, 
        columns: List[str] | None = None
    ) -> pl.DataFrame:
        """
        Returns all market data (including future market data).

        Parameters
        ----------
        symbol_id : int | np.ndarray[Any, int] | None, optional
            Which symbol_ids to retrieve the data for, by default None. If None, returns the market data for all symbols.
        columns : List[str] | None, optional
            Which columns to retrieve, by default None. If None, returns all columns.

        Returns
        -------
        pl.DataFrame
            All market data.
        """
        if symbol_id is None:
            return (
                self.market_data
                .select(pl.all() if columns is None else columns)
            )
        elif isinstance(symbol_id, int):
            symbol_id = [symbol_id]
        else:
            symbol_id = pl.Series(values=symbol_id.flatten())
        return self.market_data.filter(pl.col("symbol_id").is_in(symbol_id)).select(pl.all() if columns is None else columns)

    def get_current_data(
        self,
        lags: int = 0,
        stride: int | None = None,
        symbol_id: int | np.ndarray[Any, int] | None = None,
        columns: List[str] | None = None,
    ) -> pl.DataFrame:
        """
        Return the market data for the current timestep.

        Parameters
        ----------
        lags : int, optional
            How many lagged historical values to include, by default 0
        stride : int | None, optional
            Number of timesteps between lagged values, by default None
        symbol_id : int | np.ndarray[Any, int] | None, optional
            Which symbol_ids to retrieve the data for, by default None. If None, returns the market data for all symbols.
        columns : List[str] | None, optional
            Which columns to retrieve, by default None. If None, returns all columns.

        Returns
        -------
        pl.DataFrame
            The market data for the current timestep
        """
        if stride is None:
            stride = 1
        assert self.market_id - stride * lags + 1 >= 0
        if symbol_id is None:
            symbol_id = self._all_symbols.tolist()
        elif isinstance(symbol_id, int):
            symbol_id = [symbol_id]
        else:
            symbol_id = symbol_id.flatten().tolist()
        data = (
            self.market_data
            .filter(
                pl.col("symbol_id").is_in(symbol_id)
                & pl.col("market_id").is_in(
                    list(
                        reversed(
                            range(self.market_id, self.market_id - stride * lags - 1, -stride)
                        )
                    )
                )
            )
            .sort("market_id", "symbol_id")
        )
        return data.select(pl.all() if columns is None else columns)

    def get_prices(
        self,
        symbol_id: int | np.ndarray[Any, int] | None = None,
        side: OrderType | None = None,
    ) -> pl.DataFrame:
        """
        Get the current market prices for the specified symbol_ids.

        Parameters
        ----------
        symbol_id : int | np.ndarray[Any, int] | None, optional
            Which symbols to retrieve the prices for, by default None. If None, returns the prices for all symbols.
        side : Literal[&quot;BUY&quot;, &quot;SELL&quot;] | None, optional
            Which side to retrieve the prices for, by default None. If None, no bid-ask spread is applied.

        Returns
        -------
        pl.DataFrame
            The requested prices.
        """
        match side:
            case OrderType.BUY:
                sign = 1
            case OrderType.SELL:
                sign = -1
            case None:
                sign = 0
            case _:
                raise ValueError(f"Unknown side argument {side}.")
        prices = (
            self.get_current_data(symbol_id=symbol_id, columns=["symbol_id", "midprice"])
            .with_columns(((1 + sign * self.bid_ask_spread * 1e-4) * pl.col("midprice")).alias("price"))
        )
        return prices.select("symbol_id", "price")

    @property
    @abstractmethod
    def market_data(self) -> pl.DataFrame:
        pass


@dataclass
class HistoricalMarket(Market):
    historical_market_data: pl.DataFrame

    def __post_init__(self):
        super().__post_init__()

    @property
    def market_data(self) -> pl.DataFrame:
        return self.historical_market_data
        
            
@dataclass
class SimulatedMarket(Market):
    market_simulator: MarketSimulator

    def __post_init__(self):
        self._market_data = self.market_simulator.reset()
        super().__post_init__()
        

    @property
    def market_data(self) -> pl.DataFrame:
        return self._market_data

    def reset(self):
        self._market_data = self.market_simulator.reset()
        return super().reset()