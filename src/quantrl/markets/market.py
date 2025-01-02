from dataclasses import dataclass
from abc import ABC, abstractmethod
import polars as pl
import numpy as np
from typing import Any, Literal, List
from enum import Enum

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
            ["timestep_id", "market_id", "date_id", "time_id", "symbol_id", "midprice"],
            all_market_data.columns
        ).all(), (
            "spot_price dataframe must contain columns with timestep_id, market_id, date_id, time_id, symbol_id"
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
        assert all_market_data.schema["date_id"].is_integer()
        assert all_market_data.schema["time_id"].is_integer()
        assert all_market_data.schema["symbol_id"].is_integer()
        for col in all_market_data.columns:
            if col not in ["timestep_id", "market_id", "date_id", "time_id", "symbol_id"]:
                assert all_market_data.schema[col].is_numeric()

        
        self._all_symbols = all_market_data.select("symbol_id").unique().to_numpy().flatten()
        self._t: int | None = None
        self._market_id: int | None = None
        # market_id = self.market_data.select("date_id", "time_id").unique().sort("date_id", "time_id").with_row_index("market_id")
        # self.market_data = self.market_data.join(
        #     market_id, 
        #     on=["date_id", "time_id"],
        # )
        # self.market_data = self.market_data.sort("market_id", "symbol_id")
        

    @abstractmethod
    def reset(self, timestep: int) -> None:
        """
        Reset the market, to be used when resetting the reinforcement learning environment.

        Parameters
        ----------
        timestep : int
            Initialise the market to start at the provided timestep. E.g. if the observations contain n lagged values
            from historical data, then timestep = n will be necessary.
        """
        pass

    @abstractmethod
    def step(self) -> None:
        """
        Evolve the market by one timestep, to be used when the reinforcement learning environment takes a step.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass


@dataclass
class HistoricalMarket(Market):
    market_data: pl.DataFrame

    def __post_init__(self):
        super().__post_init__()

    def reset(self, timestep: int) -> None:
        self._t = timestep
        self._market_id = int(self.market_data.filter(pl.col("timestep_id") == self._t).select("market_id").max().item())

    def get_all_data(self, symbol_id = None, columns = None):
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
        if stride is None:
            stride = 1
        assert self._market_id - stride * lags >= 0
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
                            range(self._market_id, self._market_id - stride * lags - 1, -stride)
                        )
                    )
                )
            )
            .sort("market_id", "symbol_id")
        )
        return data.select(pl.all() if columns is None else columns)
    
    def get_prices(self, symbol_id = None, side = None):
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

    def step(self) -> None:
        self._t += 1
        self._market_id = int(self.market_data.filter(pl.col("timestep_id") == self._t).select("market_id").max().item())
        