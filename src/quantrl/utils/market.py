from dataclasses import dataclass
import polars as pl
import numpy as np
from typing import Any, Literal, List
from enum import Enum

class OrderType(Enum):
    BUY = 1
    SELL = 2

@dataclass
class Market:
    market_data: pl.DataFrame
    bid_ask_spread: int

    def __post_init__(self) -> None:
        assert np.isin(
            ["date_id", "time_id", "symbol_id", "midprice"],
            self.market_data.columns
        ).all(), (
            "spot_price dataframe must contain columns with date_id, time_id, symbol_id"
        )
        assert self.market_data.schema["date_id"].is_integer()
        assert self.market_data.schema["time_id"].is_integer()
        assert self.market_data.schema["symbol_id"].is_integer()
        for col in self.market_data.columns:
            if col not in ["date_id", "time_id", "symbol_id"]:
                assert self.market_data.schema[col].is_numeric()

        self.market_data = self.market_data.sort("date_id", "time_id", "symbol_id")
        self._t: int | None = None
        self._all_symbols = self.market_data.select("symbol_id").unique()
        timestep_id = self.market_data.select("date_id", "time_id").unique().sort("date_id", "time_id").with_row_index("timestep_id")
        self.market_data = self.market_data.join(
            timestep_id, 
            on=["date_id", "time_id"],
        )

    def reset(self, timestep: int | None = None) -> None:
        self._t = timestep or 0

    def get_data(
        self,
        lags: int = 0,
        stride: int | None = None,
        symbol_id: int | np.ndarray[Any, int] | None = None,
        columns: List[str] | None = None,
    ) -> pl.DataFrame:
        if stride is None:
            stride = 1
        assert self._t - stride * lags >= 0
        if symbol_id is None:
            symbol_id_ = self._all_symbols
        elif isinstance(symbol_id, int):
            symbol_id_ = [symbol_id]
        else:
            symbol_id_ = pl.Series(values=symbol_id.flatten())
        data = (
            self.market_data
            .filter(
                pl.col("symbol_id").is_in(symbol_id_)
                & pl.col("timestep_id").is_in(
                    list(
                        reversed(
                            range(self._t, self._t - stride * lags - 1, -stride)
                        )
                    )
                )
            )
            .sort("timestep_id", "symbol_id")
        )
        return data.select(pl.all() if columns is None else columns)
    
    def get_prices(
        self,
        symbol_id: int | np.ndarray[Any, int] | None = None,
        side: Literal["BUY", "SELL"] | None = None,
    ) -> pl.DataFrame:
        if side is not None:
            assert side in OrderType.__members__
        if side == "BUY":
            sign = 1
        elif side == "SELL":
            sign = -1
        else:
            sign = 0
        prices = (
            self.get_data(symbol_id=symbol_id, columns=["symbol_id", "midprice"])
            .with_columns(((1 + sign * self.bid_ask_spread) * pl.col("midprice")).alias("price"))
        )
        return prices.select("symbol_id", "price")

    def step(self) -> None:
        self._t += 1