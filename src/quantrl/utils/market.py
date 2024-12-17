from dataclasses import dataclass
import polars as pl
import numpy as np
from typing import Any

@dataclass
class Market:
    spot_price: pl.DataFrame

    def __post_init__(self) -> None:
        assert np.isin(
            ["time_id", "symbol_id"],
            self.spot_price.columns
        ).all(), (
            "spot_price dataframe must contain columns with date_id, time_id, symbol_id"
        )
        assert self.spot_price.schema["date_id"].is_integer()
        assert self.spot_price.schema["time_id"].is_integer()
        assert self.spot_price.schema["symbol_id"].is_integer()
        for col in self.spot_price.columns:
            if col not in ["date_id", "time_id", "symbol_id"]:
                assert self.spot_price.schema[col].is_numeric()

        self.spot_price = self.spot_price.sort("date_id", "time_id", "symbol_id")
        self._t: int | None = None
        self._all_symbols = self.spot_price.select("symbol_id").unique()
        step = self.spot_price.select("date_id", "time_id").unique().sort("date_id", "time_id").with_row_index("timestep_id")
        self.spot_price = self.spot_price.join(
            step, 
            on=["date_id", "time_id"],
        )

    def reset(self, timestep: int | None = None) -> None:
        self._t = timestep or 0

    def get_data(
        self,
        lags: int = 0,
        stride: int | None = None,
        symbol_id: int | np.ndarray[Any, int] | None = None,
    ) -> pl.DataFrame:
        if stride is None:
            stride = 1
        assert self._t - stride * lags >= 0
        if symbol_id is None:
            symbol_id_ = self._all_symbols
        elif isinstance(symbol_id, int):
            symbol_id = [symbol_id]
        else:
            symbol_id_ = pl.Series(values=symbol_id.flatten())
        data = (
            self.spot_price
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
        return data
    
    def get_prices(
        self,
        symbol_id: int | np.ndarray[Any, int] | None = None 
    ) -> pl.DataFrame:
        return self.get_data(symbol_id=symbol_id).select(["symbol_id", "price"])

    def step(self) -> None:
        self._t += 1