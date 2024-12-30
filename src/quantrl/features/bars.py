import numpy as np
import polars as pl
from typing import Iterable

def calculate_bars(
    dataframe: pl.DataFrame,
    bar_column_name: str,
    bar_size: float,
    aggregate_functions: Iterable[pl.Expr], 
) -> pl.DataFrame:
    assert np.all(np.isin(["symbol_id", "market_id", bar_column_name], dataframe.columns))
    dataframe = (
        dataframe
        .with_columns(
            pl.col(bar_column_name)
            .cum_sum()
            .over(partition_by="symbol_id", order_by="market_id")
            .mod(bar_size)
            .alias("cm")
        )
        .with_columns(
            pl.col("cm")
            .diff()
            .over(partition_by="symbol_id", order_by="market_id")
            .is_between(-np.inf, 0)
            .cast(int)
            .fill_null(0)
            .cum_sum()
            .alias("bar_id")
        )
        .drop("cm")
    )
    bars = (
        dataframe
        .group_by("bar_id", "symbol_id")
        .agg(aggregate_functions)
    ).sort("bar_id", "symbol_id")
    return bars
    