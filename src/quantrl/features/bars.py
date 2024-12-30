import numpy as np
import polars as pl
from typing import Iterable

def calculate_bars(
    dataframe: pl.DataFrame,
    bar_column_name: str,
    bar_size: float,
    aggregate_functions: Iterable[pl.Expr], 
) -> pl.DataFrame:
    """
    Use time bars to calculate new bars. A new bar is formed once 'bar_size' of 'bar_column_name' has 
    been registered. Consider e.g. pl.Series('bar_column_name', [0, 1, 0, 1, 2, 0, 3, 0, 1, 1]), with
    bar_size = 2. Then a new bar will be formed at t=3, 4, 8. That is, when 
    pl.col('bar_column_name').cum_sum().mod(bar_size) == 0.

    Parameters
    ----------
    dataframe : pl.DataFrame
        The dataframe to transform, with primary key ('market_id', 'symbol_id').
    bar_column_name : str
        The column name to aggregate on.
    bar_size : float
        The amount to fill before a new bar is generated.
    aggregate_functions : Iterable[pl.Expr]
        The aggregation expressions to fill the new dataframe.

    Returns
    -------
    pl.DataFrame
        The aggregated dataframe, with primary key ('bar_id', 'symbol_id').
    """
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
    