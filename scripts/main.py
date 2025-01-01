import quantrl as qrl
import numpy as np
import matplotlib.pyplot as plt
from river.drift.retrain import DriftRetrainingClassifier
from river.naive_bayes import BernoulliNB
from river.drift.binary import DDM
import polars as pl
np.random.seed(123)

if __name__ == "__main__":
    timestep_id = np.full(100, -1, dtype=int)
    timestep_id[np.arange(5, 100, 5)] = np.arange(19)
    market_data = {
        "timestep_id": timestep_id,
        "market_id": np.arange(100),
        "date_id": (np.arange(100) / 20).astype(int),
        "time_id": np.arange(100) % 20,
        "symbol_id": np.full(100, 0, dtype=int),
        "midprice": np.exp(np.cumsum(np.random.normal(0, 0.05, size=100)))
    }
    market_data_frame = pl.DataFrame(
        market_data, 
        schema={"timestep_id": pl.Int8, "market_id": pl.Int8, "date_id": pl.Int8, "time_id": pl.Int8, "symbol_id": pl.Int8, "midprice": pl.Float32}
    )
    market_data_frame = market_data_frame.with_columns(pl.when(pl.col("timestep_id") == -1).then(None).otherwise(pl.col("timestep_id")).alias("timestep_id"))
    market_data_frame = pl.concat(
        (
            market_data_frame,
            market_data_frame.with_columns(pl.col("symbol_id").add(1), pl.Series("midprice", np.exp(np.cumsum(np.random.normal(0, 0.05, size=100)))).cast(pl.Float32))
        ),
    )
    market = qrl.HistoricalMarket(
        market_data=market_data_frame,
        bid_ask_spread=5
    )
    cash_account = qrl.ConstantInflowCashAccount(100)
    print(cash_account.current_capital)
    predictive_model = qrl.TripleBarrierClassifier(
        model=DriftRetrainingClassifier(BernoulliNB(), DDM()),
        share_model=False,
        market=market,
        lags=10,
        stride=1,
        columns=["midprice"],
        is_stationary=[False],
        lookahead_window=10,
        take_profit=0.1,
        stop_loss=-0.15,
        labels=np.array([0, 1], dtype=int)
    )
    portfolio = qrl.TripleBarrierPortfolio(model=predictive_model)
    env = qrl.TradingEnv(
        market=market,
        cash_account=cash_account,
        portfolio=portfolio,
        predictive_model=predictive_model,
        market_observation_columns=["midprice"],
        lags=3,
        stride=1,
        episode_length=15,
        take_profit=predictive_model.take_profit,
        stop_loss=predictive_model.stop_loss,
        horizon=predictive_model.lookahead_window,
    )
    env.reset(options={"initial_timestep": 3})
    done, truncated = False, False
    for _ in range(5):
        obs, reward, done, truncated, info = env.step(np.array([0, 0.01, 0.01]))
        print(obs, reward)