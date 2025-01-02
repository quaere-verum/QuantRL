import quantrl as qrl
import numpy as np
import matplotlib.pyplot as plt
from river.drift.retrain import DriftRetrainingClassifier
from river.naive_bayes import BernoulliNB
from river.drift.binary import DDM
import polars as pl
np.random.seed(123)

if __name__ == "__main__":
    timestep_id = np.full(200, -1, dtype=int)
    timestep_id[np.arange(5, 200, 5)] = np.arange(39)
    market_data = {
        "timestep_id": timestep_id,
        "market_id": np.arange(200),
        "date_id": (np.arange(200) / 20).astype(int),
        "time_id": np.arange(200) % 20,
        "symbol_id": np.full(200, 0, dtype=int),
        "midprice": 1 + np.linspace(0, 1, 200),
        "volume": np.random.lognormal(10, 2, size=200).astype(int) 
    }
    market_data_frame = pl.DataFrame(
        market_data, 
        schema={
            "timestep_id": pl.Int16, 
            "market_id": pl.Int16, 
            "date_id": pl.Int16, 
            "time_id": pl.Int16, 
            "symbol_id": pl.Int16, 
            "midprice": pl.Float32,
            "volume": pl.Int32,
        }
    )
    market_data_frame = market_data_frame.with_columns(pl.when(pl.col("timestep_id") == -1).then(None).otherwise(pl.col("timestep_id")).alias("timestep_id"))
    market_data_frame = pl.concat(
        (
            market_data_frame,
            market_data_frame
            .with_columns(
                pl.col("symbol_id").add(1).cast(pl.Int16), 
                pl.Series("midprice", 1 + np.linspace(0, -0.5, 200)).cast(pl.Float32),
                pl.Series("volume", np.random.lognormal(10, 2, size=200).astype(int)).cast(pl.Int32)
            )
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
        buffer_size=4,
        market=market,
        lags=10,
        stride=1,
        columns=["midprice", "volume"],
        is_stationary=[False, True],
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
        market_observation_columns=["midprice", "volume"],
        lags=3,
        stride=2,
        episode_length=17,
        take_profit=predictive_model.take_profit,
        stop_loss=predictive_model.stop_loss,
        horizon=predictive_model.lookahead_window,
    )
    market_data_frame.select("market_id", "symbol_id", "midprice").to_pandas().pivot(index="market_id", columns="symbol_id", values="midprice").plot()
    plt.show()
    import time
    start = time.time()
    for _ in range(1):
        env.reset(options={"initial_timestep": 7})
        done, truncated = False, False
        while not done and not truncated:
            obs, reward, done, truncated, info = env.step(np.array([0, -0.01, -0.01]))
            print(obs, reward)
        print("=" * 50)

    print(time.time() - start)