import quantrl as qrl
import numpy as np
import matplotlib.pyplot as plt
from river.drift.retrain import DriftRetrainingClassifier
from river.naive_bayes import BernoulliNB
from river.drift.binary import DDM
import polars as pl
np.random.seed(123)

if __name__ == "__main__":
    market_data = {
        "date_id": (np.arange(100) / 20).astype(int),
        "time_id": np.arange(100) % 20,
        "symbol_id": np.full(100, 0, dtype=int),
        "midprice": np.exp(np.cumsum(np.random.normal(0, 0.05, size=100)))
    }
    market = qrl.Market(
        pl.DataFrame(market_data, schema={"date_id": pl.Int8, "time_id": pl.Int8, "symbol_id": pl.Int8, "midprice": pl.Float32}),
        bid_ask_spread=1e-4
    )
    cash_account = qrl.ConstantInflowCashAccount(100)
    predictive_model = qrl.TripleBarrierClassifier(
        classifier=DriftRetrainingClassifier(BernoulliNB(), DDM()),
        share_model=False,
        market=market,
        lags=10,
        stride=1,
        columns=["midprice"],
        is_stationary=[False],
        lookahead_window=10,
        take_profit=0.1,
        stop_loss=-0.15
    )
    portfolio = qrl.TripleBarrierPortfolio(model=predictive_model)
    portfolio.open_position(
        0,
        2.0,
        market_data["midprice"][0] * 1.18,
        None,
        "SPOT",
        None,
    )
    env = qrl.TradingEnv(
        market=market,
        cash_account=cash_account,
        portfolio=portfolio,
        predictive_model=predictive_model,
        market_observation_columns=["midprice"],
        lags=3,
        stride=1,
        action_shape=(1,),
        action_size=None,
        episode_length=50,
    )
    env.reset(options={"initial_timestep": 10})
    for _ in range(10):
        obs, reward, done, truncated, info = env.step(np.array([1.0]))
        print(obs, reward)