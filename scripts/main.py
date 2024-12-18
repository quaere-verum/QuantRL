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
        pl.DataFrame(market_data),
        bid_ask_spread=1e-4
    )
    cash_account = qrl.ConstantInflowCashAccount(100)
    portfolio = qrl.InvestmentPortfolio()
    predictive_model = qrl.TripleBarrierClassifier(
        classifier=DriftRetrainingClassifier(BernoulliNB(), DDM()),
        share_model=False,
        market=market,
        lags=10,
        stride=1,
        columns=["midprice"],
        lookahead_window=10,
        take_profit=0.01,
        stop_loss=-0.02
    )
    env = qrl.BaseEnv(
        market=market,
        cash_account=cash_account,
        portfolio=portfolio,
        predictive_model=predictive_model,
        lags=0,
        stride=1
    )

    env.reset()
    print(env.step(np.array([0])))