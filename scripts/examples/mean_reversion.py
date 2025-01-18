import quantrl as qrl
import numpy as np
import polars as pl
from dataclasses import dataclass
import pandas as pd
from river.base import Classifier


@dataclass
class PairsTradingMarketSimulator(qrl.MarketSimulator):
    estimated_spread_sigma: float
    z_score_for_action: float
    lookback_window: int
    generator: qrl.StochasticProcesses
    sigma: float
    theta: float
    sigma_spread: float
    n: int
    T: float

    def __post_init__(self):
        self._date_ids = np.floor(np.linspace(0, self.T * (self.n - 1) / self.n, self.n)).astype(int)
        self._time_ids = np.arange(self.n) % int(np.floor(self.n / self.T))
        self._market_ids = np.arange(self.n)
        self._schema = {
            "timestep_id": pl.Int32,
            "market_id": pl.Int32,
            "date_id": pl.Int32,
            "time_id": pl.Int32,
            "symbol_id": pl.Int32,
            "midprice": pl.Float64,
        }

    def reset(self):
        asset_prices = np.zeros((2, self.n))
        asset_prices[0, :] = self.generator.generate_geometric_brownian_motion(
            n_paths=1,
            initial_value=1.0,
            mu=0.0,
            cov=None,
            sigma=float(self.sigma),
            n=self.n,
            T=self.T,
            dt=None
        )
        spread = self.generator.generate_ornstein_uhlenbeck_process(
            n_paths=1,
            n=self.n,
            initial_value=0.0,
            mu=0.0,
            theta=self.theta,
            cov=None,
            sigma=self.sigma_spread,
            T=self.T,
            dt=None
        )
        asset_prices[1, :] = asset_prices[0, :] + spread.flatten()
        
        z_score_threshold = np.abs(spread / self.estimated_spread_sigma) > self.z_score_for_action
        action_times = np.flatnonzero(z_score_threshold)
        action_times = action_times[action_times >= self.lookback_window - 1]
        timestep_ids = np.full(self.n, -1, dtype=int)
        timestep_ids[action_times] = np.arange(len(action_times))
        data = (
            pd.DataFrame(
                columns=pd.MultiIndex.from_product((["midprice"], np.arange(len(asset_prices))), names=["midprice", "symbol_id"]),
                index=pd.MultiIndex.from_arrays((timestep_ids, self._market_ids, self._date_ids, self._time_ids), names=["timestep_id", "market_id", "date_id", "time_id"]),
                data=asset_prices.T
            )
            .stack(future_stack=True)
            .reset_index()
        )
        data = (
            pl.DataFrame(data=data, schema=self._schema)
            .with_columns(
                pl.when(pl.col("timestep_id") == -1)
                .then(None)
                .otherwise("timestep_id")
                .alias("timestep_id")
            )
        )

        return data

@dataclass
class PairsTradingPortfolio(qrl.Portfolio):
    take_profit: float
    stop_loss: float

    # TODO: add Greeks/risk metrics to position summary
    def summarise_positions(self, market: qrl.Market):
        return np.array(
            [
                qrl.value_portfolio(self.open_positions, market, None, True)
            ]
        )

    @property
    def summary_shape(self) -> tuple[int, ...]:
        return (1,)

    def closing_mask(self, market: qrl.Market) -> pl.Series:
        buy_prices = market.get_prices(side=qrl.OrderType.BUY).rename({"price": "buy_price"})
        sell_prices = market.get_prices(side=qrl.OrderType.SELL).rename({"price": "sell_price"})
    
        portfolio = (
            self.open_positions.join(buy_prices, on="symbol_id").join(sell_prices, on="symbol_id")
        ).with_columns(pl.when(pl.col("position") < 0).then(pl.col("buy_price")).otherwise(pl.col("sell_price")).alias("price"))
        portfolio = portfolio.with_columns(
            pl.when(
                pl.col("contract_id") == qrl.ContractType.SPOT.value
            ).then(
                (pl.col("price") / pl.col("entry_price") - 1) 
                * pl.when(pl.col("position") >= 0).then(pl.lit(1.0)).otherwise(pl.lit(-1.0))
            ).when(
                pl.col("contract_id") == qrl.ContractType.FUTURE.value
            ).then(
                None
            ).when(
                pl.col("contract_id") == qrl.ContractType.OPTION.value
            ).then(
                None
            ).alias("pnl")
        )
        return (portfolio.select("pnl").to_series() > self.take_profit) | (portfolio.select("pnl").to_series() < self.stop_loss)

class DummyClassifier(Classifier):

    def learn_one(self, X: dict[str, float], y: int) -> None:
        pass
    
    def predict_one(self, X: dict[str, float], **kwargs) -> int:
        return np.nan
    
    def predict_proba_one(self, X: dict[str, float]) -> dict[str, int]:
        return {0: np.nan, 1: np.nan}


class DummyModel(qrl.PredictiveModel):

    @property
    def performance(self) -> float:
        return np.nan 
    
    def market_to_features(self, market: qrl.Market, symbol_id: int) -> dict[str, float]:
        return {}
    def prepare_training_data(self, market):
        return pl.DataFrame(
            {
                "market_id": [-1],
                "timestep_id": [-1],
                "symbol_id": [0],
                "label": [0],
            },
            schema={
                "market_id": pl.Int16,
                "timestep_id": pl.Int16,
                "symbol_id": pl.Int16,
                "label": pl.Int16,
            }
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    market_simulator = PairsTradingMarketSimulator(
        estimated_spread_sigma=0.05,
        z_score_for_action=1.0,
        lookback_window=5,
        generator=qrl.StochasticProcesses(1),
        sigma=0.05,
        theta=4.0,
        sigma_spread=0.1,
        n=300,
        T=5,
    )
    market = qrl.SimulatedMarket(bid_ask_spread=3, market_simulator=market_simulator)
    # for _ in range(10):
    #     data = market_simulator.reset().to_pandas().pivot(index="market_id", columns="symbol_id", values=["midprice", "timestep_id"])
    #     spread = np.diff(data["midprice"].values, axis=1)
    #     plt.plot(spread)
    #     plt.axhline(market_simulator.estimated_spread_sigma, c="black")
    #     plt.axhline(-market_simulator.estimated_spread_sigma, c="black")
    #     plt.show()
    #     data = data.swaplevel(axis=1)
    #     symbols = data.columns.get_level_values(0).unique().to_numpy()
    #     for k, symbol_id in enumerate(symbols):
    #         plt.plot(data[(symbol_id, "midprice")], alpha=0.5)
    #         plt.scatter(data.dropna()[(symbol_id, "timestep_id")].index, data.dropna()[(symbol_id, "midprice")])
    #     plt.grid()
    #     plt.show()
    cash_account = qrl.ConstantInflowCashAccount(100, 1, 10)
    portfolio = PairsTradingPortfolio(0.1, -0.15)
    predictive_model = DummyModel(
        model=DummyClassifier(),
        market=market,
        labels=np.array([0, 1]),
        share_model=True,
        buffer_size=0,
    )
    env = qrl.StatArbEnv(
        market=market,
        cash_account=cash_account,
        portfolio=portfolio,
        predictive_model=predictive_model,
        lags=0,
        stride=1,
        market_observation_columns=["midprice"],
        episode_length=9,
        take_profit=0.02,
        stop_loss=-0.05,
        horizon=None,
        margin_percent=0.2,
    )
    for _ in range(10):
        done, truncated = False, False
        obs, info = env.reset()
        while not done and not truncated:
            obs, reward, done, truncated, _ = env.step(np.array([0.0, 0.0]))
            print(obs)