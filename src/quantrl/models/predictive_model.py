from dataclasses import dataclass
from river.drift.retrain import DriftRetrainingClassifier
from typing import Dict, List, Any
from abc import abstractmethod
import polars as pl
import numpy as np
import quantrl as qrl
import copy as cp

@dataclass
class PredictiveModel:
    classifier: DriftRetrainingClassifier
    market: qrl.Market
    share_model: bool

    @abstractmethod
    def prepare_training_data(self, market: qrl.Market) -> pl.DataFrame:
        pass

    def __post_init__(self) -> None:
        self.data_train = self.prepare_training_data(self.market)
        assert np.isin(
            ["timestep_id", "symbol_id", "label"],
            self.data_train.columns
        ).all()
        assert self.data_train.schema["timestep_id"].is_integer()
        assert self.data_train.schema["symbol_id"].is_integer()
        assert self.data_train.schema["label"].is_integer()
        symbols = np.sort(self.data_train.select("symbol_id").unique().to_numpy().flatten())
        self.symbols = symbols
        if not self.share_model:
            self._model = {
                int(symbol_id): cp.deepcopy(self.classifier)
                for symbol_id in symbols
            }
        else:
            cf = cp.deepcopy(self.classifier)
            self._model = {
                int(symbol_id): cf
                for symbol_id in symbols
            }
        self.features = [col for col in self.data_train.columns if col not in ["timestep_id", "symbol_id", "label"]]
        self._t: int | None = None
    
    def reset(self, timestep: int | None = None) -> None:
        self._t = timestep or 0
        if not self.share_model:
            self._model = {
                int(symbol_id): cp.deepcopy(self.classifier)
                for symbol_id in self.symbols
            }
        else:
            cf = cp.deepcopy(self.classifier)
            self._model = {
                int(symbol_id): cf
                for symbol_id in self.symbols
            }

    def step(self) -> None:
        self._t += 1
        data_train = self.data_train.filter(pl.col("timestep_id") == self._t)
        if len(data_train) > 0:
            X_train = data_train.select(self.features).to_numpy()
            y_train = data_train.select("label").to_numpy().flatten()
            symbol_ids = data_train.select("symbol_id").to_numpy().flatten().astype(int)
            for Xi, yi, symbol_id in zip(X_train, y_train, symbol_ids):
                self._model[symbol_id].learn_one(
                    dict(zip(self.features, Xi)),
                    yi
                )

    def predict(self, market: qrl.Market, symbol_id: int) -> int:
        X = self.polars_to_features(market=market, symbol_id=symbol_id)
        proba = self._model[symbol_id].predict_proba_one(X)
        if 1 in proba.keys():
            return proba[1]
        else:
            return 0
    
    @property
    @abstractmethod
    def performance(self) -> float:
        pass
    
    @abstractmethod
    def polars_to_features(self, market_data: pl.DataFrame, symbol_id: int) -> Dict[str, float]:
        pass

@dataclass
class TripleBarrierClassifier(PredictiveModel):
    lags: int
    stride: int
    columns: List[str]
    is_stationary: List[bool]
    lookahead_window: int
    take_profit: float
    stop_loss: float

    def __post_init__(self):
        super().__post_init__()
        assert len(self.columns) == len(self.is_stationary)
        self.feature_names = self.columns.copy()
        for shift in range(self.stride, self.stride * self.lags + 1, self.stride):
            self.feature_names.extend(
                [
                    f"{col}_{shift}"
                    for col in self.columns
                ]
            )

    def _triple_barrier_label(self, X: np.ndarray[Any, float]) -> np.ndarray[Any, float]:
        X = X / X[:, 0, np.newaxis] - 1
        profit = np.argmax(X > self.take_profit, axis=1)
        loss = np.argmax(X < self.stop_loss, axis=1)
        label = np.where(
            profit < loss,
            1,
            0
        )
        return label


    def prepare_training_data(self, market: qrl.Market) -> pl.DataFrame:
        data = (
            market.market_data.select("timestep_id", "symbol_id", "midprice")
            .with_columns(
                [
                    (
                        pl.col("midprice").shift(-k)
                        .over("symbol_id", order_by="timestep_id").alias(f"midprice_{k}")
                    )
                    for k in range(1, self.lookahead_window + 1)
                ]
            )
        ).drop("timestep_id", "symbol_id").to_numpy()

        label = self._triple_barrier_label(data)
        features = (
            self.market.market_data.select(
                ["timestep_id", "symbol_id"] + self.columns
            )
        )
        divisor = features.select(self.columns).clone()
        for col, stationary in zip(self.columns, self.is_stationary):
            if stationary:
                divisor._replace(col, pl.Series(col, np.ones(len(divisor))))
        for shift in range(self.stride, self.stride * self.lags + 1, self.stride):
            shifted_features = (
                (features.select(self.columns).shift(-shift) / divisor)
                .with_columns(
                    features.select("timestep_id").to_series(),
                    features.select("symbol_id").to_series(),
                )
            ).rename(dict(zip(self.columns, [f"{col}_{shift}" for col in self.columns])))
            features = features.join(
                shifted_features,
                on=["timestep_id", "symbol_id"]
            )
        for col, stationary in zip(self.columns, self.is_stationary):
            if not stationary:
                features = features.drop(col)
        data_train = features.with_columns(pl.Series(name="label", values=label))
        return data_train.drop_nulls().drop_nans()

    @property
    def performance(self) -> float:
        return np.nan
    
    def polars_to_features(self, market: qrl.Market, symbol_id: int) -> Dict[str, float]:
        features = dict(
            zip(
                self.feature_names, 
                market.get_data(
                    self.lags, 
                    self.stride, 
                    symbol_id=symbol_id, 
                    columns=self.columns
                ).to_numpy()
                .flatten()
            )
        )
        for col, stationary in zip(self.columns, self.is_stationary):
            if not stationary:
                del features[col]
        return features