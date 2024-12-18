from dataclasses import dataclass
from river.drift.retrain import DriftRetrainingClassifier
from typing import Dict
from abc import abstractmethod
import quantrl as qrl
import polars as pl
import numpy as np
import copy as cp

@dataclass
class PredictiveModel:
    classifier: DriftRetrainingClassifier
    data_train: pl.DataFrame
    share_model: bool = False

    def __post_init__(self) -> None:
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
        X_train = data_train.select(self.features).to_numpy()
        y_train = data_train.select("label").to_numpy().flatten()
        symbol_ids = data_train.select("symbol_id").to_numpy().flatten().astype(int)
        for Xi, yi, symbol_id in zip(X_train, y_train, symbol_ids):
            self._model[symbol_id].learn_one(
                dict(zip(self.features, Xi)),
                yi
            )

    @property
    @abstractmethod
    def performance(self) -> float:
        pass

    def predict(self, market_data: pl.DataFrame, symbol_id: int) -> Dict[str, float]:
        X = self.polars_to_features(market_data=market_data, symbol_id=symbol_id)
        return self._model[symbol_id].predict_proba_one(X)
    
    @abstractmethod
    def polars_to_features(self, market_data: pl.DataFrame, symbol_id: int) -> Dict[str, float]:
        pass