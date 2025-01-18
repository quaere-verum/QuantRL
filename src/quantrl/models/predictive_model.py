from __future__ import annotations
from dataclasses import dataclass
from river.base import Classifier, Regressor
from typing import Dict, List, Any, Tuple
from abc import abstractmethod
import polars as pl
import numpy as np
import quantrl as qrl
import copy as cp

@dataclass
class PredictiveModel:
    model: Classifier | Regressor
    """
    Model used to generate signals.
    """
    market: qrl.Market
    """
    Market object used to generate the training data for each timestep. Care must be taken to not introduce data leakage!    
    """
    labels: np.ndarray[Any, int]
    """
    The labels that can occur in the training data.
    """
    share_model: bool
    """
    Whether to share the classifier for each symbol_id, or use a unique one per symbol_id.
    """
    buffer_size: int
    """
    Size of the buffer that is used to calculate model performance across a rolling window.
    """

    def __post_init__(self) -> None:
        assert isinstance(self.model, (Classifier, Regressor))
        assert self.labels.min() == 0
        self.labels.sort()
        assert np.all(np.diff(self.labels) == 1)
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
                int(symbol_id): cp.deepcopy(self.model)
                for symbol_id in symbols
            }
        else:
            model = cp.deepcopy(self.model)
            self._model = {
                int(symbol_id): model
                for symbol_id in symbols
            }
        self.features = [col for col in self.data_train.columns if col not in ["timestep_id", "symbol_id", "label"]]
        self._label_container: Dict[int, np.ndarray] | None = None
        self._ptr: int | None = None
        self._t: int | None = None
        self._buffer_filled: bool | None = None
    
    @abstractmethod
    def prepare_training_data(self, market: qrl.Market) -> pl.DataFrame:
        """
        Prepares the training data that is used to update the model at each timestep.
        Take care to not introduce data leakage!

        Parameters
        ----------
        market : qrl.Market
            Market object from which the training data is derived.

        Returns
        -------
        pl.DataFrame
            The dataframe with training data. Should contain the columns 'timestep_id', 'symbol_id' and 'label'.
        """
        pass

    def reset(self) -> None:
        """
        Reset the model, to be used when resetting the reinforcement learning environment.

        Parameters
        ----------
        timestep : int
            Initialise the model to start at the provided timestep. E.g. if the inputs contain n lagged values
            from historical data, then timestep = n will be necessary.
        """
        self._t = 0
        self._ptr = 0
        if not self.share_model:
            self._model = {
                int(symbol_id): cp.deepcopy(self.model)
                for symbol_id in self.symbols
            }
        else:
            model = cp.deepcopy(self.model)
            self._model = {
                int(symbol_id): model
                for symbol_id in self.symbols
            }
        self._label_container = np.zeros(shape=(self.buffer_size, 2), dtype=float)

    def step(self) -> None:
        """
        Evolve the model by one timestep to include new data, to be used when the reinforcement learning environment takes a step.
        """
        self._t += 1
        data_train = self.data_train.filter(pl.col("timestep_id") == self._t)
        if len(data_train) > 0:
            X_train = data_train.select(self.features).to_numpy()
            y_train = data_train.select("label").to_numpy().flatten()
            symbol_ids = data_train.select("symbol_id").to_numpy().flatten().astype(int)
            for Xi, yi, symbol_id in zip(X_train, y_train, symbol_ids):
                if isinstance(self._model[symbol_id], Regressor):
                    model_pred = self._model[symbol_id].predict_one(Xi)
                else:
                    proba = self._model[symbol_id].predict_proba_one(Xi)
                    distribution = np.zeros_like(self.labels, dtype=float)
                    for key, value in proba.items():
                        distribution[key] = value
                    model_pred = distribution.argmax().item()
                self._ptr = (self._ptr + 1) % (self.buffer_size - 1)
                if self._ptr == 0:
                    self._buffer_filled = True
                self._label_container[self._ptr] = (yi, model_pred)
                self._model[symbol_id].learn_one(
                    dict(zip(self.features, Xi)),
                    yi
                )
        

    def predict(self, market: qrl.Market, symbol_id: int, *, return_distribution: bool = False) -> int | np.ndarray[Any, float]:
        """
        Makes a prediction for the given symbol_id, given the current market.

        Parameters
        ----------
        market : qrl.Market
            Market object from which to derive features.
        symbol_id : int
            symbol_id for which the prediction is made.
        return_distribution : bool
            If True, returns the probability distribution instead of the predicted label, if the model is a classifier.

        Returns
        -------
        int
            The predicted value, or the estimated probability distribution.
        """
        X = self.market_to_features(market=market, symbol_id=symbol_id)
        if isinstance(self._model[symbol_id], Regressor):
            return self._model[symbol_id].predict_one(X)
        else:
            proba = self._model[symbol_id].predict_proba_one(X)
            distribution = np.zeros_like(self.labels, dtype=float)
            for key, value in proba.items():
                distribution[key] = value
            if return_distribution:
                return distribution / distribution.sum()
            else:
                if np.all(np.isnan(distribution)):
                    return np.nan
                else:
                    return distribution.argmax().item()
    
    @property
    @abstractmethod
    def performance(self) -> float:
        """
        A measure of the model's performance, to be used as part of the observation space.

        Returns
        -------
        float
            Some measure of the model's performance (e.g. rolling average of accuracy).
        """
        pass
    
    @abstractmethod
    def market_to_features(self, market: qrl.Market, symbol_id: int) -> Dict[str, float]:
        """
        Specifies the features that should be derived from the market to make a prediction.

        Parameters
        ----------
        market : qrl.Market
            Market object to derive features from.
        symbol_id : int
            The symbol_id for which to generate features.

        Returns
        -------
        Dict[str, float]
            Dictionary with feature names and corresponding values.
        """
        pass
