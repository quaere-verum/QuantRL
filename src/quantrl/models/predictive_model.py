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
    """
    Classifier used to generate signals.
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

    @abstractmethod
    def prepare_training_data(self, market: qrl.Market) -> pl.DataFrame:
        """
        Prepares the training data that is used to update the classifier at each timestep.
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

    def __post_init__(self) -> None:
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
        """
        Reset the classifier, to be used when resetting the reinforcement learning environment.

        Parameters
        ----------
        timestep : int
            Initialise the classifier to start at the provided timestep. E.g. if the inputs contain n lagged values
            from historical data, then timestep = n will be necessary.
        """
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
        """
        Evolve the classifier by one timestep to include new data, to be used when the reinforcement learning environment takes a step.
        """
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
            If True, returns the probability distribution instead of the predicted label.

        Returns
        -------
        int
            The predicted label, or the estimated probability distribution.
        """
        X = self.market_to_features(market=market, symbol_id=symbol_id)
        proba = self._model[symbol_id].predict_proba_one(X)
        distribution = np.zeros_like(self.labels, dtype=float)
        for key, value in proba.items():
            distribution[key] = value
        if return_distribution:
            return distribution / distribution.sum()
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