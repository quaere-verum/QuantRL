from abc import ABC
import gymnasium as gym
import polars as pl
import numpy as np
from typing import Any, Dict
from dataclasses import dataclass
import quantrl as qrl

@dataclass
class BaseEnv(gym.Env):
    market: qrl.Market
    cash_account: qrl.CashAccount
    portfolio: qrl.PortfolioBase

    def __post_init__(self) -> None:
        pass

    def step(self, action: np.ndarray[Any, float | int]):
        pass

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        pass

    def render(self):
        pass