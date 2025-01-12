from dataclasses import dataclass
from abc import ABC, abstractmethod
from quantrl.utils.buffer import RolloutBuffer
import numpy as np
from numba import njit
from typing import Any
import torch


@dataclass
class BasePolicy(ABC):
    
    @abstractmethod
    def act(self, state: torch.Tensor | np.ndarray, evaluation: bool = False) -> torch.Tensor:
        pass

    @abstractmethod
    def learn(self, epochs: int, buffer: RolloutBuffer) -> None:
        pass

