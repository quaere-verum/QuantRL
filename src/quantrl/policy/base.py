from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Tuple
import gymnasium as gym
from quantrl.utils.replay_buffer import ReplayBuffer
import numpy as np
import torch

@dataclass
class BasePolicy(ABC):
    
    @abstractmethod
    def act(self, action, evaluation: bool = False) -> torch.Tensor:
        pass

    @abstractmethod
    def learn(self, epochs: int, buffer: ReplayBuffer) -> None:
        pass
