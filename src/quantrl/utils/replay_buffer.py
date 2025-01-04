import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class ReplayBuffer(ABC):
    buffer_size: int
    