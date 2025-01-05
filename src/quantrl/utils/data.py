import torch
import numpy as np

def to_torch(tensor: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor).float()
    return tensor.float()