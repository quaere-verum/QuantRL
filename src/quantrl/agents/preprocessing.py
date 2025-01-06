import torch
from enum import Enum
from typing import Tuple

class ActivationFunction(Enum):
    RELU = torch.nn.ReLU
    GELU = torch.nn.GELU
    SIGMOID = torch.nn.Sigmoid
    TANH = torch.nn.Tanh

class PreprocessingNet(torch.nn.Module):
    def __init__(self, state_dim: int, linear_dims: Tuple[int, ...], activation_fn: ActivationFunction = ActivationFunction.RELU):
        super().__init__()
        self.output_dim = linear_dims[-1]
        dims = (state_dim,) + linear_dims
        model = []
        for k in range(1, len(dims)):
            model.extend([torch.nn.Linear(dims[k - 1], dims[k]), activation_fn.value()])
        self.model = torch.nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)