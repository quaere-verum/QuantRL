import numpy as np
from torch import nn
import torch
from typing import Tuple
from enum import Enum
from quantrl.utils.data import to_torch

class ActivationFunction(Enum):
    RELU = nn.ReLU
    GELU = nn.GELU
    SIGMOID = nn.Sigmoid
    TANH = nn.Tanh

class PreprocessingNet(nn.Module):
    def __init__(self, state_dim: int, linear_dims: Tuple[int, ...], activation_fn: ActivationFunction = ActivationFunction.RELU):
        super().__init__()
        self.output_dim = linear_dims[-1]
        dims = (state_dim,) + linear_dims
        model = []
        for k in range(1, len(dims)):
            model.extend([nn.Linear(dims[k - 1], dims[k]), activation_fn.value()])
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
class ActorDiscrete(nn.Module):
    def __init__(self, preprocessing_net: PreprocessingNet, action_dim: int):
        self.actor = nn.Sequential(
            preprocessing_net,
            nn.Linear(preprocessing_net.output_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def act(self, state: torch.Tensor | np.ndarray, evaluation: bool = False) -> torch.Tensor:
        state = to_torch(state)
        action_probs: torch.Tensor = self.actor(state)
        if evaluation:
            return action_probs.argmax(dim=1).item()
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            
            return action
        
class ActorContinuous(nn.Module):
    def __init__(self, preprocessing_net: PreprocessingNet, action_dim: int):
        self.actor = nn.Sequential(
            preprocessing_net,
            nn.Linear(preprocessing_net.output_dim, 2 * action_dim),
            nn.Softmax(dim=-1)
        )

    def act(self, state: torch.Tensor | np.ndarray, evaluation: bool = False) -> torch.Tensor:
        state = to_torch(state)
        action_dist_params: torch.Tensor = self.actor(state)
        action_dim = action_dist_params.shape[1] // 2
        if evaluation:
            return action_dist_params[:, :action_dim]
        else:
            dist = torch.distributions.Normal(action_dist_params[:, :action_dim], action_dist_params[:, action_dim:])
            action = dist.sample()
            
            return action
        
class Critic(nn.Module):
    def __init__(self, preprocessing_net: PreprocessingNet):
        self.critic = nn.Sequential(
            preprocessing_net,
            nn.Linear(preprocessing_net.output_dim, 1)
        )

    def value(self, state: torch.Tensor | np.ndarray, action: torch.Tensor | np.ndarray) -> torch.Tensor:
        state = to_torch(state)
        action = to_torch(action)
        return self.critic()

class ActorCritic(nn.Module):
    def __init__(self, preprocessing_net: PreprocessingNet, action_dim):
        super().__init__()

        self.actor = nn.Sequential(
            preprocessing_net,
            nn.Linear(preprocessing_net.output_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            preprocessing_net,
            nn.Linear(preprocessing_net.output_dim, 1)
        )
    
    def __call__(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = to_torch(state)
        action = to_torch(action)
        action_probs = self.actor(state)
        dist =  torch.distributions.Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_probs = dist.probs
        
        state_value = self.critic(state)
        
        return dist_probs, action_logprobs, torch.squeeze(state_value)