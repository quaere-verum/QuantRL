from quantrl.agents.preprocessing import PreprocessingNet
import torch
from quantrl.utils.data import to_torch
import numpy as np
from typing import Tuple

class ActorContinuous(torch.nn.Module):
    def __init__(self, preprocessing_net: PreprocessingNet, action_dim: int, stochastic: bool = True):
        super().__init__()
        self.stochastic = stochastic
        self._actor = torch.nn.Sequential(
            preprocessing_net,
            torch.nn.Linear(preprocessing_net.output_dim, (1 + self.stochastic) * action_dim),
        )

    def act(self, state: torch.Tensor | np.ndarray, evaluation: bool = False) -> torch.Tensor:
        state = to_torch(state)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        action_dist_params: torch.Tensor = self._actor(state)
        action_dim = action_dist_params.shape[1] // 2
        if evaluation or not self.stochastic:
            return action_dist_params[:, :action_dim]
        else:
            dist = torch.distributions.Normal(action_dist_params[:, :action_dim], action_dist_params[:, action_dim:])
            action = dist.sample()
            
            return action
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self._actor(state)
    
class ActorDiscrete(torch.nn.Module):
    def __init__(self, preprocessing_net: PreprocessingNet, nr_actions: int, stochastic: bool = True):
        super().__init__()
        self.stochastic = stochastic
        self._actor = torch.nn.Sequential(
            preprocessing_net,
            torch.nn.Linear(preprocessing_net.output_dim, nr_actions),
            torch.nn.Softmax(dim=-1)
        )

    def act(self, state: torch.Tensor | np.ndarray, evaluation: bool = False) -> torch.Tensor:
        state = to_torch(state)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        action_probs: torch.Tensor = self.forward(state)
        if evaluation or not self.stochastic:
            return action_probs.argmax(dim=1)
        else:
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            
            return action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self._actor(state)

class Critic(torch.nn.Module):
    def __init__(self, preprocessing_net: PreprocessingNet, include_action: bool = True):
        super().__init__()
        self._preprocessing_net = preprocessing_net
        self._output_layer = torch.nn.Linear(preprocessing_net.output_dim + include_action, 1)
        self.include_action = include_action

    def value(self, state: torch.Tensor | np.ndarray, action: torch.Tensor | np.ndarray | None = None) -> torch.Tensor:
        state = to_torch(state)
        if action is not None:
            action = to_torch(action)
        if state.ndim == 1:
            state = state.unsqueeze(0)
            if action is not None:
                action = action.unsqueeze(0)
        return self.forward(state, action)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        if self.include_action:
            assert action is not None, (
                "Action must be provided when include_action = True."
            )
        x = self._preprocessing_net(state)
        if self.include_action:
            x = torch.cat((x, action), dim=1)
        return self._output_layer(x)
    
class ActorCritic(torch.nn.Module):
    def __init__(self, actor: ActorContinuous | ActorDiscrete, critic: Critic):
        super().__init__()
        self.actor = actor
        self.critic = critic
    
    def forward(self, state: torch.Tensor | np.ndarray, action: torch.Tensor | np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = to_torch(state)
        action = to_torch(action)
        action_prob_params = self.actor(state)
        
        if isinstance(self.actor, ActorDiscrete):
            dist = torch.distributions.Categorical(action_prob_params)
            if action.ndim == 2:
                assert action.shape[1] == 1
                action = action.squeeze(1)
            action_logprobs = dist.log_prob(action)
            dist_probs = dist.probs
        else:
            raise NotImplementedError()
        
        state_value = self.critic.forward(state, action)
        
        return dist_probs, action_logprobs, torch.squeeze(state_value)
    

class QNet(torch.nn.Module):
    def __init__(self, preprocessing_net: PreprocessingNet, n_actions: int):
        super().__init__()
        self._preprocessing_net = preprocessing_net
        self._output_layer = torch.nn.Linear(preprocessing_net.output_dim, n_actions)

    def value(self, state: torch.Tensor | np.ndarray) -> torch.Tensor:
        state = to_torch(state)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        return self.forward(state)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self._preprocessing_net(state)
        return self._output_layer(x)