from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar, cast, override

import gymnasium as gym
import numpy as np
import torch
from numba import njit
from copy import deepcopy

from tianshou.data import ReplayBuffer, SequenceSummaryStats, to_torch_as
from tianshou.data.types import BatchWithAdvantagesProtocol, RolloutBatchProtocol
from tianshou.policy import PGPolicy
from tianshou.policy.base import TLearningRateScheduler, TrainingStats
from tianshou.policy.modelfree.pg import TDistFnDiscrOrCont
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils.net.discrete import Actor as DiscreteActor
from tianshou.utils.net.discrete import Critic as DiscreteCritic


@dataclass(kw_only=True)
class VMPOTrainingStats(TrainingStats):
    loss: SequenceSummaryStats
    actor_loss: SequenceSummaryStats
    vf_loss: SequenceSummaryStats
    temperature_loss: SequenceSummaryStats
    kl_loss: SequenceSummaryStats


TVMPOTrainingStats = TypeVar("TVMPOTrainingStats", bound=VMPOTrainingStats)


# TODO: the type ignore here is needed b/c the hierarchy is actually broken! Should reconsider the inheritance structure.
class VMPOPolicy(PGPolicy[TVMPOTrainingStats], Generic[TVMPOTrainingStats]):  # type: ignore[type-var]
    """Implementation of VMPO. arXiv:1909.12238.

    :param actor: the actor network following the rules:
        If `self.action_type == "discrete"`: (`s_B` ->`action_values_BA`).
        If `self.action_type == "continuous"`: (`s_B` -> `dist_input_BD`).
    :param critic: the critic network. (s -> V(s))
    :param optim: the optimizer for actor and critic network.
    :param dist_fn: distribution class for computing the action.
    :param action_space: env's action space
    :param max_grad_norm: clipping gradients in back propagation.
    :param discount_factor: in [0, 1].
    :param reward_normalization: normalize estimated values to have std close to 1.
    :param deterministic_eval: if True, use deterministic evaluation.
    :param observation_space: the space of the observation.
    :param action_scaling: if True, scale the action from [-1, 1] to the range of
        action_space. Only used if the action_space is continuous.
    :param action_bound_method: method to bound action to range [-1, 1].
        Only used if the action_space is continuous.
    :param lr_scheduler: if not None, will be called in `policy.update()`.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        *,
        actor: torch.nn.Module | ActorProb | DiscreteActor,
        critic: torch.nn.Module | Critic | DiscreteCritic,
        eta: torch.nn.Parameter,
        nu: torch.nn.Parameter,
        optim: torch.optim.Optimizer,
        dist_fn: TDistFnDiscrOrCont,
        action_space: gym.Space,
        eps_eta: float,
        eps_nu: float,
        max_grad_norm: float | None = None,
        discount_factor: float = 0.99,
        reward_normalization: bool = False,
        deterministic_eval: bool = False,
        observation_space: gym.Space | None = None,
        action_scaling: bool = True,
        action_bound_method: Literal["clip", "tanh"] | None = "clip",
        lr_scheduler: TLearningRateScheduler | None = None,
    ) -> None:
        super().__init__(
            actor=actor,
            optim=optim,
            dist_fn=dist_fn,
            action_space=action_space,
            discount_factor=discount_factor,
            reward_normalization=reward_normalization,
            deterministic_eval=deterministic_eval,
            observation_space=observation_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            lr_scheduler=lr_scheduler,
        )
        self.critic = critic
        self.eps_nu = eps_nu
        self.eps_eta = eps_eta
        self.max_grad_norm = max_grad_norm
        self._actor_critic = ActorCritic(self.actor, self.critic)
        self._actor_old = deepcopy(self._actor_critic.actor)
        self.eta = eta
        self.nu = nu

    @staticmethod
    def _compile() -> None:
        f64 = np.array([0, 1], dtype=np.float64)
        f32 = np.array([0, 1], dtype=np.float32)
        b = np.array([False, True], dtype=np.bool_)
        _calculate_discounted_rewards(terminal_states=b, rewards=f64, discount_factor=0.1)
        _calculate_discounted_rewards(terminal_states=b, rewards=f32, discount_factor=0.1)

    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithAdvantagesProtocol:
        batch = self._compute_returns(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
        return batch

    def _compute_returns(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithAdvantagesProtocol:
        v_s = []
        with torch.no_grad():
            v_s.append(self.critic(batch.obs))
        batch.v_s = torch.cat(v_s, dim=0).flatten()  # old value
        v_s = batch.v_s.cpu().numpy()
        if self.rew_norm:  # unnormalize v_s & v_s_
            v_s = v_s * np.sqrt(self.ret_rms.var + self._eps)
        unnormalized_returns = _calculate_discounted_rewards(
            terminal_states=np.logical_or(batch.terminated, batch.truncated),
            rewards=batch.rew,
            discount_factor=self.gamma,
        )
        advantages = unnormalized_returns - v_s
        if self.rew_norm:
            batch.returns = unnormalized_returns / np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        batch.returns = to_torch_as(batch.returns, batch.v_s)
        batch.adv = to_torch_as(advantages, batch.v_s)
        return cast(BatchWithAdvantagesProtocol, batch)

    def learn( 
        self,
        batch: RolloutBatchProtocol,
        batch_size: int | None,
        repeat: int,
        *args: Any,
        **kwargs: Any,
    ) -> TVMPOTrainingStats:
        losses, actor_losses, vf_losses, ent_losses, temp_losses = [], [], [], [], []
        split_batch_size = batch_size or -1
        for _ in range(repeat):
            for minibatch in batch.split(split_batch_size, merge_last=True):
                values = self._actor_critic.critic(minibatch.obs)
                action_dist_new = self(minibatch).dist
                action_dist_input_old, _ = self._actor_old(minibatch.obs, state=None, info=batch.info)
                action_dist_old = self.dist_fn(action_dist_input_old)
                log_prob_new = action_dist_new.log_prob(minibatch.act)
                
                # KL loss
                kl_divergence = torch.mean(action_dist_old.probs.detach() * (action_dist_old.probs.log().detach() - action_dist_new.probs.log()), dim=1)
                kl_loss = (
                    self.nu * (self.eps_nu - kl_divergence.detach()) + 
                    self.nu.detach() * kl_divergence
                ).mean()

                # Policy loss. Use the top 50% of advantages
                top_indices = torch.sort(minibatch.adv, descending=True).indices[:len(minibatch.adv) // 2]
                weights = torch.exp(minibatch.adv[top_indices].detach() - minibatch.adv[top_indices].detach().max() / self.eta)
                weights = weights / weights.sum()
                policy_loss = -torch.sum(
                    weights * log_prob_new[top_indices]
                )

                # Temperature loss
                temperature_loss = (
                    self.eta * self.eps_eta + self.eta * torch.mean(minibatch.adv[top_indices])
                )

                # Critic loss
                critic_loss = torch.pow(to_torch_as(minibatch.returns, values) - torch.flatten(values), 2).mean()

                loss = policy_loss + kl_loss + critic_loss + temperature_loss
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                actor_losses.append(policy_loss.item())
                vf_losses.append(critic_loss.item())
                ent_losses.append(kl_loss.item())
                temp_losses.append(temperature_loss.item())
                losses.append(loss.item())

        self._actor_old.load_state_dict(deepcopy(self._actor_critic.actor.state_dict()))

        loss_summary_stat = SequenceSummaryStats.from_sequence(losses)
        actor_loss_summary_stat = SequenceSummaryStats.from_sequence(actor_losses)
        vf_loss_summary_stat = SequenceSummaryStats.from_sequence(vf_losses)
        ent_loss_summary_stat = SequenceSummaryStats.from_sequence(ent_losses)
        temp_loss_summary_stat = SequenceSummaryStats.from_sequence(temp_losses)

        return VMPOTrainingStats(  # type: ignore[return-value]
            loss=loss_summary_stat,
            actor_loss=actor_loss_summary_stat,
            vf_loss=vf_loss_summary_stat,
            kl_loss=ent_loss_summary_stat,
            temperature_loss=temp_loss_summary_stat,
        )
    
@njit
def _calculate_discounted_rewards(
    terminal_states: np.ndarray[Any, np.bool_],
    rewards: np.ndarray[Any, np.float_],
    discount_factor: float
) -> np.ndarray[Any, np.float_]:
    discounted_rewards = np.zeros(rewards.shape, dtype=np.float64)
    discounted_reward = 0
    for k in range(len(terminal_states) - 1, -1, -1):
        if terminal_states[k]:
            discounted_reward = 0
        discounted_reward = rewards[k] + (discount_factor * discounted_reward)
        discounted_rewards[k] = discounted_reward
    return discounted_rewards