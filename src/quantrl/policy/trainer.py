from quantrl.policy.base import BasePolicy
from quantrl.utils.replay_buffer import ReplayBuffer
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
from typing import Tuple

@dataclass
class Trainer:
    policy: BasePolicy
    environment: gym.Env
    replay_buffer: ReplayBuffer
    steps_per_round: int
    test_episodes: int | None
    test_frequency: int | None
    off_policy: bool

    def __post_init__(self):
        if not self.off_policy:
            assert self.steps_per_round == self.replay_buffer.buffer_size, (
                "On-policy replay buffer size should be the same as steps_per_round."
            )

    def run(self, rounds: int, epochs: int) -> None:
        for round in range(rounds):
            self.fill_replay_buffer()
            self.policy.learn(epochs, self.replay_buffer)
            if self.test_frequency is not None:
                if round % self.test_frequency == 0:
                    mean_test_reward, std_test_reward = self.test_policy()
                    print(f"\tRound {round}. Test reward: {mean_test_reward:.2f} +/- {std_test_reward:.2f}")

    def fill_replay_buffer(self) -> float:
        done, truncated = False, False
        state, _ = self.environment.reset()
        for step in range(self.steps_per_round):
            action = self.policy.act(state).item()
            next_state, reward, done, truncated, _ = self.environment.step(action)
            terminal_state = done or truncated
            self.replay_buffer.add(
                action=action,
                state=state,
                reward=0. if terminal_state else reward,
                terminal_state=terminal_state,
            )
            if terminal_state:
                state, _ = self.environment.reset()
            else:
                state = next_state

    def test_policy(self) -> Tuple[float, float]:
        rewards = np.zeros(self.test_episodes)
        for episode in range(self.test_episodes):
            total_reward = 0
            done, truncated = False, False
            state, _ = self.environment.reset()
            while not done and not truncated:
                action = self.policy.act(
                    state,
                    evaluation=True,
                ).item()
                state, reward, done, truncated, _ = self.environment.step(action)
                total_reward += reward
            rewards[episode] = total_reward
        return np.mean(rewards), np.std(rewards)