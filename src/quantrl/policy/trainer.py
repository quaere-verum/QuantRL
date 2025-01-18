from quantrl.policy.base import BasePolicy
from quantrl.utils.buffer import BaseBuffer
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
from typing import Tuple
import warnings
from tqdm import tqdm
import logging
import time
logging.basicConfig(format="%(asctime)s [%(levelname)s]: %(message)s", datefmt="%I:%M:%S")

@dataclass
class Trainer:
    policy: BasePolicy
    env_name: str
    buffer: BaseBuffer
    steps_per_round: int
    test_episodes: int  | None
    test_frequency: int | None
    off_policy: bool

    def __post_init__(self):
        if self.test_episodes is None and self.test_frequency is not None:
            raise ValueError("If test_frequency is not 'None', test_episodes must be specified.")
        if self.test_episodes is not None and self.test_frequency is None:
            warnings.warn("test_frequency is 'None', but test_episodes is not. No tests will be performed.")
        if not self.off_policy:
            assert self.steps_per_round // self.buffer.num_envs == self.buffer.buffer_size, (
                "On-policy replay buffer size should be the same as steps_per_round."
            )
        if self.buffer.num_envs == 1:
            self._env = gym.make(self.env_name)
        else:
            self._env = make_vector_env(self.env_name, self.buffer.num_envs)
        self._logger = logging.getLogger()
        self._logger.setLevel(logging.INFO)

    def run(self, rounds: int, epochs: int) -> None:
        for round in range(1, rounds + 1):
            self._logger.info(f"Training round {round} starting. Filling buffer.")
            self.fill_replay_buffer()
            self._logger.info(f"Finished filling buffer. Learning...")
            self.policy.learn(epochs, self.buffer)
            self._logger.info(f"Training round {round} finished.")
            if self.test_frequency is not None and round % self.test_frequency == 0:
                mean_test_reward, std_test_reward = self.test_policy()
                self._logger.info(f"Test reward: {mean_test_reward:.2f} +/- {std_test_reward:.2f}")

    def fill_replay_buffer(self) -> None:
        states, _ = self._env.reset(options={})
        for _ in tqdm(range(self.steps_per_round)):
            actions = self.policy.act(states).detach().numpy()
            if actions.size == 1:
                actions = actions.item()
            next_states, rewards, dones, truncateds, _ = self._env.step(actions)
            terminal_states = dones | truncateds
            self.buffer.add(
                action=actions,
                state=states,
                reward=np.where(terminal_states, 0, rewards),
                terminal_state=terminal_states,
            )
            if self.buffer.num_envs > 1:
                skip_reset = dict(zip(range(self.buffer.num_envs), np.logical_not(terminal_states)))
                reset_states, _ = self._env.reset(options=skip_reset)
                states = np.where(terminal_states[:, np.newaxis], reset_states, next_states)
            else:
                states, _ = self._env.reset() if terminal_states else (next_states, None)
            

    def test_policy(self) -> Tuple[float, float]:
        env = gym.make(self.env_name)
        rewards = np.zeros(self.test_episodes)
        for episode in range(self.test_episodes):
            total_reward = 0
            done, truncated = False, False
            state, _ = env.reset()
            while not done and not truncated:
                action = self.policy.act(
                    state,
                    evaluation=True,
                ).detach().numpy()
                if action.size == 1:
                    action = action.item()
                state, reward, done, truncated, _ = env.step(action)
                total_reward += reward
            rewards[episode] = total_reward
        return np.mean(rewards), np.std(rewards)
    

class ConditionalResetEnv(gym.Env):
    def __init__(self, env: gym.Env, env_id: int):
        self.env = env
        self.env_id = env_id
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self, *, seed = None, options = None):
        if options is not None and options.get(self.env_id, False):
            return self.env.observation_space.sample(), {}
        else:
            return self.env.reset(seed=seed, options=options)

    def step(self, action):
        return self.env.step(action)    
    
    def close(self):
        return self.env.close()


def make_vector_env(env_name: str, num_envs: int) -> gym.vector.AsyncVectorEnv:
    def make_env(env_id):
        return lambda: ConditionalResetEnv(gym.make(env_name), env_id)
    return gym.vector.AsyncVectorEnv([make_env(k) for k in range(num_envs)])