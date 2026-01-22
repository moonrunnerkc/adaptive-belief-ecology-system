# Author: Bradley R. Kinnard
"""
RL Training Loop - trains policies over belief ecology episodes.
Supports PPO-style updates with experience replay.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional
import numpy as np

from .environment import BeliefEcologyEnv, EnvConfig, StepResult
from ..agents.rl_policy import EcologyState, PolicyAction

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for RL training."""

    # training loop
    total_episodes: int = 1000
    max_steps_per_episode: int = 100
    eval_frequency: int = 50
    eval_episodes: int = 10

    # learning
    learning_rate: float = 3e-4
    gamma: float = 0.99  # discount factor
    gae_lambda: float = 0.95  # GAE parameter
    clip_epsilon: float = 0.2  # PPO clip
    entropy_coef: float = 0.01
    value_coef: float = 0.5

    # batching
    batch_size: int = 64
    epochs_per_update: int = 4
    buffer_size: int = 2048

    # early stopping
    target_return: Optional[float] = None
    patience: int = 100


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data."""

    states: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.states)

    def is_full(self, buffer_size: int) -> bool:
        return len(self) >= buffer_size

    def compute_returns_and_advantages(
        self, gamma: float, gae_lambda: float, last_value: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns."""
        n = len(self)
        returns = np.zeros(n, dtype=np.float32)
        advantages = np.zeros(n, dtype=np.float32)

        last_gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            last_gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + self.values[t]

        return returns, advantages


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""

    episode: int = 0
    total_steps: int = 0
    episode_return: float = 0.0
    episode_length: int = 0
    avg_return: float = 0.0
    best_return: float = float("-inf")
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0

    def to_dict(self) -> dict:
        return {
            "episode": self.episode,
            "total_steps": self.total_steps,
            "episode_return": self.episode_return,
            "episode_length": self.episode_length,
            "avg_return": self.avg_return,
            "best_return": self.best_return,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy": self.entropy,
        }


class RLTrainer:
    """
    RL Trainer for belief ecology policies.
    Uses PPO-style updates with heuristic policy until neural network is integrated.
    """

    def __init__(
        self,
        env: BeliefEcologyEnv,
        config: Optional[TrainingConfig] = None,
    ):
        self.env = env
        self.config = config or TrainingConfig()

        self._buffer = RolloutBuffer()
        self._metrics = TrainingMetrics()
        self._episode_returns: list[float] = []

        # callbacks
        self._on_episode_end: Optional[Callable[[TrainingMetrics], None]] = None
        self._on_update: Optional[Callable[[TrainingMetrics], None]] = None

        # simple heuristic policy (replace with neural network)
        self._policy_mean = np.zeros(BeliefEcologyEnv.ACTION_DIM, dtype=np.float32)
        self._policy_std = np.ones(BeliefEcologyEnv.ACTION_DIM, dtype=np.float32) * 0.5

    def set_on_episode_end(self, callback: Callable[[TrainingMetrics], None]) -> None:
        self._on_episode_end = callback

    def set_on_update(self, callback: Callable[[TrainingMetrics], None]) -> None:
        self._on_update = callback

    def _sample_action(self, state: np.ndarray) -> tuple[np.ndarray, float, float]:
        """
        Sample action from policy.
        Returns (action, log_prob, value_estimate).

        TODO: Replace with neural network forward pass.
        """
        # simple gaussian policy
        action = np.random.normal(self._policy_mean, self._policy_std)
        action = np.clip(action, -1.0, 1.0)

        # log probability (diagonal gaussian)
        log_prob = -0.5 * np.sum(
            ((action - self._policy_mean) / self._policy_std) ** 2
            + np.log(2 * np.pi * self._policy_std ** 2)
        )

        # simple value estimate (mean of recent returns)
        value = np.mean(self._episode_returns[-10:]) if self._episode_returns else 0.0

        return action, log_prob, value

    def _update_policy(self) -> tuple[float, float, float]:
        """
        Update policy using PPO.
        Returns (policy_loss, value_loss, entropy).

        TODO: Replace with actual gradient updates.
        """
        returns, advantages = self._buffer.compute_returns_and_advantages(
            self.config.gamma, self.config.gae_lambda
        )

        # normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # placeholder: adjust policy based on advantage-weighted actions
        states = np.array(self._buffer.states)
        actions = np.array(self._buffer.actions)

        # simple update: move mean toward high-advantage actions
        for i in range(len(actions)):
            if advantages[i] > 0:
                self._policy_mean += 0.01 * advantages[i] * actions[i]
            else:
                self._policy_mean -= 0.005 * abs(advantages[i]) * actions[i]

        self._policy_mean = np.clip(self._policy_mean, -0.5, 0.5)

        # fake losses for logging
        policy_loss = -np.mean(advantages)
        value_loss = np.mean((returns - np.array(self._buffer.values)) ** 2)
        entropy = np.sum(np.log(self._policy_std))

        self._buffer.clear()

        return float(policy_loss), float(value_loss), float(entropy)

    def train(self) -> TrainingMetrics:
        """
        Run training loop.
        Returns final metrics.
        """
        logger.info(f"starting training for {self.config.total_episodes} episodes")

        for episode in range(1, self.config.total_episodes + 1):
            self._metrics.episode = episode

            # run episode
            ep_return, ep_length = self._run_episode()

            self._episode_returns.append(ep_return)
            self._metrics.episode_return = ep_return
            self._metrics.episode_length = ep_length
            self._metrics.total_steps += ep_length

            # running average
            recent = self._episode_returns[-100:]
            self._metrics.avg_return = float(np.mean(recent))
            self._metrics.best_return = max(self._metrics.best_return, ep_return)

            # update policy when buffer is full
            if self._buffer.is_full(self.config.buffer_size):
                p_loss, v_loss, entropy = self._update_policy()
                self._metrics.policy_loss = p_loss
                self._metrics.value_loss = v_loss
                self._metrics.entropy = entropy

                if self._on_update:
                    self._on_update(self._metrics)

            # episode callback
            if self._on_episode_end:
                self._on_episode_end(self._metrics)

            # logging
            if episode % 10 == 0:
                logger.info(
                    f"episode {episode}: return={ep_return:.3f}, "
                    f"avg={self._metrics.avg_return:.3f}, best={self._metrics.best_return:.3f}"
                )

            # early stopping
            if self.config.target_return and self._metrics.avg_return >= self.config.target_return:
                logger.info(f"reached target return {self.config.target_return}")
                break

        logger.info(f"training complete: {self._metrics.total_steps} total steps")
        return self._metrics

    def _run_episode(self) -> tuple[float, int]:
        """Run single episode, collecting experience."""
        obs, _ = self.env.reset()
        done = False
        ep_return = 0.0
        step_count = 0

        while not done:
            action, log_prob, value = self._sample_action(obs)
            result = self.env.step(action)

            self._buffer.add(
                state=obs,
                action=action,
                reward=result.reward,
                value=value,
                log_prob=log_prob,
                done=result.terminated or result.truncated,
            )

            obs = result.observation
            ep_return += result.reward
            step_count += 1
            done = result.terminated or result.truncated

        return ep_return, step_count

    def evaluate(self, num_episodes: int = 10) -> dict:
        """
        Evaluate current policy without exploration.
        Returns evaluation metrics.
        """
        returns = []
        lengths = []

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            ep_return = 0.0
            step_count = 0

            while not done:
                # deterministic action (no noise)
                action = np.clip(self._policy_mean, -1.0, 1.0)
                result = self.env.step(action)

                obs = result.observation
                ep_return += result.reward
                step_count += 1
                done = result.terminated or result.truncated

            returns.append(ep_return)
            lengths.append(step_count)

        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)),
            "min_return": float(np.min(returns)),
            "max_return": float(np.max(returns)),
        }

    def save_policy(self, path: str) -> None:
        """Save policy parameters."""
        np.savez(path, mean=self._policy_mean, std=self._policy_std)
        logger.info(f"saved policy to {path}")

    def load_policy(self, path: str) -> None:
        """Load policy parameters."""
        data = np.load(path)
        self._policy_mean = data["mean"]
        self._policy_std = data["std"]
        logger.info(f"loaded policy from {path}")


__all__ = [
    "RLTrainer",
    "TrainingConfig",
    "TrainingMetrics",
    "RolloutBuffer",
]
