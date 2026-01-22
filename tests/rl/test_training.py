# Author: Bradley R. Kinnard
"""Tests for RL Training Loop."""

import pytest
import numpy as np

from backend.rl.training import (
    RLTrainer,
    TrainingConfig,
    TrainingMetrics,
    RolloutBuffer,
)
from backend.rl.environment import BeliefEcologyEnv, EnvConfig
from backend.core.models.belief import Belief, BeliefStatus, OriginMetadata


def _make_belief(confidence: float = 0.8) -> Belief:
    return Belief(
        content="test",
        confidence=confidence,
        status=BeliefStatus.Active,
        origin=OriginMetadata(source="test"),
    )


class TestRolloutBuffer:
    def test_add_and_len(self):
        buffer = RolloutBuffer()
        buffer.add(
            state=np.zeros(5),
            action=np.zeros(3),
            reward=1.0,
            value=0.5,
            log_prob=-0.1,
            done=False,
        )
        assert len(buffer) == 1

    def test_clear(self):
        buffer = RolloutBuffer()
        buffer.add(np.zeros(5), np.zeros(3), 1.0, 0.5, -0.1, False)
        buffer.clear()
        assert len(buffer) == 0

    def test_is_full(self):
        buffer = RolloutBuffer()
        for _ in range(10):
            buffer.add(np.zeros(5), np.zeros(3), 1.0, 0.5, -0.1, False)

        assert buffer.is_full(10)
        assert not buffer.is_full(20)

    def test_compute_returns_and_advantages(self):
        buffer = RolloutBuffer()
        for i in range(5):
            buffer.add(
                state=np.zeros(5),
                action=np.zeros(3),
                reward=1.0,
                value=0.5,
                log_prob=-0.1,
                done=i == 4,
            )

        returns, advantages = buffer.compute_returns_and_advantages(
            gamma=0.99, gae_lambda=0.95
        )

        assert len(returns) == 5
        assert len(advantages) == 5
        assert returns.dtype == np.float32


class TestTrainingConfig:
    def test_default_values(self):
        config = TrainingConfig()
        assert config.total_episodes == 1000
        assert config.learning_rate == 3e-4
        assert config.gamma == 0.99

    def test_custom_values(self):
        config = TrainingConfig(total_episodes=100, gamma=0.95)
        assert config.total_episodes == 100
        assert config.gamma == 0.95


class TestTrainingMetrics:
    def test_to_dict(self):
        metrics = TrainingMetrics(episode=5, episode_return=10.5)
        d = metrics.to_dict()

        assert d["episode"] == 5
        assert d["episode_return"] == 10.5


class TestRLTrainer:
    def test_init(self):
        env = BeliefEcologyEnv()
        trainer = RLTrainer(env)

        assert trainer.env is env
        assert trainer.config is not None

    def test_sample_action_shape(self):
        env = BeliefEcologyEnv()
        trainer = RLTrainer(env)

        state = np.zeros(BeliefEcologyEnv.STATE_DIM)
        action, log_prob, value = trainer._sample_action(state)

        assert action.shape == (BeliefEcologyEnv.ACTION_DIM,)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_sample_action_clipped(self):
        env = BeliefEcologyEnv()
        trainer = RLTrainer(env)

        state = np.zeros(BeliefEcologyEnv.STATE_DIM)
        for _ in range(100):
            action, _, _ = trainer._sample_action(state)
            assert np.all(action >= -1.0)
            assert np.all(action <= 1.0)

    def test_run_episode(self):
        env = BeliefEcologyEnv(EnvConfig(max_steps_per_episode=10))
        trainer = RLTrainer(env)

        env.reset()
        ep_return, ep_length = trainer._run_episode()

        assert isinstance(ep_return, float)
        assert ep_length <= 10

    def test_update_policy(self):
        env = BeliefEcologyEnv()
        trainer = RLTrainer(env)

        # fill buffer with some data
        for _ in range(10):
            trainer._buffer.add(
                state=np.zeros(BeliefEcologyEnv.STATE_DIM),
                action=np.random.randn(BeliefEcologyEnv.ACTION_DIM),
                reward=np.random.randn(),
                value=0.0,
                log_prob=-1.0,
                done=False,
            )

        p_loss, v_loss, entropy = trainer._update_policy()

        assert isinstance(p_loss, float)
        assert isinstance(v_loss, float)
        assert isinstance(entropy, float)
        assert len(trainer._buffer) == 0  # buffer cleared

    def test_evaluate(self):
        env = BeliefEcologyEnv(EnvConfig(max_steps_per_episode=5))
        trainer = RLTrainer(env)

        metrics = trainer.evaluate(num_episodes=3)

        assert "mean_return" in metrics
        assert "std_return" in metrics
        assert "mean_length" in metrics


class TestTrainingLoop:
    def test_short_training(self):
        config = TrainingConfig(
            total_episodes=5,
            max_steps_per_episode=10,
            buffer_size=20,
        )
        env = BeliefEcologyEnv(EnvConfig(max_steps_per_episode=10))
        trainer = RLTrainer(env, config)

        metrics = trainer.train()

        assert metrics.episode == 5
        assert metrics.total_steps > 0

    def test_callbacks(self):
        config = TrainingConfig(
            total_episodes=3,
            max_steps_per_episode=5,
            buffer_size=10,
        )
        env = BeliefEcologyEnv(EnvConfig(max_steps_per_episode=5))
        trainer = RLTrainer(env, config)

        episode_ends = []
        trainer.set_on_episode_end(lambda m: episode_ends.append(m.episode))

        trainer.train()

        assert len(episode_ends) == 3

    def test_early_stopping(self):
        config = TrainingConfig(
            total_episodes=100,
            max_steps_per_episode=5,
            buffer_size=10,
            target_return=1000.0,  # unreachable high target
        )
        env = BeliefEcologyEnv(EnvConfig(max_steps_per_episode=5))
        trainer = RLTrainer(env, config)

        metrics = trainer.train()

        # should run all episodes since target is unreachable
        assert metrics.episode == 100


class TestPolicySaveLoad:
    def test_save_and_load(self, tmp_path):
        env = BeliefEcologyEnv()
        trainer = RLTrainer(env)

        # modify policy
        trainer._policy_mean = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

        path = str(tmp_path / "policy.npz")
        trainer.save_policy(path)

        # load into new trainer
        trainer2 = RLTrainer(env)
        trainer2.load_policy(path)

        np.testing.assert_array_almost_equal(
            trainer._policy_mean, trainer2._policy_mean
        )
