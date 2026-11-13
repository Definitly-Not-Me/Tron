# test_tron_env.py
import numpy as np
import pytest

from bots import RandomBot  # or whatever your default controller is
from gym_env import TronEnv


@pytest.fixture
def env():
	e = TronEnv(opp=RandomBot(), max_ticks=100, render_mode=None)
	yield e
	e.close()


class TestReset:
	def test_seed_determinism(self):
		e1 = TronEnv(opp=RandomBot(), render_mode=None)
		e2 = TronEnv(opp=RandomBot(), render_mode=None)
		obs1, _ = e1.reset(seed=123)
		obs2, _ = e2.reset(seed=123)
		np.testing.assert_array_equal(obs1, obs2)
		e1.close()
		e2.close()

	def test_grid_size_varies(self, env):
		sizes = set()
		for _ in range(10):
			_, info = env.reset(seed=None)
			sizes.add(info["grid_size"])
		assert len(sizes) > 1  # should not be identical every time


class TestStep:
	def test_all_actions_valid(self, env):
		env.reset(seed=0)
		for action in range(env.action_space.n):
			print("Action", action)
			obs, reward, terminated, truncated, info = env.step(action)

			print("Obs: ", obs)
			# if terminated:
			# 	print("Env terminated early! Restarting...", flush=True)
			# 	env.reset(seed=0)
			assert env.observation_space.contains(obs), (
				"Observation space should contain recent observations"
			)

	def test_action_out_of_range_raises(self, env):
		env.reset(seed=0)
		with pytest.raises((IndexError, KeyError)):
			env.step(6)

	def test_truncation_at_max_ticks(self):
		env = TronEnv(opp=RandomBot(), max_ticks=5, render_mode=None)
		env.reset(seed=0)
		truncated = False
		for _ in range(10):
			_, _, terminated, truncated, _ = env.step(2)  # straight
			if terminated or truncated:
				break
		assert truncated or terminated
		env.close()


class TestHeadless:
	def test_full_episode_completes(self):
		env = TronEnv(opp=RandomBot(), max_ticks=200, render_mode=None)
		env.reset(seed=42)
		steps = 0
		for _ in range(200):
			_, _, term, trunc, _ = env.step(env.action_space.sample())
			steps += 1
			if term or trunc:
				break
		assert steps <= 200
		assert term or trunc
		env.close()
