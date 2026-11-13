from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from bots import Controller
from engine import (
	Action,
	Alive,
	Dead,
	Direction,
	Player,
)
from engine import (
	reset as init_world,
)
from engine import (
	step as get_outcome,
)
from model import N_FEATURES, get_features_from
from render import Renderer


class TronEnv(gym.Env):
	metadata = {"render_modes": ["human"], "render_fps": 30}

	def __init__(
		self,
		opp: Controller,
		max_ticks: int = 500,
		render_mode: str | None = None,
		starting_size: tuple[int, int] = (30, 30),
	) -> None:
		super().__init__()

		self.starting_size = starting_size
		self.grid_size = starting_size
		self.world = init_world(arena_dim=self.grid_size)  # pyright: ignore
		self.opp = opp
		self.agentid = 2

		self.observation_space = spaces.Box(
			-np.inf, np.inf, shape=(N_FEATURES,), dtype=np.float32
		)

		self.action_space = spaces.Discrete(6)
		self._action_map = {
			0: (-1, False),
			1: (-1, True),
			2: (0, False),
			3: (0, True),
			4: (1, False),
			5: (1, True),
		}

		assert (
			render_mode is None or render_mode in self.metadata["render_modes"]
		)
		self.render_mode = render_mode
		self.renderer = Renderer(self.world)
		self.max_ticks = max_ticks
		self.clock = None

	def _get_obs(self):
		f = get_features_from(self.world, self.agentid)

		return np.nan_to_num(f).astype(np.float32)

	def _get_info(self, dead_world: Dead):
		return {
			"terminal_cause": dead_world.autopsy,
			"steps": dead_world.world.tick,
			"agent_survive": dead_world.world.players[self.agentid - 1].alive,
			"grid_size": self.grid_size,
			"opponent": self.opp.name(),
		}

	def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
		super().reset(seed=seed)

		self.grid_size = tuple(
			int(d * (1 + 2 * self.np_random.random()))
			for d in self.starting_size
		)
		self.world = init_world(self.grid_size, seed=seed)  # pyright: ignore
		self.renderer = Renderer(self.world)
		observation = self._get_obs()
		info = {
			"grid_size": self.grid_size,
			"opponent": self.opp.name(),
		}

		return observation, info

	def get_agent(self) -> Player:
		return self.world.players[self.agentid - 1]

	def step(self, action: int) -> tuple[Any, ...]:
		direction, boost = self._action_map[action]
		agent = self.get_agent()
		direction = Direction((int(agent.direction) + direction) % 4)

		opp_move = self.opp.choose_move(
			world=self.world, bot_id=1, rng=self.np_random
		)

		engine_action = Action(
			p1_dir=opp_move, p2_dir=direction, p2_boost=boost
		)
		print(
			f"P1_dir: {int(opp_move)}\t | P2_dir: {int(direction)}",
		)
		outcome = get_outcome(self.world, engine_action)
		self.world = outcome.world

		match outcome:
			case Dead():
				terminated = True
				info = self._get_info(outcome)

			case _:
				terminated = False
				info = {
					"grid_size": self.grid_size,
					"opponent": self.opp.name(),
				}

		reward = self._get_reward(outcome)
		truncated = self.world.tick >= self.max_ticks
		observation = self._get_obs()
		self._render_frame(terminated)

		return observation, reward, terminated, truncated, info

	def _render_frame(self, terminated: bool | None = None):

		if self.render_mode == "human":
			if self.clock is None:
				self.clock = pygame.time.Clock()
			if not terminated:
				self.renderer.draw(self.world)
			else:
				# TODO: find a way to get real winner id
				self.renderer.draw_game_over(self.world, 0)  # pyright: ignore

			self.clock.tick(self.metadata["render_fps"])

	def _get_reward(self, outcome: Dead | Alive) -> int:
		match outcome:
			case Dead(winner_id) if winner_id is not None:
				reward = 1 if winner_id == self.agentid else -1
			case Dead(winner_id) if winner_id is None:
				reward = 0
			case _:
				reward = 0

		return reward
