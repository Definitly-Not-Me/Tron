from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from config import ACTION_MAP, MIN_GRID, N_FEATURES
from src.bots import Controller
from src.engine import Action, Alive, Autopsy, Dead, Direction, Player
from src.engine import (
	reset as init_world,
)
from src.engine import step as get_outcome
from src.model import get_features_from
from src.render import Renderer


class TronEnv(gym.Env):
	metadata = {"render_modes": ["human"], "render_fps": 30}

	def __init__(
		self,
		opp: Controller,
		max_ticks: int = 1000,
		render_mode: str | None = None,
		starting_size: tuple[int, int] = (44, 44),
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
		self._action_map = ACTION_MAP

		assert (
			render_mode is None or render_mode in self.metadata["render_modes"]
		)
		self.render_mode = render_mode
		print("Render mode", render_mode)
		self.renderer = Renderer(self.world, render_mode or "")
		self.max_ticks = max_ticks
		self.clock = None

	def _get_obs(self):
		f = get_features_from(self.world, self.agentid)

		return np.nan_to_num(f).astype(np.float32)

	def _get_info(self, dead_world: Dead):
		return {
			"terminal_cause": dead_world.autopsy,
			"steps": dead_world.world.tick,
			"agent_survived": dead_world.world.players[self.agentid - 1].alive,
			"grid_size": self.grid_size,
			"opponent": self.opp.name,
		}

	def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
		super().reset(seed=seed)

		self.grid_size = tuple(
			max(
				int(
					d
					* (
						1
						+ self.np_random.integers(-1, 2)
						* self.np_random.random()
					)
				),
				MIN_GRID,
			)
			for d in self.starting_size
		)
		self.world = init_world(self.grid_size, seed=seed)  # pyright: ignore

		# TODO: find a way to render without spamming
		# self.renderer = Renderer(self.world)
		observation = self._get_obs()
		info = {
			"grid_size": self.grid_size,
			"opponent": self.opp.name,
		}

		return observation, info

	def get_agent(self) -> Player:
		return self.world.players[self.agentid - 1]

	def step(self, action: int) -> tuple[Any, ...]:
		direction, boost = self._action_map[action]
		agent = self.get_agent()
		direction = Direction((int(agent.direction) + direction) % 4)
		winner = 0

		opp_move, _ = self.opp.choose_move(
			world=self.world, bot_id=1, rng=self.np_random
		)

		engine_action = Action(
			p1_dir=opp_move, p2_dir=direction, p2_boost=boost
		)
		outcome = get_outcome(self.world, engine_action)
		self.world = outcome.world

		match outcome:
			case Dead(_, _, winner_id):
				terminated = True
				info = self._get_info(outcome)
				winner = winner_id
			case _:
				terminated = False
				info = {
					"grid_size": self.grid_size,
					"opponent": self.opp.name,
				}

		truncated = self.world.tick >= self.max_ticks
		observation = self._get_obs()
		reward = self._get_reward(outcome, observation)
		self._render_frame(winner, terminated)

		return observation, reward, terminated, truncated, info

	def _render_frame(
		self, winner_id: int | None, terminated: bool | None = None
	):

		if self.render_mode == "human":
			if self.clock is None and self.renderer is not None:
				self.clock = pygame.time.Clock()
			if not terminated:
				self.renderer.draw(self.world)
			else:
				# TODO: find a way to get real winner id
				self.renderer.draw_game_over(self.world, winner_id)  # pyright: ignore

			pygame.display.flip()
			self.clock.tick(self.metadata["render_fps"])

	def _get_reward(self, outcome: Dead | Alive, observation) -> float:

		match outcome:
			case Dead(_, cause, winner_id) if winner_id is not None:
				if winner_id == self.agentid:
					return 1
				match cause:
					case Autopsy.OUT_OF_BOUND:
						base = -1
					case Autopsy.SUICIDE:
						base = -0.75
					case _:
						base = -0.5

				return base

			case Dead(_, cause, winner_id) if winner_id is None:
				# If you can't beat them, make them give up :)
				return 0.1
			case _:
				reacheable_by_me_ratio = observation[0]
				reachable_by_opp_ratio = observation[1]

				reward = 0.05 * (
					reacheable_by_me_ratio - reachable_by_opp_ratio
				)

		return reward
