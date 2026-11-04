from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

### Config
BOOST_SPEED_GAIN = 1

### -------- Types -----------


class Direction(IntEnum):
	UP = 0
	RIGHT = 1
	DOWN = 2
	LEFT = 3


DX = [0, 1, 0, -1]
# Pygame utilise y du haut vers le bas
DY = [-1, 0, 1, 0]


@dataclass(frozen=True)
class Player:
	x: int
	y: int
	direction: Direction = Direction.UP
	speed: int = 1
	alive: bool = True
	boost_active: bool = False


@dataclass(frozen=True)
class World:
	grid: np.ndarray
	players: list[Player]
	tick: int = 0
	rng_state: dict | None = None

	def __post_init__(self):
		for i, p in enumerate(self.players, start=1):
			if self.grid[p.y, p.x] != i:
				grid = self.grid.copy()
				grid[p.y, p.x] = i
				object.__setattr__(self, "grid", grid)


@dataclass(frozen=True)
class Action:
	p1_dir: Direction
	p2_dir: Direction
	p1_boost: bool = False
	p2_boost: bool = False


@dataclass(frozen=True)
class Alive:
	world: World


@dataclass(frozen=True)
class Dead:
	winner: str | None = None


### ----- Core -----


def get_path(player: Player, direction: Direction) -> list[tuple[int, int]]:
	"""
	Compute the path of a player
	"""

	if direction == (player.direction + 2) % 4:
		direction = player.direction

	speed: int = player.speed + (BOOST_SPEED_GAIN if player.boost_active else 0)
	cells = []
	nx, ny = player.x, player.y

	for _ in range(speed):
		nx += DX[direction]
		ny += DY[direction]
		cells.append((nx, ny))

	return cells


def move(
	player: Player, player_path: list[tuple[int, int]], world: World
) -> Player:
	for i, (x, y) in enumerate(player_path):
		if (
			not (0 <= x < world.grid.shape[1] and 0 <= y < world.grid.shape[0])
			or world.grid[y, x] != 0
		):
			last_valid_x, last_valid_y = player_path[i - 1]
			return Player(
				x=last_valid_x if i > 0 else player.x,
				y=last_valid_y if i > 0 else player.y,
				direction=player.direction,
				speed=0,
				boost_active=False,
				alive=False,
			)

	return Player(
		x=player_path[-1][0],
		y=player_path[-1][1],
		direction=player.direction,
		speed=player.speed,
		boost_active=not player.boost_active and True,
		alive=True,
	)


def step(world: World, action: Action) -> Alive | Dead:

	path_p1 = get_path(world.players[0], action.p1_dir)
	path_p2 = get_path(world.players[1], action.p2_dir)

	P1 = move(world.players[0], path_p1, world)
	P2 = move(world.players[1], path_p2, world)

	if (set(path_p1) & set(path_p2)) or (not P1.alive and not P2.alive):
		return Dead(winner=None)
	if not P1.alive:
		return Dead(winner="P2")
	if not P2.alive:
		return Dead(winner="P1")

	new_grid = world.grid.copy()
	new_grid[[y for _, y in path_p1], [x for x, _ in path_p1]] = 1
	new_grid[[y for _, y in path_p2], [x for x, _ in path_p2]] = 2

	new_world = World(
		grid=new_grid,
		players=[P1, P2],
		tick=world.tick + 1,
		rng_state=world.rng_state,
	)
	return Alive(world=new_world)


### ------- Setup --------


def reset(screen_dim: tuple[int, int], seed: int) -> World:
	grid = np.zeros(screen_dim, dtype=np.int8)

	P1 = Player(
		x=screen_dim[0] // 2, y=screen_dim[1] - 6, direction=Direction.DOWN
	)
	P2 = Player(x=screen_dim[0] // 2, y=5, direction=Direction.UP)

	grid[P1.y, P1.x] = 1
	grid[P2.y, P2.x] = 2
	rng_state = None

	if seed is not None:
		rng = np.random.default_rng(seed)
		rng_state = rng.bit_generator.state

	return World(players=[P1, P2], tick=0, grid=grid, rng_state=rng_state)
