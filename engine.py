from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from enum import IntEnum

import numpy as np

### Config
BOOST_SPEED_GAIN = 2
STARTING_BOOST: int = 8

### -------- Types -----------


class Direction(IntEnum):
	UP = 0
	RIGHT = 1
	DOWN = 2
	LEFT = 3


class Autopsy(IntEnum):
	SUICIDE = 0
	OUT_OF_BOUND = 1
	TAIL_CRASH = 2
	HEAD_ON = 3


DX = [0, 1, 0, -1]
# Pygame utilise y du haut vers le bas
DY = [-1, 0, 1, 0]


@dataclass(frozen=True)
class Player:
	x: int
	y: int
	id: int
	direction: Direction = Direction.UP
	speed: int = 1
	alive: bool = True
	boost_active: bool = False
	boost_remaining: int = STARTING_BOOST


@dataclass(frozen=True)
class World:
	grid: np.ndarray
	players: list[Player]
	tick: int = 0
	rng_state: dict | None = None

	def __post_init__(self):
		for p in self.players:
			if self.grid[p.y, p.x] != p.id:
				grid = self.grid.copy()
				grid[p.y, p.x] = p.id
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
	world: World
	autopsy: Autopsy | None
	winner_id: int | None = None


@dataclass(frozen=True)
class Move:
	player: Player
	cause: Autopsy | None
	path: list[tuple[int, int]]


### ----- Core -----


def resolve_direction(direction: Direction, previous_direction) -> Direction:
	if direction == (previous_direction + 2) % 4:
		direction = previous_direction
	return direction


def get_path(player: Player, direction: Direction) -> list[tuple[int, int]]:
	"""
	Compute the path of a player
	"""
	direction = resolve_direction(direction, player.direction)

	speed: int = player.speed + (
		BOOST_SPEED_GAIN
		if player.boost_active and player.boost_remaining
		else 0
	)
	cells = []
	nx, ny = player.x, player.y

	for _ in range(speed):
		nx += DX[direction]
		ny += DY[direction]
		cells.append((nx, ny))

	if not cells:
		cells = [(player.x, player.y)]
		warnings.warn(
			f"Player {player.id}'s path didn't resolve. Cannot step further",
			stacklevel=2,
		)

	return cells


def move(player: Player, direction: Direction, world: World) -> Move:
	direction = resolve_direction(direction, player.direction)
	player_path = get_path(player, direction)
	cause = None

	for i, (x, y) in enumerate(player_path):
		if not (0 <= x < world.grid.shape[1] and 0 <= y < world.grid.shape[0]):
			cause = Autopsy.OUT_OF_BOUND
		elif world.grid[y][x] != 0:
			cause = (
				Autopsy.TAIL_CRASH
				if world.grid[y][x] != player.id
				else Autopsy.SUICIDE
			)

		if cause is not None:
			last_valid_x, last_valid_y = (
				player_path[i - 1] if i > 0 else (player.x, player.y)
			)
			return Move(
				Player(
					x=last_valid_x,
					y=last_valid_y,
					direction=direction,
					speed=0,
					boost_active=False,
					alive=False,
					id=player.id,
				),
				cause,
				player_path[:i],
			)

	# print(f"Boost remaining for Player {player.id} = {player.boost_remaining}")

	return Move(
		Player(
			x=player_path[-1][0],
			y=player_path[-1][1],
			direction=direction,
			speed=player.speed,
			boost_active=False,
			alive=True,
			id=player.id,
			boost_remaining=player.boost_remaining - 1
			if player.boost_active and player.boost_remaining > 0
			else player.boost_remaining,
		),
		None,
		player_path,
	)


def step(world: World, action: Action) -> Alive | Dead:
	p1 = replace(world.players[0], boost_active=action.p1_boost)
	p2 = replace(world.players[1], boost_active=action.p2_boost)

	move_p1 = move(p1, action.p1_dir, world)
	move_p2 = move(p2, action.p2_dir, world)

	P1 = move_p1.player
	P2 = move_p2.player

	new_grid = world.grid.copy()
	new_grid[[y for _, y in move_p1.path], [x for x, _ in move_p1.path]] = 1
	new_grid[[y for _, y in move_p2.path], [x for x, _ in move_p2.path]] = 2

	new_world = World(
		grid=new_grid,
		players=[P1, P2],
		tick=world.tick + 1,
		rng_state=world.rng_state,
	)

	# print(f">> P1{(P1.x, P1.y)} | \t P2{(P2.x, P2.y)}")
	# print("-" * 15)

	if not P1.alive and not P2.alive:
		return Dead(winner_id=None, world=new_world, autopsy=Autopsy.HEAD_ON)
	if not P1.alive:
		return Dead(winner_id=2, world=new_world, autopsy=move_p1.cause)
	if not P2.alive:
		return Dead(winner_id=1, world=new_world, autopsy=move_p2.cause)

	return Alive(world=new_world)


### ------- Setup --------


def reset(arena_dim: tuple[int, int], seed: int | None = None) -> World:
	W, H = arena_dim
	grid = np.zeros((H, W), dtype=np.int8)

	P1 = Player(
		id=1, x=arena_dim[0] // 2, y=arena_dim[1] - 6, direction=Direction.UP
	)
	P2 = Player(id=2, x=arena_dim[0] // 2, y=5, direction=Direction.DOWN)

	grid[P1.y, P1.x] = 1
	grid[P2.y, P2.x] = 2
	rng_state = None

	if seed is not None:
		rng = np.random.default_rng(seed)
		rng_state = rng.bit_generator.state

	return World(players=[P1, P2], tick=0, grid=grid, rng_state=rng_state)
