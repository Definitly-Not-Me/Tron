import numpy as np
import pytest

from engine import (
	Action,
	Alive,
	Dead,
	Direction,
	Player,
	World,
	get_path,
	move,
	reset,
	step,
)


class TestGetPath:
	def test_basic_move(self):
		p = Player(x=5, y=5, direction=Direction.RIGHT)
		path = get_path(p, Direction.RIGHT)
		assert path == [(6, 5)]

	def test_boost_double_speed(self):
		p = Player(x=5, y=5, direction=Direction.RIGHT, boost_active=True)
		path = get_path(p, Direction.RIGHT)

		assert path == [(6, 5), (7, 5)]

	def test_reversal_no_op(self):
		p = Player(x=5, y=5, direction=Direction.RIGHT)
		path = get_path(p, Direction.LEFT)
		assert path == [(6, 5)]

	def test_turn(self):
		p = Player(x=5, y=5, direction=Direction.RIGHT)
		path = get_path(p, Direction.UP)
		assert path == [(5, 4)]


class TestMove:
	@pytest.fixture()
	def world(self, h=20, w=20):
		return World(
			grid=np.zeros((h, w), dtype=np.int8),
			players=[
				Player(x=5, y=10, direction=Direction.RIGHT),
				Player(x=14, y=10, direction=Direction.LEFT),
			],
		)

	def test_successfull_move(self, world):
		p = world.players[0]
		path = get_path(p, Direction.RIGHT)
		new_p = move(p, path, world)
		assert new_p.x == 6
		assert new_p.y == 10
		assert new_p.alive is True

	def test_wall_death(self, world):
		p = Player(x=19, y=15, direction=Direction.RIGHT)
		path = get_path(p, Direction.RIGHT)
		new_p = move(p, path, world)

		assert new_p.alive == False
		assert new_p.x == 19

	def test_cannot_pass_through_walls(self, world):
		p = Player(x=19, y=15, direction=Direction.UP)
		path = get_path(p, Direction.RIGHT)
		new_p = move(p, path, world)

		assert new_p.x == 19, (
			"If a player turn into a wall while hugging it, keep him in place"
		)

	def test_player_collision(self, world):
		p = Player(x=3, y=10, direction=Direction.RIGHT)
		path = get_path(p, Direction.RIGHT)  # [(4,10)]
		new_p = move(p, path, world)
		assert new_p.alive is True

		path = get_path(new_p, Direction.RIGHT)  # [(5,10)]
		new_p = move(new_p, path, world)

		assert new_p.alive is False


class TestStep:
	@pytest.fixture()
	def world(self, h=20, w=20):
		return World(
			grid=np.zeros((h, w), dtype=np.int8),
			players=[
				Player(x=5, y=10, direction=Direction.RIGHT),
				Player(x=19, y=10, direction=Direction.LEFT),
			],
		)

	def test_basic_move(self, world):
		action = Action(p1_dir=Direction.RIGHT, p2_dir=Direction.LEFT)
		result = step(world, action)

		assert isinstance(result, Alive)
		assert result.world.tick == 1, "Tick should be updated at each step"
		assert result.world.players[0].x == 6, (
			"Player position should be updated at each step"
		)
		assert result.world.players[1].x == 18, (
			"Player position should be updated at each step"
		)

	def test_grid_updated(self, world):
		action = Action(p1_dir=Direction.RIGHT, p2_dir=Direction.LEFT)
		result = step(world, action)

		assert isinstance(result, Alive)
		# BUG: step() uses [x, y] indexing but numpy is [row=y, col=x]
		# This test will catch that bug
		assert result.world.grid[10, 6] == 1, "Grid should be updated each step"
		assert result.world.grid[10, 18] == 2, (
			"Grid should be updated each step"
		)

	def win_by_wall_death(self, world):
		action = Action(p1_dir=Direction.RIGHT, p2_dir=Direction.LEFT)
		result = step(world, action)

		assert isinstance(result, Dead), (
			"Player should die when running in walls"
		)
		assert result.winner == "P1"

	def test_draw(self):
		grid = np.zeros((10, 10), dtype=np.int8)
		world = World(
			grid=grid,
			players=[
				Player(x=4, y=5, direction=Direction.RIGHT),
				Player(x=5, y=5, direction=Direction.LEFT),
			],
		)
		action = Action(p1_dir=Direction.RIGHT, p2_dir=Direction.LEFT)
		result = step(world, action)
		assert isinstance(result, Dead)
		assert result.winner is None, "Heads-on collision should result in draw"

	def test_trail_death(self, world):
		grid = np.zeros((10, 10), dtype=np.int8)
		grid[5, 7] = 2  # P2's trail at (x=7, y=5)
		world = World(
			grid=grid,
			players=[
				Player(x=6, y=5, direction=Direction.RIGHT),
				Player(x=3, y=5, direction=Direction.LEFT),
			],
		)
		action = Action(p1_dir=Direction.RIGHT, p2_dir=Direction.LEFT)
		result = step(world, action)
		assert isinstance(result, Dead), (
			"Player should die when steping on a trail"
		)
		assert result.winner == "P2", (
			"Last standing Player should be declared winner"
		)


class TestReset:
	def initial_positions(self):
		world = reset((20, 20), 3)
		p1, p2 = world.players

		assert world.grid[p1.y, p1.x] == 1, "Initial position is not marked"
		assert world.grid[p2.y, p2.x] == 2, "Initial position is not marked"
		assert p1.alive is True, "Player cannot spawn dead!"
		assert p2.alive is True
