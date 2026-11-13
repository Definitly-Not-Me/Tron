import numpy as np
import pytest

from engine import (
	Action,
	Alive,
	Autopsy,
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
		p = Player(x=5, y=5, id=1, direction=Direction.RIGHT)
		path = get_path(p, Direction.RIGHT)
		assert path == [(6, 5)]

	def test_boost_triple_speed(self):
		p = Player(x=5, y=5, id=1, direction=Direction.RIGHT, boost_active=True)
		path = get_path(p, Direction.RIGHT)

		assert path == [(6, 5), (7, 5), (8, 5)]

	def test_reversal_no_op(self):
		p = Player(x=5, y=5, id=1, direction=Direction.RIGHT)
		path = get_path(p, Direction.LEFT)
		assert path == [(6, 5)]

	def test_turn(self):
		p = Player(x=5, y=5, id=1, direction=Direction.RIGHT)
		path = get_path(p, Direction.UP)
		assert path == [(5, 4)]


class TestMove:
	@pytest.fixture()
	def world(self, h=20, w=20):
		return World(
			grid=np.zeros((h, w), dtype=np.int8),
			players=[
				Player(x=5, y=10, id=1, direction=Direction.RIGHT),
				Player(x=14, y=10, id=2, direction=Direction.LEFT),
			],
		)

	def test_successfull_move(self, world):
		p = world.players[0]
		result = move(p, Direction.RIGHT, world)

		assert result.player.x == 6
		assert result.player.y == 10
		assert result.player.alive is True
		assert result.cause is None
		assert result.path == [(6, 10)]

	def test_wall_death(self, world):
		p = Player(x=19, y=15, id=1, direction=Direction.RIGHT)
		result = move(p, Direction.RIGHT, world)

		assert result.player.alive == False
		assert result.player.x == 19
		assert result.cause is Autopsy.OUT_OF_BOUND
		assert result.path == []

	def test_cannot_pass_through_walls(self, world):
		p = Player(x=19, y=15, id=1, direction=Direction.UP)
		result = move(p, Direction.RIGHT, world)

		assert result.player.x == 19, (
			"If a player turn into a wall while hugging it, keep him in place"
		)
		assert result.player.alive == False
		assert result.cause is Autopsy.OUT_OF_BOUND

	def test_player_collision(self, world):
		p = Player(x=3, y=10, id=1, direction=Direction.RIGHT)
		result = move(p, Direction.RIGHT, world)
		assert result.player.alive is True

		result = move(result.player, Direction.RIGHT, world)

		assert result.player.alive is False
		assert result.cause is Autopsy.SUICIDE


class TestStep:
	@pytest.fixture()
	def world(self, h=20, w=20):
		return World(
			grid=np.zeros((h, w), dtype=np.int8),
			players=[
				Player(x=5, y=10, id=1, direction=Direction.RIGHT),
				Player(x=19, y=19, id=2, direction=Direction.LEFT),
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
		assert result.world.grid[10, 6] == 1, "Grid should be updated each step"
		assert result.world.grid[19, 18] == 2, (
			"Grid should be updated each step"
		)

	def test_win_by_wall_death(self, world):
		action = Action(p1_dir=Direction.RIGHT, p2_dir=Direction.DOWN)
		result = step(world, action)

		assert isinstance(result, Dead), (
			"Player should die when running in walls"
		)
		assert result.winner_id == 1

	def test_draw(self):
		grid = np.zeros((10, 10), dtype=np.int8)
		world = World(
			grid=grid,
			players=[
				Player(x=4, y=5, id=1, direction=Direction.RIGHT),
				Player(x=5, y=5, id=2, direction=Direction.LEFT),
			],
		)
		action = Action(p1_dir=Direction.RIGHT, p2_dir=Direction.LEFT)
		result = step(world, action)
		assert isinstance(result, Dead)
		assert result.winner_id is None, (
			"Heads-on collision should result in draw"
		)
		assert result.autopsy is Autopsy.HEAD_ON

	def test_trail_death(self, world):
		grid = np.zeros((10, 10), dtype=np.int8)
		grid[5, 7] = 2  # P2's trail at (x=7, y=5)
		world = World(
			grid=grid,
			players=[
				Player(x=6, y=5, id=1, direction=Direction.RIGHT),
				Player(x=3, y=5, id=2, direction=Direction.LEFT),
			],
		)
		action = Action(p1_dir=Direction.RIGHT, p2_dir=Direction.LEFT)
		result = step(world, action)
		assert isinstance(result, Dead), (
			"Player should die when steping on a trail"
		)
		assert result.winner_id == 2, (
			"Last standing Player should be declared winner"
		)
		assert result.autopsy is Autopsy.TAIL_CRASH


class TestReset:
	def test_initial_positions(self):
		world = reset((20, 20), 3)
		p1, p2 = world.players

		assert world.grid[p1.y, p1.x] == 1, "Initial position is not marked"
		assert world.grid[p2.y, p2.x] == 2, "Initial position is not marked"
		assert p1.id == 1, "P1 should spawn with id 1"
		assert p2.id == 2, "P2 should spawn with id 2"
		assert p1.alive is True, "Player cannot spawn dead!"
		assert p2.alive is True

	def test_numpy_grid_shape_match_arena(self):
		world = reset((30, 20), 3)

		assert world.grid.shape == (20, 30)
