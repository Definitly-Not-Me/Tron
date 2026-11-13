import numpy as np
import pytest

from engine import (
	Direction,
	Player,
	World,
)
from model import (
	N_BINS_COORD,
	N_BINS_DIST,
	N_FEATURES,
	_bin_coord,
	_bin_dist,
	_one_hot,
	_voronoi,
	get_features_from,
	get_valid_moves,
)


class TestOneHot:
	def test_one_hot_basic(self):
		v = _one_hot(3, 4)

		assert v.tolist() == [0.0, 0.0, 0.0, 1.0], (
			"The resulting list doesn't match what's expected"
		)


class TestBinCoord:
	def test_bin_coord_extremes(self):
		assert _bin_coord(9, 10) == N_BINS_COORD - 1
		assert _bin_coord(-10, 10) == 0
		assert _bin_coord(0, 10) == N_BINS_COORD // 2


class TestBinDist:
	def test_bin_dist(self):
		assert _bin_dist(0, 10) == 0
		assert _bin_dist(10, 10) == N_BINS_DIST - 1
		assert _bin_dist(5, 10) == N_BINS_DIST // 2


class TestValidMoves:
	@pytest.fixture()
	def grid(self):
		return np.zeros((5, 5), dtype=np.int8)

	def test_valid_moves_open_grid(self, grid):
		valids = get_valid_moves(grid, (2, 2))

		assert valids == [1, 1, 1, 1], (
			"In the middle of an open grid, all moves should be valid"
		)

	def test_valid_moves_obstacle_above(self, grid):
		grid[0, 2] = 1
		valids = get_valid_moves(grid, (2, 1))

		assert valids[Direction.UP] == 0, (
			"Cell above (2,0) should not be valid when at pos (2,1)"
		)
		assert valids[Direction.RIGHT] == 1
		assert valids[Direction.DOWN] == 1
		assert valids[Direction.LEFT] == 1

	def test_valid_moves_during_boost(self, grid):
		grid[0, 2] = 1
		valids = get_valid_moves(grid, (2, 2), boost_active=True)

		assert valids[Direction.UP] == 0, "Shouldn't be free during boost"
		assert valids[Direction.RIGHT] == 1, "Should be free"

	def test_valid_moves_edge_of_grid(self, grid):
		valids = get_valid_moves(grid, (0, 0))

		assert valids[Direction.UP] == 0, "Should be Out of Bounds"
		assert valids[Direction.LEFT] == 0, "Should be Out of Bounds"
		assert valids[Direction.RIGHT] == 1, "Should be free"
		assert valids[Direction.DOWN] == 1


class TestVoronoi:
	@pytest.fixture()
	def grid(self):
		return np.zeros((5, 5), dtype=np.int8)

	def test_voronoi_adjacent(self, grid):
		my_reach, opp_reach, total = _voronoi(grid, (0, 0), (4, 4))

		assert my_reach == opp_reach, (
			"Should be equal since they have equal chances of winning"
		)
		assert total == 25

	def test_voronoi_symmetric(self, grid):
		my_reach, opp_reach, total = _voronoi(grid, (0, 2), (4, 2))
		assert my_reach == opp_reach

	def test_voronoi_against_obstacle(self, grid):
		grid = np.zeros((3, 3), dtype=np.int8)
		grid[:, 1] = 1
		my_reach, opp_reach, total = _voronoi(grid, (0, 1), (2, 2))
		print(grid)

		assert (
			my_reach == 3
		)  # Normally 2 but players head is not placed during test
		assert opp_reach == 3

	def test_voronoi_no_open_cells(self):
		grid = np.ones((3, 3), dtype=np.int8)
		my_reach, opp_reach, total = _voronoi(grid, (1, 1), (0, 0))

		assert total == 0


class TestGetFeatures:
	@pytest.fixture()
	def world(self):
		grid = np.zeros((5, 5), dtype=np.int8)
		p1 = Player(2, 2, id=1, boost_active=True)
		p2 = Player(0, 0, id=2)

		return World(grid, [p1, p2])

	def test_features_shape(self, world):
		features = get_features_from(world, 1)
		# from f1 to f9
		expected_dim = 2 + 4 + 4 + 2 + 3 + 6 + 4 + 4 + 1 + 1

		assert features.shape == (expected_dim,)
		assert expected_dim == N_FEATURES, (
			"Remember to update N_FEATURES after each modification to the features vector"
		)

	def test_feature_my_direction(self, world):
		f = get_features_from(world, 1)

		# default direction is up
		assert f[2:6].tolist() == [1.0, 0.0, 0.0, 0.0]

	def test_feature_boost(self, world):
		f = get_features_from(world, 1)

		assert f[10:12].tolist() == [1.0, 0.0]

	def test_raises_for_missing_player(self, world):
		with pytest.raises(ValueError, match="not found"):
			get_features_from(world, 3)

	def test_feature_valid_move(self, world):
		f = get_features_from(world, 1)

		# print(f)
		assert f[12:15].tolist() == [1.0, 1.0, 1.0]
