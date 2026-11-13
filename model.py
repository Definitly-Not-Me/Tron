from collections import deque

import numpy as np

from engine import BOOST_SPEED_GAIN, DX, DY, Direction, World

N_BINS_DIST = 4
N_BINS_COORD = 3
N_FEATURES = 31


def _voronoi(
	grid: np.ndarray, my_head: tuple[int, int], opp_head: tuple[int, int]
) -> tuple[int, int, int]:
	"""
	Note: Here
	    * 0 means claimable my 'me'
	    * 1 means claimable by 'opponent'
	    * -1 means unclaimed
	"""
	grid_h, grid_w = grid.shape
	owner = np.full((grid_h, grid_w), -1, dtype=np.int8)
	q = deque()
	total_open = int(np.sum(grid == 0))

	owner[my_head[1], my_head[0]] = 0
	q.append((my_head[0], my_head[1], 0))

	owner[opp_head[1], opp_head[0]] = 1
	q.append((opp_head[0], opp_head[1], 1))

	while q:
		x, y, o = q.popleft()
		for dir in list(Direction):
			nx = x + DX[dir]
			ny = y + DY[dir]

			if not (0 <= nx < grid_w and 0 <= ny < grid_h):
				continue

			contested = abs(nx - my_head[0]) + abs(ny - my_head[1]) == abs(
				nx - opp_head[0]
			) + abs(ny - opp_head[1])

			if contested:
				continue

			if owner[ny, nx] != -1:
				continue

			if grid[ny, nx] != 0:
				continue

			owner[ny, nx] = o
			q.append((nx, ny, o))

	reachable_by_me = int(np.sum(owner == 0))
	reachable_by_opp = int(np.sum(owner == 1))

	return (reachable_by_me, reachable_by_opp, total_open)


def _one_hot(idx, n):
	vec = np.zeros(n, dtype=np.float32)
	vec[idx] = 1.0

	return vec


def get_valid_moves(
	grid: np.ndarray, pos: tuple[int, int], boost_active: bool = False
) -> list[int]:
	x, y = pos
	blocked = [0 for _ in range(4)]
	speed = 1 if not boost_active else BOOST_SPEED_GAIN
	for dir in list(Direction):
		nx, ny = x, y
		for _ in range(speed):
			nx += DX[dir]
			ny += DY[dir]

			if (
				not (0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0])
				or grid[ny, nx] != 0
			):
				blocked[dir] = 1
				break

	return [not b for b in blocked]


def _bin_coord(delta, size) -> int:
	rel = (delta + (size - 1)) / (2 * (size - 1))

	return min(int(rel * N_BINS_COORD), N_BINS_COORD - 1)


def _bin_dist(dist, max_dist) -> int:
	if max_dist == 0:
		return 0

	rel = dist / max_dist
	return min(int(rel * N_BINS_DIST), N_BINS_DIST - 1)


def get_features_from(world: World, playerid: int) -> np.ndarray:
	"""
	component of the Feature vector
	  * f1: reachable_by_me ratio, reachable by opp ratio
	  * f2: me direction
	  * f3: opp direction
	  * f4: boost active (opp, me)
	  * f5: next cells occupation
	  * f6: bined relative coordinate opp
	  * f7: bined distance me-opp
	  * f8: bined distance to closest obstacle
	  * f9: normalized grid size
	  * f10: my boost available
	"""

	me = next((p for p in world.players if p.id == playerid), None)
	opp = next((p for p in world.players if p.id != playerid), None)

	if me is None or opp is None:
		raise ValueError(f"Player with id {playerid} not found.")

	grid = world.grid
	grid_h, grid_w = grid.shape
	my_reach, opp_reach, total = _voronoi(
		grid=grid, my_head=(me.x, me.y), opp_head=(opp.x, opp.y)
	)
	total = max(total, 1)
	f1 = np.array([my_reach / total, opp_reach / total], dtype=np.float32)

	f2 = _one_hot(me.direction, len(Direction))

	f3 = _one_hot(opp.direction, len(Direction))

	f4 = np.array([me.boost_active, opp.boost_active], dtype=np.float32)

	free_cells = get_valid_moves(grid=grid, pos=(me.x, me.y))

	# -1 = left of current dir, 0 = current dir, +1 = right of current dir
	f5 = np.array(
		[free_cells[(me.direction + r) % 4] for r in [-1, 0, 1]],
		dtype=np.float32,
	)

	dx = me.x - opp.x
	dy = me.y - opp.y
	f6 = np.concatenate(
		[
			_one_hot(_bin_coord(dx, grid_w), N_BINS_COORD),
			_one_hot(_bin_coord(dy, grid_h), N_BINS_COORD),
		]
	)

	# Manhattan distance
	dist = abs(dx) + abs(dy)

	# -1 becausee grid include wall
	max_dist = (grid_h - 1) + (grid_w - 1)

	f7 = _one_hot(_bin_dist(dist=dist, max_dist=max_dist), N_BINS_DIST)

	obstacles_mask = (grid != 0) & (grid != playerid)
	if np.any(obstacles_mask):
		obstacles_cells = np.argwhere(obstacles_mask)
		dists = np.abs(obstacles_cells[:, 1] - me.x) + np.abs(
			obstacles_cells[:, 0] - me.y
		)
		closest_obstacle = int(np.min(dists))
	else:
		closest_obstacle = max_dist
	f8 = _one_hot(_bin_dist(closest_obstacle, max_dist), N_BINS_DIST)

	f9 = np.array([grid_h * grid_w / 400], dtype=np.float32)

	f10 = np.array([1.0 if me.boost_remaining > 0 else 0.0], dtype=np.int32)

	return np.concatenate([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]).astype(
		np.float32
	)
