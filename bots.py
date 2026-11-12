import random
from collections import deque

from engine import DX, DY, Direction, Player, World, move


def _find(world: World, playerid: int) -> Player:
	return next(p for p in world.players if p.id == playerid)


class Controller:
	def choose_move(
		self, world: World, bot_id: int, rng: random.Random
	) -> Direction:
		raise NotImplementedError

	def branches(self, me: Player) -> list[Direction]:
		return [Direction((me.direction + r) % 4) for r in [-1, 0, 1]]

	def legal_moves(self, world: World, bot_id: int) -> list[Direction]:
		me = _find(world, bot_id)

		return [d for d in self.branches(me) if move(me, d, world).player.alive]

	# @property
	def name(self):
		raise NotImplementedError


## ------ Bots ---------


class RandomBot(Controller):
	"""
	Level 0
	"""

	def choose_move(
		self, world: World, bot_id: int, rng: random.Random
	) -> Direction:
		legal = self.legal_moves(world, bot_id)

		return rng.choice(legal) if legal else _find(world, bot_id).direction

	def name(self):
		return "Randombot"


class NaiveBot(Controller):
	"""
	Level 1
	"""

	def choose_move(
		self, world: World, bot_id: int, rng: random.Random
	) -> Direction:
		me = _find(world, bot_id)
		legal = self.legal_moves(world, bot_id)

		if not legal:
			return me.direction

		if me.direction in legal:
			return me.direction

		return rng.choice(legal)

	def name(self):
		return "NaiveBot"


def _open_distance(world: World, me: Player, dir: Direction) -> int:
	grid = world.grid
	grid_h, grid_w = grid.shape
	sx = me.x + DX[dir]
	sy = me.y + DY[dir]

	if not (0 <= sx < grid_w and 0 <= sy < grid_h) or grid[sy, sx] != 0:
		return 0

	seen = {(sx, sy)}
	frontier = deque([(sx, sy)])

	while frontier:
		x, y = frontier.popleft()
		for d in Direction:
			nx, ny = x + DX[d], y + DY[d]
			if (
				0 <= nx < grid_w
				and 0 <= ny < grid_h
				and grid[ny, nx] == 0
				and (nx, ny) not in seen
			):
				seen.add((nx, ny))
				frontier.append((nx, ny))
	return len(seen)


class WallHuggerBot(Controller):
	"""
	Level 2
	"""

	def choose_move(
		self, world: World, bot_id: int, rng: random.Random
	) -> Direction:
		me = _find(world, bot_id)
		legal = self.legal_moves(world, bot_id)

		if not legal:
			return me.direction

		scored = [(_open_distance(world, me, d), d) for d in legal]
		best = max(score for score, _ in scored)

		return rng.choice([d for score, d in scored if score == best])

	def name(self):
		return "WallHugger"


class BlockerBot(Controller):
	"""
	Level 3
	"""

	def choose_move(
		self, world: World, bot_id: int, rng: random.Random
	) -> Direction:

		PROJECT_STEPS = 3
		me = _find(world, bot_id)
		opp = next(p for p in world.players if p.id != me.id)
		legal = self.legal_moves(world, bot_id)
		targets = []
		x, y = opp.x, opp.y

		if not legal:
			return me.direction

		for _ in range(PROJECT_STEPS):
			x += DX[opp.direction]
			y += DY[opp.direction]
			targets.append((x, y))

			def interception_cost(dir: Direction) -> int:
				nx, ny = me.x + DX[dir], me.y + DY[dir]
				return min(abs(nx - tx) + abs(ny - ty) for tx, ty in targets)

		scored = [(interception_cost(d), d) for d in legal]
		best = min(score for score, _ in scored)
		return rng.choice([d for score, d in scored if score == best])

	def name(self):
		return "BlockerBot"
