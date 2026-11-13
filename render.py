import pygame

from engine import World

CELL = 30
MARGIN = 40
TRAIL_MARGIN = 1
BG = (10, 10, 30)
P1_COLOR = (1, 180, 255)
P2_COLOR = (255, 60, 60)
GRID_COLOR = (30, 30, 30)
WHITE = (225, 225, 225)
GREEN = (0, 255, 0)
GOLD = (255, 215, 0)
PLAYERS_COLORS = [P1_COLOR, P2_COLOR]
ARENA_BORDER_COLOR = (10, 14, 20)
ARENA_BORDER_WIDTH = 4
ARENA_BORDER_RADIUS = 8
ARENA_BG = (0, 0, 1)


class Renderer:
	def __init__(self, world: World) -> None:
		height, width = world.grid.shape
		self.arena_w, self.arena_h = width * CELL, height * CELL
		self.off_x = MARGIN
		self.off_y = MARGIN
		self.screen = pygame.display.set_mode(
			(self.arena_w + 2 * MARGIN, self.arena_h + 2 * MARGIN)
		)

		self._center_arena()
		pygame.display.set_caption("Tron")

	def _rect(self, x: int, y: int, margin: int = 0) -> pygame.Rect:
		return pygame.Rect(
			self.off_x + x * CELL + margin,
			self.off_y + y * CELL + margin,
			CELL - 2 * margin,
			CELL - 2 * margin,
		)

	def _glow(self, x: int, y: int, layers):
		for spread, color in layers:
			r = self._rect(x, y, margin=2 - spread)
			s = pygame.Surface(r.size, pygame.SRCALPHA)
			pygame.draw.rect(s, color, s.get_rect(), border_radius=4)
			self.screen.blit(s, r.topleft, special_flags=pygame.BLEND_RGB_ADD)

	def _center_arena(self):
		screen_w, screen_h = self.screen.get_size()
		self.off_x = (screen_w - self.arena_w) // 2
		self.off_y = (screen_h - self.arena_h) // 2

	def draw(self, world: World):
		h, w = world.grid.shape
		self._center_arena()
		self.screen.fill(BG)
		arena = pygame.Rect(self.off_x, self.off_y, self.arena_w, self.arena_h)

		pygame.draw.rect(
			self.screen, ARENA_BG, arena, border_radius=ARENA_BORDER_RADIUS
		)

		for x in range(w + 1):
			pygame.draw.line(
				self.screen,
				GRID_COLOR,
				(self.off_x + x * CELL, self.off_y),
				(self.off_x + x * CELL, self.off_y + h * CELL),
			)

		for y in range(h + 1):
			pygame.draw.line(
				self.screen,
				GRID_COLOR,
				(self.off_x, self.off_y + y * CELL),
				(self.off_x + w * CELL, self.off_y + y * CELL),
			)

		for y in range(h):
			for x in range(w):
				v = world.grid[y, x]

				if v == world.players[0].id:
					pygame.draw.rect(
						self.screen, P1_COLOR, self._rect(x, y, TRAIL_MARGIN)
					)
				elif v == world.players[1].id:
					pygame.draw.rect(
						self.screen, P2_COLOR, self._rect(x, y, TRAIL_MARGIN)
					)

		for i, p in enumerate(world.players):
			color = PLAYERS_COLORS[i]
			pygame.draw.rect(
				self.screen, color, self._rect(p.x, p.y), border_radius=3
			)
			pygame.draw.rect(
				self.screen,
				WHITE,
				self._rect(p.x, p.y, CELL // 3),
				border_radius=1,
			)

		pygame.draw.rect(
			self.screen,
			ARENA_BORDER_COLOR,
			arena,
			width=ARENA_BORDER_WIDTH,
			border_radius=ARENA_BORDER_RADIUS,
		)

		pygame.display.flip()

	def draw_game_over(self, world: World, winner_id: int | None):
		self.draw(world)
		overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
		overlay.fill((0, 0, 0, 128))
		self.screen.blit(overlay, (0, 0))

		text = "DRAW" if winner_id is None else f"P{winner_id} WINS!"
		color = (
			GREEN
			if winner_id is None
			else PLAYERS_COLORS[winner_id - 1]
		)
		font = pygame.font.Font(None, CELL * 4)

		text_surface = font.render(text, True, color)
		text_rect = text_surface.get_rect(center=self.screen.get_rect().center)

		print("Game over!")

		self.screen.blit(text_surface, text_rect)
		pygame.display.flip()

	def quit(self):
		pygame.quit()
