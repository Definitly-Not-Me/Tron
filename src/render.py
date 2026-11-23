import os

import pygame

from config import (
	ARENA_BG,
	ARENA_BORDER_COLOR,
	ARENA_BORDER_RADIUS,
	ARENA_BORDER_WIDTH,
	BG,
	BLACK_OVERLAY,
	CELL,
	FONT_HUD,
	FONT_HUD_BOLD,
	FONT_MENU,
	GREEN,
	GRID_COLOR,
	MARGIN,
	P1_COLOR,
	P2_COLOR,
	PLAYERS_COLORS,
	SM_BLACK,
	TRAIL_MARGIN,
	WHITE,
)
from src.engine import World


class Renderer:
	def __init__(self, world: World, render_mode: str = "human") -> None:
		height, width = world.grid.shape
		self.arena_w, self.arena_h = width * CELL, height * CELL
		self.off_x = MARGIN
		self.off_y = MARGIN

		if render_mode != "human":
			os.environ["SDL_VIDEODRIVERS"] = "fake"
		else:
			self.screen = pygame.display.set_mode(
				(self.arena_w + 2 * MARGIN, self.arena_h + 2 * MARGIN),
				pygame.RESIZABLE,
			)

			pygame.font.init()
			self.font_hud = pygame.font.Font(FONT_HUD, 38)
			self.font_hud_bold = pygame.font.Font(FONT_HUD_BOLD, 48)
			self.font_reg = pygame.font.Font(FONT_MENU, 38)

			pygame.display.set_caption("Tron - Light Cycles")

	def add_text(
		self,
		msg: str,
		pos: tuple[int, int],
		font: pygame.font.Font | None = None,
		color: tuple[int, int, int] | pygame.Color = WHITE,
	):
		f = font or self.font_reg
		text_surface = f.render(msg, False, color)
		self.screen.blit(text_surface, pos)

	def add_text_center(
		self,
		msg: str,
		y: int,
		font: pygame.font.Font | None = None,
		color: tuple[int, int, int] | pygame.Color = WHITE,
	):
		f = font or self.font_reg
		text_surface = f.render(msg, False, color)
		rect = text_surface.get_rect(center=(self.screen.get_width() // 2, y))
		self.screen.blit(text_surface, rect)

	def handle_resize(self, w: int, h: int):
		self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
		self._center_arena()

	def _rect(self, x: int, y: int, margin: int = 0) -> pygame.Rect:
		return pygame.Rect(
			self.off_x + x * CELL + margin,
			self.off_y + y * CELL + margin,
			CELL - 2 * margin,
			CELL - 2 * margin,
		)

	def _center_arena(self):
		screen_w, screen_h = self.screen.get_size()
		self.off_x = (screen_w - self.arena_w) // 2
		self.off_y = (screen_h - self.arena_h) // 2

	def draw(self, world: World):
		if self.screen is None:
			return
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
		if self.screen is None:
			return

		self.draw(world)
		overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
		overlay.fill(BLACK_OVERLAY)
		self.screen.blit(overlay, (0, 0))

		text = "DRAW" if winner_id is None else f"P{winner_id} WINS!"
		color = GREEN if winner_id is None else PLAYERS_COLORS[winner_id - 1]

		text_surface = self.font_hud_bold.render(text, True, color)
		text_rect = text_surface.get_rect(center=self.screen.get_rect().center)

		# print("Game over!")

		self.screen.blit(text_surface, text_rect)

	def clear(self):
		self.screen.fill(SM_BLACK)

	def quit(self):
		pygame.quit()
