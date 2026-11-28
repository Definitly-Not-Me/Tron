import os
from typing import Any

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
	FONT_TITLE,
	GOLD,
	GREEN,
	GRID_COLOR,
	MARGIN,
	MAX_BOOST,
	MENU_BG,
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
		self.screen: Any | pygame.Surface = None
		self.arena_w, self.arena_h = width * CELL, height * CELL
		self.off_x = MARGIN
		self.off_y = MARGIN
		self.cell = CELL

		if render_mode != "human":
			os.environ["SDL_VIDEODRIVERS"] = "fake"
		else:
			info = pygame.display.Info()
			screen_w, screen_h = info.current_w, info.current_h
			if screen_w > 0 and screen_h > 0:
				scale = min(screen_w * 0.8 / width, screen_h * 0.8 / height)
				window_w = max(1, int(width * scale))
				window_h = max(1, int(height * scale))
			else:
				window_w, window_h = width * CELL, height * CELL

			self.screen = pygame.display.set_mode(
				(window_w, window_h),
				pygame.RESIZABLE,
			)

			pygame.font.init()
			self.font_hud = pygame.font.Font(FONT_HUD, 38)
			self.font_hud_bold = pygame.font.Font(FONT_HUD_BOLD, 48)
			self.font_reg = pygame.font.Font(FONT_MENU, 38)

			self._layout((width, height))
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

	def _layout(
		self,
		arena_dim: tuple[int, int],
		screen_dim: tuple[int, int] | None = None,
	):
		g_w, g_h = arena_dim
		s_w, s_h = screen_dim or self.screen.get_size()
		self.cell = max(1, min(s_w // g_w, s_h // g_h))
		self.arena_w = self.cell * g_w
		self.arena_h = self.cell * g_h
		self._center_arena((s_w, s_h))

	def handle_resize(
		self, arena_dim: tuple[int, int], screen_dim: tuple[int, int]
	):
		self._layout(arena_dim, screen_dim)

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

	def _rect(self, x: int, y: int, margin: int = 0) -> pygame.Rect:
		return pygame.Rect(
			self.off_x + x * self.cell + margin,
			self.off_y + y * self.cell + margin,
			self.cell - 2 * margin,
			self.cell - 2 * margin,
		)

	def _center_arena(self, screen_dim: tuple[int, int] | None = None):
		screen_w, screen_h = screen_dim or self.screen.get_size()
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
				(self.off_x + x * self.cell, self.off_y),
				(self.off_x + x * self.cell, self.off_y + h * self.cell),
			)

		for y in range(h + 1):
			pygame.draw.line(
				self.screen,
				GRID_COLOR,
				(self.off_x, self.off_y + y * self.cell),
				(self.off_x + w * self.cell, self.off_y + y * self.cell),
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
				self._rect(p.x, p.y, self.cell // 3),
				border_radius=1,
			)

		pygame.draw.rect(
			self.screen,
			ARENA_BORDER_COLOR,
			arena,
			width=ARENA_BORDER_WIDTH,
			border_radius=ARENA_BORDER_RADIUS,
		)

	def draw_hud(self, scores: tuple[int, int], boost: tuple[int, int]):
		if self.screen is None:
			return
		screen_w = self.screen.get_width()
		y = MARGIN
		score_p1, score_p2 = scores

		parts = [
			(self.font_hud, "P1", WHITE),
			(self.font_hud_bold, str(score_p1), P1_COLOR),
			(self.font_hud, "  —  ", WHITE),
			(self.font_hud_bold, str(score_p2), P2_COLOR),
			(self.font_hud, "P2", WHITE),
		]

		surfaces = [f.render(t, True, c) for f, t, c in parts]
		total_w = sum(s.get_width() for s in surfaces)
		x = (screen_w - total_w) // 2

		for s in surfaces:
			self.screen.blit(s, (x, y))
			x += s.get_width()

		dy = y + self.font_hud_bold.get_height() + 8
		bar_w, spacing = 8, 13
		group_w = (MAX_BOOST - 1) * spacing + bar_w
		centers = (
			self.off_x + self.arena_w // 4,
			self.off_x + 3 * self.arena_w // 4,
		)
		for cx, n in zip(centers, boost, strict=True):
			left = cx - group_w // 2
			for i in range(MAX_BOOST):
				r = pygame.Rect(left + i * spacing, dy, bar_w, 8)
				pygame.draw.rect(
					self.screen,
					GREEN if i < n else SM_BLACK,
					r,
					border_radius=2,
				)

	def draw_countdown(self, n: int = 3):
		overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
		overlay.fill(BLACK_OVERLAY)

		msg = "GO!" if n <= 0 else str(n)
		color = GREEN if n <= 0 else WHITE
		txt_srf = self.font_hud_bold.render(msg, True, color)

		self.screen.blit(overlay, (0, 0))
		self.screen.blit(
			txt_srf, txt_srf.get_rect(center=self.screen.get_rect().center)
		)

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


class MainMenu:
	def __init__(self, renderer: Renderer) -> None:
		self.renderer = renderer
		self._button = pygame.Rect(0, 0, 240, 60)
		self.hovered = False
		self.bg = pygame.image.load(MENU_BG)
		self._scaled_bg = self.bg
		self.font_title = pygame.font.Font(FONT_TITLE, 48)
		self._layout()

	def _layout(self):
		w, h = self.renderer.screen.get_size()
		self._button.center = (w // 2, h // 2)
		self._scaled_bg = self._fit_cover(self.bg, w, h)

	def _fit_cover(self, img: pygame.Surface, w: int, h: int):
		iw, ih = self.bg.get_size()
		scale = max(w / iw, h / ih)

		return pygame.transform.smoothscale(
			img, (int(iw * scale), int(ih * scale))
		)

	def handle_input(self, event):
		match event.type:
			case pygame.MOUSEMOTION:
				self.hovered = self._button.collidepoint(event.pos)
				return False
			case pygame.MOUSEBUTTONDOWN if event.button == 1:
				return self.hovered
			case pygame.KEYDOWN if event.key == pygame.K_RETURN:
				return True
			case _:
				return False

	def render(self):
		r = self.renderer

		# NOTE: tied to infinite loop bug in main()
		sw, sh = r.screen.get_size()
		iw, ih = self._scaled_bg.get_size()
		x, y = (sw - iw) // 2, (sh - ih) // 2

		r.screen.blit(self._scaled_bg, (x, y))

		r.add_text_center(
			"Arrow keys or WASD to move. SPACE to boost",
			int(0.1 * r.screen.get_height()),
		)
		btn_text = self.font_title.render(
			"PLAY", True, WHITE if self.hovered else GOLD
		)
		button = self._button.inflate(12, 6) if self.hovered else self._button
		pygame.draw.rect(
			r.screen,
			GREEN if self.hovered else SM_BLACK,
			button,
			border_radius=8,
		)
		r.screen.blit(btn_text, btn_text.get_rect(center=button.center))
