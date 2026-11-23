# main.py
import math
from pathlib import Path

import numpy as np
import pygame

from config import ARENA_DIM, FPS, LOOP_MUSIC
from src.bots import BlockerBot, NaiveBot, RandomBot, WallHuggerBot
from src.engine import Action, Alive, Dead, Direction, reset, step
from src.model import Agent, AgentController
from src.render import MainMenu, Renderer

WEIGHTS = "/home/encrid/Projets/tron-rl/logs/TRAINING-21-11-2026_21:42:22/WEIGHTS/TIER-1/weights_17499.npy"
W, H = ARENA_DIM
BOTS = [NaiveBot, BlockerBot, WallHuggerBot, RandomBot]
SEED = 3


def load_agent_controller(path: str | Path) -> AgentController:
	a = Agent()
	a.load(Path(path))

	return AgentController(a)


def human_action(keys, current_dir: Direction) -> tuple[Direction, bool]:
	d = current_dir
	if keys[pygame.K_UP] or keys[pygame.K_w]:
		d = Direction.UP
	elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
		d = Direction.DOWN
	elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
		d = Direction.LEFT
	elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
		d = Direction.RIGHT
	return d, keys[pygame.K_SPACE]


def main():
	pygame.init()

	world = reset((W, H), SEED)
	rng = np.random.default_rng()
	rng.bit_generator.state = world.rng_state

	renderer = Renderer(world)
	clock = pygame.time.Clock()

	pygame.mixer.init()
	pygame.mixer.music.load(LOOP_MUSIC)
	pygame.mixer.music.play(-1)

	state = "menu"
	countdown = 0.0
	agent = load_agent_controller(WEIGHTS)
	menu = MainMenu(renderer)
	scores = (0, 0)
	game_over_t = 1.0

	while True:
		dt = clock.tick(FPS) / 1000

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				return

			# BUG: infinite resize loop on wm like niri
			if event.type == pygame.VIDEORESIZE:
				renderer._center_arena()
				if state == "menu":
					menu._layout()
			# 	renderer.handle_resize(event.w, event.h)

			if state == "menu" and menu.handle_input(event):
				world = reset((W, H))
				countdown = 3.0
				state = "countdown"

		match state:
			case "menu":
				menu.render()
			case "countdown":
				countdown -= dt
				if countdown <= 0:
					state = "game"

				renderer.draw(world)
				renderer.draw_countdown(math.ceil(countdown))
			case "game":
				keys = pygame.key.get_pressed()
				player_dir, player_boost = human_action(
					keys, world.players[0].direction
				)
				bot_dir, bot_boost = agent.choose_move(
					world=world, bot_id=2, rng=rng
				)
				act = Action(
					p1_dir=player_dir,
					p2_dir=bot_dir,
					p1_boost=player_boost,
					p2_boost=bot_boost,
				)
				result = step(world, act)

				match result:
					case Alive(w):
						world = w
						renderer.draw(world)
						renderer.draw_hud(
							scores,
							tuple(p.boost_remaining for p in world.players),
						)
					case Dead(world=w, autopsy=_, winner_id=winner):
						world = w
						scores = tuple(
							s + (1 if winner == i else 0)
							for i, s in enumerate(scores, start=1)
						)

						last_winner = winner
						state = "game_over"

			case "game_over":
				game_over_t -= dt
				if game_over_t <= 0:
					world = reset((W, H), SEED)
					countdown = 3.0
					state = "countdown"
					game_over_t = 1.0
				renderer.draw(world)
				renderer.draw_game_over(world, last_winner)

		pygame.display.flip()


if __name__ == "__main__":
	main()
