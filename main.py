# main.py
import random
import sys
from random import choice

import pygame

from bots import BlockerBot, NaiveBot, RandomBot, WallHuggerBot
from engine import Action, Dead, Direction, reset, step
from render import Renderer

FPS = 25
W, H = 64, 40
BOTS = [NaiveBot, BlockerBot, WallHuggerBot, RandomBot]


def new_game(seed: int | None = None):
	world = reset((W, H), seed=42)
	bot = choice(BOTS)()

	print("New game started with bot ", bot.name())

	return world, bot


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

	clock = pygame.time.Clock()
	world, bot = new_game(seed=42)
	render = Renderer(world)
	over = False
	rng = random.Random(42)

	while True:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				render.quit()
				sys.exit()

		if over:
			if pygame.key.get_pressed()[pygame.K_r]:
				world, bot = new_game(42)
				over = False
			clock.tick(FPS)
			continue

		keys = pygame.key.get_pressed()
		p1_dir, p1_boost = human_action(keys, world.players[0].direction)
		# p2_dir = random.choice(
		# 	[Direction.RIGHT, Direction.LEFT, Direction.DOWN]
		# )  # placeholder
		p2_dir = bot.choose_move(world, 2, rng)
		p2_boost = False

		action = Action(
			p1_dir=p1_dir, p2_dir=p2_dir, p1_boost=p1_boost, p2_boost=p2_boost
		)
		result = step(world, action)

		if isinstance(result, Dead):
			over = True
			render.draw_game_over(result.world, result.winner_id)
			clock.tick(FPS)
			continue

		world = result.world
		render.draw(world)
		clock.tick(FPS)


if __name__ == "__main__":
	main()
