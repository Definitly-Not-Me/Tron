# main.py
import random
import sys

import pygame

from engine import Action, Dead, Direction, reset, step
from render import Renderer

FPS = 20
W, H = 44, 44


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
	world = reset((H, W), seed=42)
	render = Renderer(world)
	over = False

	while True:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				render.quit()
				sys.exit()

		if over:
			if pygame.key.get_pressed()[pygame.K_r]:
				world = reset((H, W), seed=None)
				over = False
			clock.tick(FPS)
			continue

		keys = pygame.key.get_pressed()
		p1_dir, p1_boost = human_action(keys, world.players[0].direction)
		p2_dir = random.choice(
			[Direction.RIGHT, Direction.LEFT, Direction.DOWN]
		)  # placeholder
		p2_boost = False

		action = Action(
			p1_dir=p1_dir, p2_dir=p2_dir, p1_boost=p1_boost, p2_boost=p2_boost
		)
		result = step(world, action)

		if isinstance(result, Dead):
			over = True
			render.draw_game_over(result.world, result.winner)
			clock.tick(FPS)
			continue

		world = result.world
		render.draw(world)
		clock.tick(FPS)


if __name__ == "__main__":
	main()
