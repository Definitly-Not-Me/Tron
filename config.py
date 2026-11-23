from pathlib import Path

import pygame

## Game visual
FPS = 30
ARENA_DIM = 64, 40
CELL = 30
MARGIN = 40
TRAIL_MARGIN = 1
ARENA_BORDER_WIDTH = 4
ARENA_BORDER_RADIUS = 8


# Engine settings
BOOST_SPEED_GAIN = 2
STARTING_BOOST: int = 8
MIN_GRID = 14


# Features
# TODO: find better way to do than update manually
N_BINS_DIST = 4
N_BINS_COORD = 3
N_FEATURES = 31


# model settings
N_ACTIONS = 6
ACTION_MAP = {
	0: (-1, False),
	1: (-1, True),
	2: (0, False),
	3: (0, True),
	4: (1, False),
	5: (1, True),
}

# training settings
ALPHA = 0.05
GAMMA = 0.95
STARTING_EPSILON = 1.0
FINAL_EPSILON = 0.05
EPSILON_DECAY_STEPS = 12_000

# Pygame colors
BG = (10, 10, 30)
P1_COLOR = (1, 180, 255)
P2_COLOR = (255, 60, 60)
GRID_COLOR = (30, 30, 30)
WHITE = (225, 225, 225)
GREEN = (0, 255, 0)
GOLD = (255, 215, 0)
SM_BLACK = (20, 20, 35)
BLACK_OVERLAY = (0, 0, 0, 80)
PLAYERS_COLORS = [P1_COLOR, P2_COLOR]
ARENA_BORDER_COLOR = (10, 14, 20)
ARENA_BG = (0, 0, 1)

# Misc
WORKSPACE_DIR = Path(__file__).parent
assert WORKSPACE_DIR.exists()
ASSETS = WORKSPACE_DIR / "assets"
assert ASSETS.exists()
FONT_MENU = ASSETS / "fonts" / "Orbitron" / "Orbitron-VariableFont_wght.ttf"

FONT_TITLE = ASSETS / "fonts" / "Press_Start_2P" / "PressStart2P-Regular.ttf"

FONT_HUD = ASSETS / "fonts" / "Orbitron" / "Orbitron-VariableFont_wght.ttf"

FONT_HUD_BOLD = ASSETS / "fonts" / "Orbitron" / "static" / "Orbitron-Bold.ttf"
MENU_BG = ASSETS / "menu.png"
LOOP_MUSIC = ASSETS / "music" / "u_98o9hlkn7r-cyberpunk-music.ogg"
CRASH_SOUND = ASSETS / "music" / "freesound_community-medium-explosion.ogg"
