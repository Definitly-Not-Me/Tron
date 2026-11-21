# Tron

An implementation of the Tron Light cycles game with Reinforcement Learning using pygame.

The game is mainly designed for PvE just like the [original](https://pexy.io/game/tron/).
The RL agent is trained using vanilla Q-learning and LFA (Linear Function Approximation).
Its performance isn't AlphaZero worthy but it occasionally manages to beat me.

>[!NOTE]
> This is a Work in Progress

## Getting started

The main game only relies on the Pygame and Numpy libraries and a python 3.14 interpreter.

If you have those you should be fine.

To play the game just run:

```
  python3 main.py
```

or just `python main.py` for some.

---
As of now, the Reinforcement Learning agent isn't ready yet, so
each match is against a randomly selected bot from a bot tier list namely:

* Random (tier 0)
* Naive (tier 1)
* Blocker (tier 2)
* Min Voronoi (tier 3)
