from __future__ import annotations

import argparse
import csv
from collections import deque
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from src.bots import BlockerBot, Controller, NaiveBot, RandomBot, WallHuggerBot
from src.gym_env import TronEnv
from src.model import Agent


class CurriculumManager:
	LADDER = [RandomBot, NaiveBot, WallHuggerBot, BlockerBot]

	def __init__(
		self,
		window: int = 50,
		min_win_rate: float = 0.8,
		promote_streak: int = 10,
	) -> None:
		self.level = 0
		self.window = deque(maxlen=max(1, window))
		self.streak = 0
		self.req_win_rate = min_win_rate
		self.req_streak = promote_streak

	def promote(self) -> None:
		if self.level < len(self.LADDER) - 1:
			self.level += 1
		self.window.clear()
		self.streak = 0

	def next_opponent(self) -> Controller:
		return self.LADDER[self.level]()

	def record(self, outcome: str) -> None:
		self.window.append(outcome)
		self.streak = self.streak + 1 if outcome == "win" else 0

	def should_promote(self) -> bool:
		if self.level >= len(self.LADDER) - 1:
			return False
		if len(self.window) < (self.window.maxlen or 0):
			return False

		wins = sum(1 for outcome in self.window if outcome == "win")
		return (
			wins / len(self.window) >= self.req_win_rate
			and self.streak >= self.req_streak
		)

	@property
	def win_rate(self) -> float:
		if not self.window:
			return 0.0
		return sum(1 for outcome in self.window if outcome == "win") / len(
			self.window
		)


class LiveWatcher:
	def __init__(self, agent: Agent) -> None:
		self._enabled = False

		try:
			import matplotlib.pyplot as plt  # noqa: PLC0415
		except ImportError:
			print("matplotlib is not installed; --watch disabled.", flush=True)
			return

		self._plt = plt
		plt.ion()

		self.fig, (self.ax_w, self.ax_m) = plt.subplots(1, 2, figsize=(13, 5))
		self.im = self.ax_w.imshow(
			agent.W,
			aspect="auto",
			interpolation="nearest",
			cmap="viridis",
		)
		self.ax_w.set_title("Q weights")
		self.ax_w.set_xlabel("feature")
		self.ax_w.set_ylabel("action")
		self.fig.colorbar(self.im, ax=self.ax_w)

		self.ax_eps = self.ax_m.twinx()
		(self.reward_line,) = self.ax_m.plot(
			[], [], color="tab:blue", label="reward"
		)
		(self.td_line,) = self.ax_m.plot(
			[], [], color="tab:orange", label="|TD|"
		)
		(self.eps_line,) = self.ax_eps.plot(
			[], [], color="tab:green", label="epsilon"
		)
		(self.eval_line,) = self.ax_eps.plot(
			[], [], color="tab:red", label="eval win", linewidth=1
		)

		self.ax_m.set_title("Training metrics")
		self.ax_m.set_xlabel("episode")
		self.ax_m.set_ylabel("reward / |TD|")
		self.ax_eps.set_ylabel("epsilon / win rate")
		self.ax_eps.set_ylim(0.0, 1.05)

		lines = [self.reward_line, self.td_line, self.eps_line, self.eval_line]
		self.ax_m.legend(
			lines,
			[line.get_label() for line in lines],
			loc="upper left",
			fontsize=8,
		)

		self.episodes: list[int] = []
		self.rewards: list[float] = []
		self.tds: list[float] = []
		self.epsilons: list[float] = []
		self.eval_wins: list[float] = []

		self.fig.tight_layout()
		self.fig.show()
		plt.pause(0.001)
		self._enabled = True

	def update(  # noqa: PLR0913, PLR0917
		self,
		episode: int,
		agent: Agent,
		reward: float,
		mean_td: float,
		epsilon: float,
		eval_win_rate: float,
	) -> None:
		if not self._enabled:
			return

		self.episodes.append(episode)
		self.rewards.append(reward)
		self.tds.append(mean_td)
		self.epsilons.append(epsilon)
		self.eval_wins.append(eval_win_rate)

		self.im.set_data(agent.W)
		low, high = float(agent.W.min()), float(agent.W.max())
		self.im.set_clim(low, high + 1e-6)

		self.reward_line.set_data(self.episodes, self.rewards)
		self.td_line.set_data(self.episodes, self.tds)
		self.eps_line.set_data(self.episodes, self.epsilons)
		self.eval_line.set_data(self.episodes, self.eval_wins)

		self.ax_m.relim()
		self.ax_m.autoscale_view()
		self.ax_eps.set_ylim(0.0, 1.05)

		self.fig.canvas.draw_idle()
		self._plt.pause(0.001)

	def close(self) -> None:
		if not self._enabled:
			return
		self._plt.ioff()
		self._plt.close(self.fig)


def classify_outcome(
	info: dict[str, Any], terminated: bool, truncated: bool
) -> str:

	if terminated:
		return "win" if info["agent_survived"] else "loss"

	if truncated:
		return "timeout"

	# TODO: something about this
	return "draw"


def evaluate(
	env: TronEnv,
	agent: Agent,
	curriculum: CurriculumManager,
	episodes: int,
	seed: int,
) -> float:
	if episodes <= 0:
		return 0.0

	env.opp = curriculum.next_opponent()
	wins = 0

	for i in range(episodes):
		obs, _ = env.reset(seed=seed + i)
		while True:
			q = agent.q_values(obs)
			best = q.max()
			action = int(env.np_random.choice(np.flatnonzero(q == best)))
			obs, _, terminated, truncated, info = env.step(action)
			if terminated or truncated:
				break

		outcome = classify_outcome(info, terminated, truncated)
		curriculum.record(outcome)
		wins += outcome == "win"

	return wins / episodes


def get_env_ready(args: argparse.Namespace) -> dict[str, Any]:
	base = Path(args.out)
	timetag = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
	output_path = Path(base / f"TRAINING-{timetag}")
	log_path = Path(output_path / "CSV" / "training.csv")
	weights_path = Path(args.resume_from or "")
	agent = Agent()
	rng = np.random.default_rng(args.seed)
	watcher = LiveWatcher(agent) if args.watch else None
	render_mode = "human" if args.render else None
	curriculum = CurriculumManager(
		window=args.window,
		min_win_rate=args.min_win_rate,
		promote_streak=args.promote_streak,
	)
	env = TronEnv(
		opp=curriculum.next_opponent(),
		max_ticks=args.max_ticks,
		render_mode=render_mode,
		starting_size=tuple(args.starting_size),
	)

	output_path.mkdir(parents=True, exist_ok=True)
	log_path.parent.mkdir(parents=True, exist_ok=True)

	if args.resume_from:
		assert weights_path.exists(), (
			f"The path provided to --resume-from doesn't exist.\nPath: {weights_path}"
		)
		print(">> Resuming from :", weights_path)
		agent.load(weights_path)

	return {
		"agent": agent,
		"rng": rng,
		"watcher": watcher or None,
		"env": env,
		"logfile": log_path,
		"outpath": output_path,
		"curriculum_manager": curriculum,
	}


def train(args: argparse.Namespace) -> Agent:
	settings = get_env_ready(args)

	output_path, log_path = settings["outpath"], settings["logfile"]
	curriculum = settings["curriculum_manager"]
	env = settings["env"]
	agent = settings["agent"]
	rng = settings["rng"]
	watcher = settings["watcher"]
	started = perf_counter()

	with open(log_path, "w", newline="") as logfile:
		writer = csv.writer(logfile)
		writer.writerow(
			[
				"episode",
				"tier",
				"opponent",
				"steps",
				"train_reward",
				"outcome",
				"epsilon",
				"mean_abs_td",
				"weight_max",
				"eval_win_rate",
			]
		)

		try:
			for ep in range(args.episodes):
				tier = curriculum.level
				env.opp = curriculum.next_opponent()
				opponent = env.opp.name

				obs, _ = env.reset(seed=args.seed + ep)
				ep_reward, td_errors, steps = 0.0, [], 0

				while True:
					action = agent.select_action(obs, rng)
					next_obs, reward, terminated, truncated, info = env.step(
						action
					)

					delta = agent.update(
						obs, action, reward, next_obs, terminated
					)
					td_errors.append(abs(delta))

					ep_reward += reward
					obs = next_obs
					steps += 1

					if terminated or truncated:
						break

				outcome = classify_outcome(info, terminated, truncated)
				mean_td = float(np.mean(td_errors)) if td_errors else 0.0
				eval_win_rate: float | None = None

				if args.eval_every > 0 and (ep + 1) % args.eval_every == 0:
					eval_win_rate = evaluate(
						env,
						agent,
						curriculum,
						args.eval_episodes,
						args.seed + ep,
					)

					if curriculum.should_promote():
						curriculum.promote()
						current = curriculum.LADDER[curriculum.level].name
						print(
							f"ep {ep + 1}: PROMOTED {opponent} -> {current} "
							f"(eval {eval_win_rate:.0%})",
							flush=True,
						)
						# Reset epsilon after each promotion
						agent._steps = 0

				writer.writerow(
					[
						ep + 1,
						tier,
						opponent,
						steps,
						round(ep_reward, 4),
						outcome,
						round(agent.epsilon, 4),
						round(mean_td, 4),
						round(float(np.abs(agent.W).max()), 4),
						""
						if eval_win_rate is None
						else round(eval_win_rate, 4),
					]
				)
				logfile.flush()

				if (
					args.checkpoint_every > 0
					and (ep + 1) % args.checkpoint_every == 0
				):
					agent.save(
						output_path
						/ "WEIGHTS"
						/ f"TIER-{tier}"
						/ f"weights_{ep}"
					)
					eval_text = (
						"n/a"
						if eval_win_rate is None
						else f"{eval_win_rate:.0%}"
					)
					elapsed = perf_counter() - started
					print(
						f"ep {ep + 1:>5} | tier {tier} ({opponent}) | "
						f"steps {steps:>3} | reward {ep_reward:>6.2f} | "
						f"{outcome:<7} | eps {agent.epsilon:.3f} | "
						f"|TD| {mean_td:.4f} | eval {eval_text} | "
						f"{elapsed:.1f}s",
						flush=True,
					)

				if (
					watcher is not None
					and args.plot_every > 0
					and (ep + 1) % args.plot_every == 0
				):
					watcher.update(
						ep + 1,
						agent,
						ep_reward,
						mean_td,
						agent.epsilon,
						float("nan")
						if eval_win_rate is None
						else eval_win_rate,
					)

			print(
				f">> Training ended after {perf_counter() - started:.1f}s",
				flush=True,
			)
		finally:
			agent.save(output_path / "WEIGHTS" / "weights_final")
			env.close()
			if watcher is not None:
				watcher.close()

	return agent


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Train the Tron Q-learning agent."
	)
	parser.add_argument("--episodes", type=int, default=1000)
	parser.add_argument("--seed", type=int, default=3)
	parser.add_argument("--out", default="logs/")
	parser.add_argument("--checkpoint-every", type=int, default=50)
	parser.add_argument("--eval-every", type=int, default=100)
	parser.add_argument("--eval-episodes", type=int, default=10)
	parser.add_argument("--max-ticks", type=int, default=1000)
	parser.add_argument("--window", type=int, default=50)
	parser.add_argument("--min-win-rate", type=float, default=0.8)
	parser.add_argument("--promote-streak", type=int, default=10)
	parser.add_argument(
		"--starting-size",
		type=int,
		nargs=2,
		default=[30, 30],
		metavar=("W", "H"),
	)
	parser.add_argument("--render", action="store_true")
	parser.add_argument("--watch", action="store_true")
	parser.add_argument("--plot-every", type=int, default=10)
	parser.add_argument("--resume-from", type=Path)

	return parser.parse_args()


if __name__ == "__main__":
	train(parse_args())
