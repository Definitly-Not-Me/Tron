#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENTRY = ROOT / "main.py"
ASSETS = ROOT / "assets"
WEIGHTS = ASSETS / "weights.npy"
BUILD = ROOT / "build"
DIST = ROOT / "dist"
APP_NAME = "tron"
SUPPORTED = {"Windows", "Linux", "Darwin"}


def _find_linux_sdl3() -> list[str]:
	"""Find SDL3 libs, copy them renamed to their SONAME, return --add-binary args."""
	import shutil
	import tempfile

	seen: set[str] = set()
	tmpdir = Path(tempfile.mkdtemp())
	results: list[str] = []

	for f in sorted(Path("/nix/store").glob("*/lib/libSDL3*.so.*")):
		m = re.match(r"(libSDL3[a-z_]*\.so\.0)\.\d+\.\d+", f.name)
		if m and m.group(1) not in seen:
			seen.add(m.group(1))
			dest = tmpdir / m.group(1)
			shutil.copy2(f, dest)
			results.append(f"{dest}{os.pathsep}.")

	return results


def main() -> None:
	system = platform.system()
	if system not in SUPPORTED:
		raise SystemExit(f"Unsupported platform: {system}")

	if not ENTRY.is_file():
		raise SystemExit(f"Missing entry point: {ENTRY}")
	if not ASSETS.is_dir():
		raise SystemExit(f"Missing assets directory: {ASSETS}")
	if not WEIGHTS.is_file():
		raise SystemExit(f"Missing weights: {WEIGHTS}")

	if importlib.util.find_spec("PyInstaller") is None:
		raise SystemExit(
			"PyInstaller is not installed. Run: python -m pip install pyinstaller"
		)
	shutil.rmtree(DIST, ignore_errors=True)
	BUILD.mkdir(parents=True, exist_ok=True)
	extra_binaries: list[str] = []
	if system == "Linux":
		sdl_libs = _find_linux_sdl3()
		if not sdl_libs:
			raise SystemExit(
				"Could not locate SDL3 shared libraries in /nix/store"
			)
		for lib in sdl_libs:
			extra_binaries.extend(["--add-binary", lib])

	cmd = [
		sys.executable,
		"-m",
		"PyInstaller",
		"--noconfirm",
		"--clean",
		"--onedir",
		"--windowed",
		"--name",
		APP_NAME,
		"--specpath",
		str(BUILD),
		"--workpath",
		str(BUILD / "work"),
		"--distpath",
		str(DIST),
		*extra_binaries,
		"--add-data",
		f"{ASSETS}{os.pathsep}assets",
		str(ENTRY),
	]
	subprocess.run(cmd, cwd=ROOT, check=True)

	output = {
		"Darwin": DIST / f"{APP_NAME}.app",
		"Windows": DIST / APP_NAME / f"{APP_NAME}.exe",
		"Linux": DIST / APP_NAME / APP_NAME,
	}[system]
	if not output.exists():
		raise SystemExit(f"Build finished but output is missing: {output}")

	print(f"Built: {output}")


if __name__ == "__main__":
	main()
