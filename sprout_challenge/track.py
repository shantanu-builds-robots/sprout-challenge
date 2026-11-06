"""Tracks: a straight course along +x, built from sections, as a MuJoCo heightfield.

A track is a list of sections laid end to end, each starting and ending at ground level (height 0), on a strip
`width` metres wide centred on y = 0. The robot starts at x = START_X and finishes when its torso crosses
`finish_x` (FINISH_MARGIN before the end of the last section).

Section types and their parameters (metres unless noted):

  flat      length
  rough     length, height (bump height), cell (bump size), seed
  stairs    steps, rise, tread, landing             up `steps` steps, a landing, down again
  hurdles   heights (list), spacing, thickness, run_in
  ramp      rise, run, top                          up over `run`, a flat top, down over `run`
  bridge    length, width, height, ramp             a raised walkway with ramps at both ends
  waves     length, amplitude, wavelength
  gap       length, depth                           a trench across the track
  barrier   height, depth, run_in, run_out, side, bypass, bonus
                                                    a solid block across one side of the track (and the
                                                    centreline), leaving a `bypass` metres wide lane open on
                                                    the other side: jump it for a `bonus` (seconds off the
                                                    finish time), or walk around it for none

Tracks are YAML files (see sprout_challenge/tracks/practice.yaml):

  name: practice
  width: 6
  sections:
    - {type: flat, length: 5}
    - {type: stairs, steps: 4, rise: 0.08, tread: 0.5, landing: 2}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

RES = 0.05              # m per heightfield cell
START_X = 1.0           # m: where the robot starts
FINISH_MARGIN = 1.0     # m: the finish line is this far before the end of the track
BASE = 0.5              # m: depth of the heightfield's base below ground level (room for gaps)

TRACKS_DIR = Path(__file__).resolve().parent / "tracks"


def _flat(x, y, p):
    return np.zeros_like(x)


def _rough(x, y, p):
    rng = np.random.default_rng(int(p.get("seed", 0)))
    cell, hmax, L = float(p.get("cell", 0.25)), float(p.get("height", 0.05)), float(p["length"])
    grid = rng.uniform(0, hmax, (int(np.ceil(6 * 10 / cell)) + 2, int(np.ceil(L / cell)) + 2))
    i = np.clip((x / cell).astype(int), 0, grid.shape[1] - 1)
    j = np.clip(((y + 30) / cell).astype(int), 0, grid.shape[0] - 1)
    fade = np.clip(np.minimum(x, L - x) / 0.5, 0, 1)
    return grid[j, i] * fade


def _stairs(x, y, p):
    n, rise, tread, landing = int(p["steps"]), float(p["rise"]), float(p["tread"]), float(p["landing"])
    up = np.clip(np.floor(x / tread) + 1, 0, n)
    end = 2 * n * tread + landing
    down = np.clip(np.floor((end - x) / tread) + 1, 0, n)
    return rise * np.minimum(up, down) * (x < end)


def _hurdles(x, y, p):
    hs, spacing = [float(h) for h in p["heights"]], float(p.get("spacing", 1.5))
    thick, run_in = float(p.get("thickness", 0.1)), float(p.get("run_in", 1.0))
    h = np.zeros_like(x)
    for i, hh in enumerate(hs):
        x0 = run_in + i * spacing
        h = np.where((x >= x0) & (x < x0 + thick), hh, h)
    return h


def _ramp(x, y, p):
    rise, run, top = float(p["rise"]), float(p["run"]), float(p.get("top", 1.0))
    r = np.clip(np.minimum(x / run, (2 * run + top - x) / run), 0, 1)
    return rise * r


def _bridge(x, y, p):
    L, w, hgt, ramp = float(p["length"]), float(p.get("width", 1.0)), float(p.get("height", 0.2)), float(p.get("ramp", 1.0))
    total = L + 2 * ramp
    r = np.clip(np.minimum(x / ramp, (total - x) / ramp), 0, 1)
    return np.where(np.abs(y) <= w / 2, hgt * r, 0.0)


def _waves(x, y, p):
    L, a, wl = float(p["length"]), float(p.get("amplitude", 0.12)), float(p.get("wavelength", 1.6))
    fade = np.clip(np.minimum(x, L - x) / 0.5, 0, 1)
    return a / 2 * (1 + np.sin(2 * np.pi * x / wl) * np.cos(2 * np.pi * y / (1.5 * wl))) * fade


def _gap(x, y, p):
    return np.full_like(x, -float(p.get("depth", 0.3)))


def barrier_span(p, width):
    """(y0, y1) the barrier covers: from the side named by `side` to `bypass` metres short of the other edge."""
    bypass = float(p.get("bypass", 2.5))
    if p.get("side", "left") == "left":
        return -width / 2 + bypass, width / 2
    return -width / 2, width / 2 - bypass


def _barrier(x, y, p):
    run_in, depth, h = float(p.get("run_in", 2.0)), float(p.get("depth", 0.3)), float(p.get("height", 0.4))
    y0, y1 = barrier_span(p, p["_width"])
    return np.where((x >= run_in) & (x < run_in + depth) & (y >= y0) & (y <= y1), h, 0.0)


SECTIONS = {
    "flat": (_flat, lambda p: float(p["length"])),
    "rough": (_rough, lambda p: float(p["length"])),
    "stairs": (_stairs, lambda p: 2 * int(p["steps"]) * float(p["tread"]) + float(p["landing"])),
    "hurdles": (_hurdles, lambda p: float(p.get("run_in", 1.0)) + len(p["heights"]) * float(p.get("spacing", 1.5))),
    "ramp": (_ramp, lambda p: 2 * float(p["run"]) + float(p.get("top", 1.0))),
    "bridge": (_bridge, lambda p: float(p["length"]) + 2 * float(p.get("ramp", 1.0))),
    "waves": (_waves, lambda p: float(p["length"])),
    "gap": (_gap, lambda p: float(p["length"])),
    "barrier": (_barrier, lambda p: float(p.get("run_in", 2.0)) + float(p.get("depth", 0.3)) + float(p.get("run_out", 2.0))),
}


@dataclass
class Track:
    name: str
    sections: list[dict]
    width: float = 6.0
    starts: list[float] = field(init=False)

    def __post_init__(self):
        for s in self.sections:
            if s.get("type") not in SECTIONS:
                raise ValueError(f"unknown section type {s.get('type')!r}; one of {sorted(SECTIONS)}")
        self.starts, x = [], 0.0
        for s in self.sections:
            self.starts.append(x)
            x += SECTIONS[s["type"]][1](s)
        self.length = x

    @property
    def finish_x(self) -> float:
        return self.length - FINISH_MARGIN

    def section_at(self, x: float) -> str:
        i = max([0] + [k for k, x0 in enumerate(self.starts) if x >= x0])
        return f"{i}:{self.sections[i]['type']}"

    def heights(self) -> np.ndarray:
        """(nrow, ncol) heights [m] relative to ground level; row 0 at -y, column 0 at x = 0."""
        ncol = int(round(self.length / RES)) + 1
        nrow = int(round(self.width / RES)) + 1
        xs, ys = np.linspace(0, self.length, ncol), np.linspace(-self.width / 2, self.width / 2, nrow)
        X, Y = np.meshgrid(xs, ys)
        h = np.zeros_like(X)
        for s, x0 in zip(self.sections, self.starts):
            fn, length = SECTIONS[s["type"]]
            L = length(s)
            sel = (X >= x0) & (X < x0 + L)
            h = np.where(sel, fn(X - x0, Y, {**s, "_width": self.width}), h)
        return h

    def barriers(self) -> list[dict]:
        """Every barrier, in world coordinates: x0..x1 along the track, y0..y1 across it, its height, the bonus
        for clearing it, and which side the bypass lane is on (bypass_y = the lane's centre)."""
        out = []
        for s, x0 in zip(self.sections, self.starts):
            if s["type"] != "barrier":
                continue
            a = x0 + float(s.get("run_in", 2.0))
            y0, y1 = barrier_span(s, self.width)
            bypass_y = (-self.width / 2 + y0) / 2 if s.get("side", "left") == "left" else (y1 + self.width / 2) / 2
            out.append({"x0": a, "x1": a + float(s.get("depth", 0.3)), "y0": y0, "y1": y1,
                        "height": float(s.get("height", 0.4)), "bonus": float(s.get("bonus", 5.0)),
                        "bypass_y": bypass_y})
        return out

    @classmethod
    def load(cls, name_or_path: str | Path) -> "Track":
        p = Path(name_or_path)
        if not p.suffix:
            p = TRACKS_DIR / f"{name_or_path}.yaml"
        return cls.from_yaml(p.read_text(), default_name=p.stem)

    @classmethod
    def from_yaml(cls, text: str, default_name: str = "track") -> "Track":
        import yaml
        d = yaml.safe_load(text)
        return cls(d.get("name", default_name), d["sections"], float(d.get("width", 6.0)))

    def to_yaml(self) -> str:
        import yaml
        return yaml.safe_dump({"name": self.name, "width": self.width, "sections": self.sections}, sort_keys=False)


def generate(seed: int, difficulty: float = 0.5, n: int = 6) -> Track:
    """A random practice track: a run-up, n random sections of the given difficulty (0 easy .. 1 hard), a
    run-out. Every section type the competition uses can appear; sizes scale with difficulty."""
    rng = np.random.default_rng(seed)
    d = float(np.clip(difficulty, 0, 1))
    u = lambda lo, hi: float(np.round(rng.uniform(lo, hi), 3))   # noqa: E731
    makers = {
        "rough": lambda: {"type": "rough", "length": u(4, 7), "height": u(0.02, 0.03 + 0.05 * d), "seed": int(rng.integers(1e6))},
        "stairs": lambda: {"type": "stairs", "steps": int(rng.integers(2, 6)), "rise": u(0.04, 0.05 + 0.07 * d),
                           "tread": u(0.45, 0.6), "landing": u(1.5, 2.5)},
        "hurdles": lambda: {"type": "hurdles", "heights": sorted(u(0.04, 0.05 + 0.13 * d) for _ in range(int(rng.integers(2, 5)))),
                            "spacing": u(1.4, 2.0), "thickness": 0.1},
        "ramp": lambda: {"type": "ramp", "rise": u(0.2, 0.3 + 0.4 * d), "run": u(3.0, 4.5), "top": u(0.8, 1.5)},
        "bridge": lambda: {"type": "bridge", "length": u(4, 7), "width": u(1.2 - 0.3 * d, 1.4 - 0.3 * d),
                           "height": u(0.1, 0.1 + 0.15 * d), "ramp": 1.0},
        "waves": lambda: {"type": "waves", "length": u(3, 5), "amplitude": u(0.04, 0.05 + 0.1 * d), "wavelength": u(1.4, 2.0)},
    }
    kinds = list(makers)
    secs = [{"type": "flat", "length": 4.0}]
    for _ in range(n):
        secs.append(makers[kinds[int(rng.integers(len(kinds)))]]())
        secs.append({"type": "flat", "length": u(1.0, 2.0)})
    secs.append({"type": "flat", "length": 3.0})
    return Track(f"random_{seed}_d{d:g}", secs)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Write a random practice track as YAML.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--difficulty", type=float, default=0.5)
    ap.add_argument("--sections", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t = generate(a.seed, a.difficulty, a.sections)
    text = t.to_yaml()
    if a.out:
        Path(a.out).write_text(text)
        print(f"wrote {a.out} ({t.length:.1f} m)")
    else:
        print(text)


if __name__ == "__main__":
    main()
