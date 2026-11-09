"""How a track looks: a colour texture painted onto the heightfield, one colour per section type.

Purely visual (the physics never reads it). Painted from the track itself, so every track, including random and
secret ones, gets the same look: each section type has its own colour, with darker 1 m grid lines for scale;
stairs alternate shades step by step; hurdles and barriers are painted in yellow-and-black hazard stripes; a
bridge's walkway is bright with darker ground on either side; a barrier's bypass lane is green.
"""
from __future__ import annotations

import numpy as np

from .track import Track, barrier_span

PX_PER_M = 20

COLOURS = {                     # RGB 0..1
    "flat": (0.30, 0.34, 0.42),
    "rough": (0.55, 0.40, 0.26),
    "stairs": (0.93, 0.55, 0.18),
    "hurdles": (0.36, 0.30, 0.55),
    "ramp": (0.55, 0.32, 0.70),
    "bridge": (0.18, 0.62, 0.62),
    "waves": (0.20, 0.45, 0.80),
    "gap": (0.25, 0.25, 0.25),
    "barrier": (0.36, 0.38, 0.46),
}
HAZARD = ((0.98, 0.80, 0.10), (0.08, 0.08, 0.08))
BYPASS = (0.25, 0.70, 0.35)


def texture(track: Track) -> np.ndarray:
    """(height, width, 3) uint8 image covering the whole track: columns along x (0 .. length), rows along y
    (row 0 at -width/2)."""
    W, H = int(round(track.length * PX_PER_M)), int(round(track.width * PX_PER_M))
    x = (np.arange(W) + 0.5) / PX_PER_M
    y = (np.arange(H) + 0.5) / PX_PER_M - track.width / 2
    X, Y = np.meshgrid(x, y)
    img = np.zeros((H, W, 3))
    for s, x0 in zip(track.sections, track.starts):
        from .track import SECTIONS
        L = SECTIONS[s["type"]][1](s)
        sel = (X >= x0) & (X < x0 + L)
        c = np.array(COLOURS.get(s["type"], COLOURS["flat"]))
        img[sel] = c
        u = X - x0
        if s["type"] == "stairs":
            step = np.floor(u / float(s["tread"])).astype(int)
            shade = np.where(step % 2 == 0, 1.0, 0.82)[..., None]
            img = np.where(sel[..., None], c * shade, img)
        elif s["type"] == "hurdles":
            run_in, spacing, thick = float(s.get("run_in", 1.0)), float(s.get("spacing", 1.5)), float(s.get("thickness", 0.1))
            for i in range(len(s["heights"])):
                a = run_in + i * spacing
                bar = sel & (u >= a - 0.02) & (u < a + thick + 0.02)
                stripe = (np.floor((Y + 3) / 0.25).astype(int) % 2 == 0)
                img[bar & stripe], img[bar & ~stripe] = HAZARD
        elif s["type"] == "bridge":
            walk = sel & (np.abs(Y) <= float(s.get("width", 1.0)) / 2)
            img[sel & ~walk] = np.array(COLOURS["flat"]) * 0.7
            img[walk] = c
        elif s["type"] == "barrier":
            y0, y1 = barrier_span(s, track.width)
            run_in, depth = float(s.get("run_in", 2.0)), float(s.get("depth", 0.3))
            lane = sel & ~((Y >= y0) & (Y <= y1))
            img[lane] = BYPASS
            block = sel & (u >= run_in - 0.02) & (u < run_in + depth + 0.02) & (Y >= y0) & (Y <= y1)
            stripe = (np.floor((X + Y) / 0.2).astype(int) % 2 == 0)
            img[block & stripe], img[block & ~stripe] = HAZARD
    # 1 m grid lines, and a soft darkening towards the track's edges
    grid = (np.abs(X - np.round(X)) < 0.5 / PX_PER_M + 1e-9) | (np.abs(Y - np.round(Y)) < 0.5 / PX_PER_M + 1e-9)
    img[grid] *= 0.75
    edge = np.clip(1.0 - (np.abs(Y) - (track.width / 2 - 0.4)) / 0.4 * 0.35, 0.65, 1.0)
    img *= edge[..., None]
    return np.clip(np.round(img * 255), 0, 255).astype(np.uint8)
