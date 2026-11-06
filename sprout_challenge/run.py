"""Run a controller on a track and watch it, or save a video.

    python -m sprout_challenge.run SUBMISSION [--track practice] [--seed 0] [--no-random] [--view]
                                   [--video out.mp4] [--max-time 180]

--view opens the MuJoCo viewer in real time (needs a display); --video renders a chase-camera mp4 (needs the
"video" extra: pip install sprout-challenge[video]). Without either, it just prints the result.
"""
from __future__ import annotations

import argparse
import json
import time

import mujoco
import numpy as np

from .runner import run
from .sim import CTRL_DT, Sim
from .track import Track


class _Video:
    def __init__(self, sim: Sim, path: str, width=1280, height=720, fps=25):
        import imageio.v2 as imageio
        self.sim, self.every = sim, max(1, int(round(1 / (fps * CTRL_DT))))
        self.r = mujoco.Renderer(sim.model, height, width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = sim.model.body("root").id
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 3.5, -18, 125
        self.w = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=8)
        self.i = 0

    def __call__(self, sim):
        if self.i % self.every == 0:
            self.r.update_scene(sim.data, self.cam)
            self.w.append_data(self.r.render())
        self.i += 1

    def close(self):
        self.w.close()


class _Viewer:
    def __init__(self, sim: Sim):
        import mujoco.viewer
        self.v = mujoco.viewer.launch_passive(sim.model, sim.data)
        self.v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.v.cam.trackbodyid = sim.model.body("root").id
        self.v.cam.distance, self.v.cam.elevation, self.v.cam.azimuth = 3.5, -18, 125
        self.t0 = time.perf_counter()

    def __call__(self, sim):
        self.v.sync()
        lag = sim.time - (time.perf_counter() - self.t0)
        if lag > 0:
            time.sleep(lag)

    def close(self):
        self.v.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("submission")
    ap.add_argument("--track", default="practice")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-random", action="store_true", help="nominal robot, no sensor noise")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--video", default=None)
    ap.add_argument("--max-time", type=float, default=180.0)
    ap.add_argument("--budget", type=float, default=0.02)
    a = ap.parse_args()
    track = Track.load(a.track)
    sim = Sim(track)
    hook = _Viewer(sim) if a.view else _Video(sim, a.video) if a.video else None
    try:
        r = run(a.submission, track, a.seed, not a.no_random, a.max_time, a.budget, on_step=hook, sim=sim)
    finally:
        if hook:
            hook.close()
    print(json.dumps(r.to_dict(), indent=1))
    if a.video:
        print("video:", a.video)


if __name__ == "__main__":
    main()
