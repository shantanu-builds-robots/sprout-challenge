"""Run a controller on a track: the loop shared by local runs and the grader.

The controller runs in its own process and only ever receives Observations: it cannot reach the simulator. Each
act call has a wall-clock budget; a late answer is discarded (the robot holds its previous targets for that
step). A controller that crashes, hangs past `hang_s`, or returns the wrong shape stops
controlling: the robot holds its last targets until the run ends.
"""
from __future__ import annotations

import importlib.util
import inspect
import multiprocessing as mp
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .controller import Controller
from .sim import CTRL_DT, SCAN_POINTS, Randomization, Sim
from .track import START_X, Track


def load_controller_class(path: str | Path):
    """`path` is a submission folder holding controller.py, a .py file, or "file.py:ClassName"."""
    path = str(path)
    name = None
    if ":" in path and not Path(path).exists():
        path, name = path.rsplit(":", 1)
    p = Path(path)
    if p.is_dir():
        p = p / "controller.py"
    sys.path.insert(0, str(p.parent))
    spec = importlib.util.spec_from_file_location(f"submission_{p.parent.name}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if name:
        return getattr(mod, name)
    found = [c for _, c in inspect.getmembers(mod, inspect.isclass)
             if issubclass(c, Controller) and c is not Controller and c.__module__ == mod.__name__]
    if len(found) != 1:
        raise ValueError(f"{p} must define exactly one Controller subclass (found {[c.__name__ for c in found]})")
    return found[0]


def _worker(path, conn, quiet=False):
    """Controller process: load the submission, then answer ("reset", info) and ("act", obs) messages. With
    quiet (official grading), the controller's stdout and stderr go nowhere, so it cannot write into the grader's
    output."""
    if quiet:
        import os
        null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null, 1)
        os.dup2(null, 2)
    try:
        ctrl = load_controller_class(path)()
        conn.send(("ok", None))
    except Exception:
        conn.send(("error", traceback.format_exc()))
        return
    while True:
        try:
            msg, payload = conn.recv()
        except EOFError:
            return
        if msg == "stop":
            return
        try:
            out = ctrl.reset(payload) if msg == "reset" else ctrl.act(payload)
            conn.send(("ok", None if msg == "reset" else np.asarray(out, float)))
        except Exception:
            conn.send(("error", traceback.format_exc()))


class ControllerProcess:
    def __init__(self, path, load_timeout=120.0, quiet=False):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker, args=(str(path), child, quiet), daemon=True)
        self.proc.start()
        if not self.conn.poll(load_timeout):
            self.close()
            raise TimeoutError("controller took too long to load")
        status, err = self.conn.recv()
        if status != "ok":
            self.close()
            raise RuntimeError(f"controller failed to load:\n{err}")

    def call(self, msg, payload, timeout):
        self.conn.send((msg, payload))
        if not self.conn.poll(timeout):
            raise TimeoutError(f"{msg} took longer than {timeout} s")
        status, out = self.conn.recv()
        if status != "ok":
            raise RuntimeError(out)
        return out

    def close(self):
        try:
            self.conn.send(("stop", None))
        except Exception:
            pass
        self.proc.join(1.0)
        if self.proc.is_alive():
            self.proc.kill()


@dataclass
class Result:
    track: str
    seed: int | None
    outcome: str                # finished, fell, off track, out of time, controller error
    time_s: float               # simulated seconds (the finish time if finished)
    distance_m: float           # progress along the track from the start line
    section: str                # where the run ended
    late_steps: int = 0         # act calls over budget
    barriers: list = field(default_factory=list)    # per barrier: route, clean, bonus
    bonus_s: float = 0.0        # seconds of bonus earned (counts only for a finish)
    score_s: float | None = None    # finish time minus bonus (finished runs only)
    error: str | None = None
    path: list = field(default_factory=list, repr=False)   # (t, x, y, z) every 0.1 s

    def to_dict(self, with_path=False):
        d = asdict(self)
        if not with_path:
            d.pop("path")
        return d


def run(controller_path, track: Track, seed: int | None = 0, randomize: bool = True, max_time: float = 180.0,
        budget: float = 0.02, hang_s: float = 5.0, on_step=None, sim: Sim | None = None,
        quiet: bool = False) -> Result:
    """One attempt. on_step(sim) is called after every control step (for viewers and videos)."""
    sim = sim or Sim(track)
    obs = sim.reset(seed, Randomization() if randomize else None)
    info = {"dt": CTRL_DT, "default_pose": sim.default_pose.copy(), "joint_lower": sim.joint_lower.copy(),
            "joint_upper": sim.joint_upper.copy(), "scan_points": SCAN_POINTS.copy(), "finish_x": track.finish_x,
            "track_width": track.width, "barriers": [dict(b) for b in sim.barriers]}
    error, late, outcome, path = None, 0, None, []
    try:
        ctrl = ControllerProcess(controller_path, quiet=quiet)
    except Exception as e:
        return Result(track.name, seed, "controller error", 0.0, 0.0, track.section_at(START_X), error=str(e))
    try:
        ctrl.call("reset", info, timeout=60.0)
    except Exception as e:
        error = f"reset: {e}"
    action, alive = sim.default_pose.copy(), error is None
    steps = int(round(max_time / CTRL_DT))
    for i in range(steps):
        if alive:
            t0 = time.perf_counter()
            try:
                out = ctrl.call("act", obs, timeout=hang_s)
                if out.shape != (12,):
                    raise ValueError(f"act returned shape {out.shape}, expected (12,)")
            except Exception as e:
                error, alive, out = f"t={sim.time:.2f} s: {e}", False, None
            took = time.perf_counter() - t0
            if out is not None:
                if took > budget:                 # late: discarded, the robot holds its previous targets
                    late += 1
                else:
                    action = out
        obs = sim.step(action)
        if on_step is not None:
            on_step(sim)
        if i % 5 == 0:
            p = sim.torso_pos
            path.append((round(sim.time, 2), *[round(float(v), 3) for v in p]))
        outcome = sim.status()
        if outcome:
            break
    ctrl.close()
    p = sim.torso_pos
    outcome = outcome or "out of time"
    dist = float(np.clip(p[0], START_X, track.finish_x) - START_X)
    bars = sim.barrier_results()
    bonus = sum(b["bonus"] for b in bars) if outcome == "finished" else 0.0
    score = round(sim.time - bonus, 2) if outcome == "finished" else None
    return Result(track.name, seed, outcome, round(sim.time, 2), round(dist, 2), track.section_at(float(p[0])),
                  late_steps=late, error=error, barriers=bars, bonus_s=round(bonus, 2), score_s=score, path=path)
