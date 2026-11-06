"""Fast checks that the simulator, the runner, the grader's signing and the examples work."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from sprout_challenge import Sim, Track, generate
from sprout_challenge.grade import MARK, _mac, grade, verify
from sprout_challenge.runner import run

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_tracks_build_and_stand():
    for t in (Track.load("practice"), Track.load("flat"), generate(3, 0.8)):
        sim = Sim(t)
        obs = sim.reset(0)
        for _ in range(100):
            obs = sim.step(sim.default_pose)
        assert sim.status() is None, t.name
        assert obs.height_scan.shape == (13, 7) and abs(obs.height_scan[4, 3] + 0.85) < 0.05


def test_practice_heights():
    t = Track.load("practice")
    h, sim = t.heights(), Sim(t)
    assert h.min() == 0.0 and abs(h.max() - 0.55) < 1e-9
    assert abs(sim.ground(np.array([t.starts[3] + 4.5, 0.0])) - 0.32) < 1e-6        # stairs landing


def test_stand_still_runs_out_of_time():
    r = run(EXAMPLES / "stand_still", Track.load("flat"), seed=0, max_time=2.0)
    assert r.outcome == "out of time" and r.error is None


def test_broken_controller_is_reported(tmp_path):
    (tmp_path / "controller.py").write_text(
        "from sprout_challenge import Controller\nclass Bad(Controller):\n    def act(self, obs):\n        return [0.0]\n")
    r = run(tmp_path, Track.load("flat"), seed=0, max_time=1.0)
    assert r.error and "shape" in r.error


def test_signed_result_round_trip_and_forgery():
    key = os.urandom(32).hex()
    res = {"score": {"finished": True, "time_s": 50.0}}
    line = MARK + json.dumps({"result": res, "mac": _mac(key, res)})
    assert verify(key, "noise\n" + line + "\n") == res
    forged = MARK + json.dumps({"result": {"score": {"finished": True, "time_s": 1.0}}, "mac": _mac(key, res)})
    with pytest.raises(ValueError):
        verify(key, forged)
    with pytest.raises(ValueError):
        verify(key, line + "\n" + line)


@pytest.mark.slow
def test_ppo_baseline_finishes_practice():
    res = grade(EXAMPLES / "ppo_baseline", Track.load("practice"), runs=1, log=lambda m: None)
    assert res["score"]["finished"] and res["score"]["time_s"] < 70


def _carry_over(sim, b, z):
    """Move the robot rigidly across barrier b at torso height z (no physics), judging contacts each step."""
    import mujoco
    sim.reset(0)
    d = sim.data
    for x in np.linspace(b["x0"] - 1.5, b["x1"] + 1.5, 300):
        d.qpos[0:3] = [x, (b["y0"] + b["y1"]) / 2, z]
        d.qvel[:] = 0
        mujoco.mj_forward(sim.model, d)
        sim._judge_barriers()
    return sim.barrier_results()[0]


def test_barrier_judging():
    t = Track("b", [{"type": "flat", "length": 3}, {"type": "barrier", "bonus": 5}, {"type": "flat", "length": 4}])
    sim = Sim(t)
    b = sim.barriers[0]
    high = _carry_over(sim, b, 2.0)                  # sails over: clean, bonus
    assert high == {"route": "over", "clean": True, "bonus": 5.0}
    low = _carry_over(sim, b, 0.86)                  # legs through the barrier: touched, no bonus
    assert low["route"] == "over" and not low["clean"] and low["bonus"] == 0.0


@pytest.mark.slow
def test_baseline_takes_the_bypass():
    r = run(EXAMPLES / "ppo_baseline", Track.load("practice"), seed=0, max_time=120)
    assert r.outcome == "finished" and r.barriers[0]["route"] == "around" and r.bonus_s == 0.0
