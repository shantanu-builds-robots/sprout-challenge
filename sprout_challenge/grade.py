"""Grade one submission: several randomised runs on a track, scored by the median run.

    python -m sprout_challenge.grade SUBMISSION [--track practice] [--runs 5] [--seed 1000] [--out result.json]

SUBMISSION is a folder with controller.py (plus any files it loads), or a single .py file.

Each run gets its own seed: foot friction, body mass, motor strength, sensor noise and the start pose all vary a
little (sim.Randomization). A run's score is its finish time if it finished, else the distance it covered; any
finish beats any non-finish. The entry's score is its median run, so one lucky or unlucky run does not decide it.

Official grading (scripts/grade_all.sh) uses --stdin: the track, seed, run count and a one-time key arrive as
JSON on standard input, so the submission, which runs in the same container, never sees them in a file or on a
command line. The result is printed signed with the key (HMAC-SHA256) and checked outside the container with
--verify, so a submission cannot forge it.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
from pathlib import Path

from .runner import run
from .track import Track

MARK = "SPROUT-RESULT "


def run_key(r: dict):
    """Sort key: finishers by score (finish time minus barrier bonuses), then everyone else by distance."""
    return (0, r["score_s"], 0.0) if r["outcome"] == "finished" else (1, 0.0, -r["distance_m"])


def grade(submission, track: Track, runs=5, seed=1000, max_time=180.0, budget=0.02, log=print, quiet=False) -> dict:
    results = []
    for i in range(runs):
        r = run(submission, track, seed=seed + i, randomize=True, max_time=max_time, budget=budget,
                quiet=quiet).to_dict()
        results.append(r)
        bonus = f" - {r['bonus_s']:g} s bonus = {r['score_s']:.2f} s" if r["outcome"] == "finished" and r["bonus_s"] else ""
        what = f"FINISHED in {r['time_s']:.2f} s{bonus}" if r["outcome"] == "finished" else \
            f"{r['outcome']} after {r['distance_m']:.1f} m ({r['section']}, {r['time_s']:.1f} s)"
        log(f"  run {i + 1}/{runs}: {what}" + (f", {r['late_steps']} late steps" if r["late_steps"] else "")
            + (f"\n    error: {r['error'].strip().splitlines()[-1]}" if r["error"] else ""))
    median = sorted(results, key=run_key)[len(results) // 2]
    return {"submission": Path(submission).name, "track": track.name, "track_length_m": round(track.finish_x - 1.0, 2),
            "runs": results, "score": {"finished": median["outcome"] == "finished", "time_s": median["time_s"],
                                       "bonus_s": median["bonus_s"], "score_s": median["score_s"],
                                       "distance_m": median["distance_m"], "outcome": median["outcome"]},
            "finishes": sum(r["outcome"] == "finished" for r in results)}


def _mac(key: str, res: dict) -> str:
    return hmac.new(bytes.fromhex(key), json.dumps(res, sort_keys=True).encode(), hashlib.sha256).hexdigest()


def verify(key: str, text: str) -> dict:
    """The signed result in a grader's output, checked against the key; raises if missing or forged."""
    lines = [ln for ln in text.splitlines() if ln.startswith(MARK)]
    if len(lines) != 1:
        raise ValueError(f"expected one signed result line, found {len(lines)}")
    blob = json.loads(lines[0][len(MARK):])
    if not hmac.compare_digest(_mac(key, blob["result"]), blob["mac"]):
        raise ValueError("result signature does not match: output was tampered with")
    return blob["result"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("submission", nargs="?")
    ap.add_argument("--track", default="practice", help="a track name in tracks/ or a path to a track YAML")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--max-time", type=float, default=180.0)
    ap.add_argument("--budget", type=float, default=0.02, help="seconds per act call")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stdin", action="store_true", help="official grading: settings and key as JSON on stdin")
    ap.add_argument("--verify", action="store_true", help="check a signed grader output on stdin (key in $SPROUT_KEY)")
    a = ap.parse_args()

    if a.verify:
        res = verify(os.environ["SPROUT_KEY"], sys.stdin.read())
        print(json.dumps(res, indent=1))
        return

    if a.stdin:
        cfg = json.loads(sys.stdin.read())
        track = Track.from_yaml(cfg["track"])
        key, seed, runs = cfg["key"], int(cfg["seed"]), int(cfg.get("runs", 5))
        log = lambda m: print(m, file=sys.stderr, flush=True)       # noqa: E731
        log(f"{Path(a.submission).name} on {track.name} ({track.finish_x - 1.0:.1f} m), {runs} runs")
        res = grade(a.submission, track, runs, seed, a.max_time, a.budget, log=log, quiet=True)
        print(MARK + json.dumps({"result": res, "mac": _mac(key, res)}), flush=True)
        return

    track = Track.load(a.track)
    print(f"{Path(a.submission).name} on {track.name} ({track.finish_x - 1.0:.1f} m), {a.runs} runs", flush=True)
    res = grade(a.submission, track, a.runs, a.seed, a.max_time, a.budget, log=lambda m: print(m, flush=True))
    s = res["score"]
    print(f"score: {'%.2f s (finished in %.2f s, %g s bonus)' % (s['score_s'], s['time_s'], s['bonus_s']) if s['finished'] else '%s after %.1f m' % (s['outcome'], s['distance_m'])}"
          f" (median of {a.runs}; {res['finishes']} finished)")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
