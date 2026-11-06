"""Rank graded submissions: python -m sprout_challenge.leaderboard results/ [--out leaderboard.md]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def mean_cost(r: dict, max_time: float = 180.0) -> float:
    """Tie-break: average over all runs of the finish time, or for a run that did not finish, the time limit
    plus the metres it was short of the finish."""
    L = r["track_length_m"]
    return sum(x["score_s"] if x["outcome"] == "finished" else max_time + (L - x["distance_m"]) for x in r["runs"]) \
        / len(r["runs"])


def rank(results: list[dict]) -> list[dict]:
    def key(r):
        s = r["score"]
        return (0, s["score_s"], mean_cost(r)) if s["finished"] else (1, -s["distance_m"], mean_cost(r))
    return sorted(results, key=key)


def table(results: list[dict]) -> str:
    rows = ["| # | Submission | Score (median run) | Finish time | Bonus | Finishes | Notes |",
            "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rank(results), 1):
        s = r["score"]
        res = f"**{s['score_s']:.2f} s**" if s["finished"] else f"{s['outcome']}, {s['distance_m']:.1f} m"
        fin = f"{s['time_s']:.2f} s" if s["finished"] else "-"
        bonus = f"-{s['bonus_s']:g} s" if s["finished"] and s["bonus_s"] else "-"
        errs = sorted({x["error"].strip().splitlines()[-1] for x in r["runs"] if x.get("error")})
        rows.append(f"| {i} | {r['submission']} | {res} | {fin} | {bonus} | {r['finishes']}/{len(r['runs'])} "
                    f"| {'; '.join(errs)[:80]} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = [json.loads(p.read_text()) for p in sorted(Path(a.results).glob("*.json"))]
    track = res[0]["track"] if res else "?"
    text = f"# Leaderboard: {track}\n\n" + table(res) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
