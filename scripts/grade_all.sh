#!/usr/bin/env bash
# Grade every submission in a folder, each in its own locked-down container, then write the leaderboard.
#
#   scripts/grade_all.sh SUBMISSIONS_DIR TRACK_YAML [RESULTS_DIR] [SEED] [RUNS]
#
# SUBMISSIONS_DIR holds one folder per entry (each with controller.py). TRACK_YAML is the track to grade on: the
# practice track while testing, the secret arena for the final. SEED is the base seed for the randomised runs;
# keep the final one private.
#
# Each entry runs with no network, 1 CPU, 2 GB of memory, a read-only filesystem (plus a small /tmp), no Linux
# capabilities, as an unprivileged user, and a wall-clock limit. The track, seed and a fresh random key reach
# the grader only on stdin, so the entry never sees them as a file or argument; nothing on the host is mounted
# writable. The grader prints its result signed with the key, and a separate container (without the entry)
# checks the signature before the result is saved, so an entry cannot forge its score.
set -euo pipefail
subs="$(cd "$1" && pwd)"; track="$2"
results="$(mkdir -p "${3:-results}" && cd "${3:-results}" && pwd)"; seed="${4:-1000}"; runs="${5:-5}"
repo="$(cd "$(dirname "$0")/.." && pwd)"
docker image inspect sprout-challenge >/dev/null 2>&1 || docker build -q -f "$repo/docker/Dockerfile" -t sprout-challenge "$repo"
logs="$results/logs"; mkdir -p "$logs"
shopt -s nullglob
entries=("$subs"/*/)
[ ${#entries[@]} -gt 0 ] || { echo "no submission folders in $subs" >&2; exit 1; }
for dir in "${entries[@]}"; do
    name="$(basename "$dir")"
    echo "== $name"
    key="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
    cfg="$(python3 -c 'import json, sys; print(json.dumps({"track": open(sys.argv[1]).read(), "seed": int(sys.argv[2]),
        "runs": int(sys.argv[3]), "key": sys.argv[4]}))' "$track" "$seed" "$runs" "$key")"
    out="$(mktemp)"
    if printf '%s' "$cfg" | timeout --kill-after=30 "${ENTRY_TIMEOUT:-1800}" docker run -i --rm --name "sprout-grade-$name" \
        --network none --cpus 1 --memory 2g --pids-limit 256 --read-only --tmpfs /tmp:size=256m \
        --cap-drop ALL --security-opt no-new-privileges --user 65534:65534 \
        -v "$dir":/submission/"$name":ro \
        sprout-challenge python -m sprout_challenge.grade "/submission/$name" --stdin >"$out" 2> >(tee "$logs/$name.log" >&2)
    then
        if docker run -i --rm -e SPROUT_KEY="$key" sprout-challenge python -m sprout_challenge.grade --verify \
            <"$out" >"$results/$name.json.tmp" 2>>"$logs/$name.log"; then
            mv "$results/$name.json.tmp" "$results/$name.json"
        else
            rm -f "$results/$name.json.tmp"; echo "   $name: REJECTED (result missing or signature invalid; see $logs/$name.log)"
        fi
    else
        echo "   $name: grading failed or timed out (no result)"
    fi
    rm -f "$out"
done
docker run --rm -v "$results":/results:ro sprout-challenge python -m sprout_challenge.leaderboard /results \
    | tee "$results/leaderboard.md"
