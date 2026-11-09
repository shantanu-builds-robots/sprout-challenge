# Sprout Challenge 🌱🏁

**Write a controller that gets Sprout down an obstacle course. Fastest robot wins $20.**

Sprout is a 28 kg four-legged robot with 12 motors (150 N·m each) and legs about a metre long. Your controller
gets its sensors 50 times a second and sends back 12 joint angles. Between the start and the finish line are
bumps, stairs, hurdles, a ramp, a narrow bridge, waves, and a fork where the brave **jump a 40 cm barrier for
a time bonus** while everyone else walks around it. Use any method you like: a hand-written gait, PID,
model-predictive control, reinforcement learning, or something nobody has tried yet.

| | |
|---|---|
| **Robot** | Sprout ([sprout-gym](https://github.com/shantanu-builds-robots/sprout-gym)), simulated in MuJoCo |
| **Your job** | `act(observation) -> 12 joint targets`, called at 50 Hz |
| **Practice** | the practice track in this repo, plus a generator for endless variations |
| **Final** | a secret arena built from the same section types, revealed with the results |
| **Prize** | $20 for the fastest robot (see [RULES.md](RULES.md)) |

## Quick start

```bash
git clone https://github.com/shantanu-builds-robots/sprout-challenge && cd sprout-challenge
pip install -e ".[video]"          # Python 3.10+; MuJoCo, numpy, PyYAML (+ imageio for videos)

python -m sprout_challenge.run examples/ppo_baseline --view                  # watch the baseline (needs a display)
python -m sprout_challenge.run examples/trot --video trot.mp4                # or save a video
python -m sprout_challenge.grade examples/ppo_baseline --track practice      # score it like the judges will
```

The grader prints each run and the score:

```
ppo_baseline on practice (56.8 m), 5 runs
  run 1/5: FINISHED in 61.94 s
  ...
score: 61.94 s (finished in 61.94 s, 0 s bonus) (median of 5; 5 finished)
```

## Writing a controller

A submission is a folder with a `controller.py` that defines one `Controller` subclass:

```python
import numpy as np
from sprout_challenge import Controller

class MyController(Controller):
    def reset(self, info):
        self.pose = info["default_pose"]          # the crouched standing pose, (12,)

    def act(self, obs):
        targets = self.pose.copy()
        # ... your controller here ...
        return targets                            # 12 joint angles [rad]
```

Put anything else it needs (weights, lookup tables) in the same folder and load it relative to `__file__`.
Start from one of the examples:

| Example | What it is | Practice track (median of 5) |
|---|---|---|
| [`examples/stand_still`](examples/stand_still/controller.py) | holds the starting pose: the smallest valid entry | 0 m |
| [`examples/trot`](examples/trot/controller.py) | a hand-written trot with steering and balance, ~100 lines | ~10 m (stopped at the stairs) |
| [`examples/ppo_baseline`](examples/ppo_baseline/controller.py) | a neural network trained with reinforcement learning, run in numpy; walks around the barrier | **finishes in 61.9 s**, no bonus |

### Joints and actions

Joint order is **back-right, back-left, front-left, front-right** legs, three joints each:

| Joint | Moves | Near the standing pose |
|---|---|---|
| `thigh` (abduction) | the leg sideways | + moves the foot outward, ~0.83 m/rad |
| `leg` (hip) | the leg fore and aft | + moves the foot forward, ~0.86 m/rad |
| `shin` (knee) | the knee | + moves the foot forward ~0.37 m/rad and up ~0.25 m/rad |

`act` returns the 12 target angles for the robot's position servos (stiffness 200 N·m/rad, torque limit
150 N·m). Targets are clipped to the joint limits (`info["joint_lower"]`, `info["joint_upper"]`); a non-finite
value keeps the previous target.

### What the controller sees

`act(obs)` gets an `Observation` ([sim.py](sprout_challenge/sim.py)) with sensor noise on:

| Field | Shape | Meaning |
|---|---|---|
| `time` | | seconds since the start |
| `joint_pos`, `joint_vel` | 12 | joint angles [rad] and velocities [rad/s] |
| `gyro` | 3 | body angular velocity [rad/s] |
| `accelerometer` | 3 | body acceleration [m/s²], gravity included |
| `gravity` | 3 | unit gravity vector in the body frame: `(0, 0, -1)` when level |
| `base_lin_vel` | 3 | body velocity [m/s], body frame (as a state estimator gives it) |
| `position` | 3 | torso centre [m]: x along the track, y left of the centreline, z up |
| `yaw` | | heading [rad]; 0 faces the finish |
| `foot_contact` | 4 | which feet touch the ground |
| `height_scan` | 13 × 7 | terrain height minus torso height [m] on a grid from 0.8 m behind to 1.6 m ahead and 0.6 m to each side, turned with the robot (`info["scan_points"]`); about −0.86 on flat ground |
| `last_action` | 12 | the targets applied last step |

`reset(info)` gets `dt` (0.02 s), `default_pose`, `joint_lower`, `joint_upper`, `scan_points`, `finish_x`,
`track_width` and `barriers`: a list with each barrier's `x0`/`x1` (along the track), `y0`/`y1` (the part of the
track it covers), `height`, `bonus` (seconds) and `bypass_y` (the centre of the lane around it).

### Time budget

Each `act` call has **20 ms** of wall-clock time on the grading machine (one CPU core). A late answer is thrown
away and the robot holds its previous targets for that step, as a real robot would. The grader reports late
steps; keep well under budget. Controllers run in their own process and only ever see observations.

## The track

The course runs along +x, 6 m wide. The robot starts at x = 1 m facing the finish, and the run ends when it:

- **finishes**: its torso crosses the finish line;
- **falls**: its torso touches the ground, drops within 0.3 m of the terrain, or tilts past 60°;
- **leaves the track**: within 0.3 m of the edge;
- or reaches the time limit (180 s).

Section types: `flat`, `rough`, `stairs`, `hurdles`, `ramp`, `bridge`, `waves`, `gap` (a trench) and
`barrier`. See
[track.py](sprout_challenge/track.py) for their parameters and
[tracks/practice.yaml](sprout_challenge/tracks/practice.yaml) for the practice track. Make new ones:

```bash
python -m sprout_challenge.track --seed 7 --difficulty 0.8 --out my_track.yaml
python -m sprout_challenge.grade my_entry --track my_track.yaml
```

The final arena uses the same section types with different sizes and order, so practise on variations, not
only on the practice track.

### Barriers: jump or go around

At a `barrier` the track forks. A solid block (40 cm high and 30 cm deep on the practice track) covers one side
of the track and the centreline; a lane about 2.5 m wide stays open on the other side.

- **Over:** the torso passes over the barrier and **no part of the robot touches it** (its top or a face above
  the ground). You earn the barrier's bonus: **5 s off your finish time** on the practice track.
- **Around:** walk through the open lane. No bonus, no penalty.
- **Touched it:** climbing onto it, kicking it or clipping it with a leg on the way over still lets you
  continue, but earns no bonus.

The route is decided where the torso passes the barrier's middle. Bonuses count only if you finish.

## Scoring

Each entry runs **5 times** on the arena. Every run is slightly different: foot friction (0.6–1.0), body mass
(±5%), motor strength (±5%), sensor noise and the start position all vary, from seeds only the organisers know.

- A run that finishes scores its **finish time minus any barrier bonuses**; one that doesn't scores the
  **distance** it covered. Any finish beats any non-finish.
- An entry's score is its **median run** (the 3rd best of 5), so one lucky or unlucky run doesn't decide it.
- Lowest median score wins; ties go to the better average over all 5 runs.

## Running exactly like the grader

The judges grade every entry in this Docker image, with no network, 1 CPU and 2 GB of memory:

```bash
docker build -f docker/Dockerfile -t sprout-challenge .
docker run --rm --network none --cpus 1 --memory 2g -v "$PWD/my_entry":/submission/my_entry:ro sprout-challenge \
    python -m sprout_challenge.grade /submission/my_entry --track practice
```

The image has Python 3.11, MuJoCo, numpy, scipy and onnxruntime; use only those at grading time (train with
anything, then export, for example to numpy arrays or ONNX).

## For organisers

```bash
python3 scripts/snapshot.py entries.tsv snapshots/                     # at the deadline: freeze every entry
scripts/grade_all.sh snapshots/submissions arena.yaml results/ <secret seed>   # grade everyone, write the leaderboard
```

`entries.tsv` lists `handle<TAB>repo URL`, one entry per line. Run the snapshot right at the deadline: git commit
dates can be set to anything, so the commit that counts is the one fetched then. `snapshots/manifest.tsv` records
each entry's commit, a hash of its files and the snapshot time; failed clones, missing `controller.py` and dropped
symlinks are reported there rather than guessed.

Keep `arena.yaml` and the seed private until results are announced (`arena*.yaml` is git-ignored). The script
gives each entry the track and seed only on stdin, never as a file inside its container, and accepts a result
only if it carries a valid signature from a one-time key, so an entry can neither read the arena from disk nor
forge its score. Inside the container the grader also makes itself non-dumpable (the entry cannot open its memory
or its stdin/stdout through `/proc`) and, between runs, kills every leftover process and empties `/tmp` and
`/dev/shm`, so nothing an entry learns on one run carries over to the next. Logs go to `results/logs/`.

## Licence

MIT. Sprout's model comes from [sprout-gym](https://github.com/shantanu-builds-robots/sprout-gym).
