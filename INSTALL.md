# Installation and first entry

This guide gets the Sprout Challenge running on your computer, then walks you from the examples to a submitted
entry. Plan on about 10 minutes for the install and an hour for a first entry of your own.

You don't need a GPU or a powerful computer: the simulator runs faster than real time on one CPU core.

**Contents:** [1. What you need](#1-what-you-need) · [2. Install](#2-install) · [3. Check it works](#3-check-it-works) ·
[4. Watch a robot run](#4-watch-a-robot-run) · [5. Your first entry](#5-your-first-entry-step-by-step) ·
[6. Check it like the judges](#6-check-it-like-the-judges-docker) · [7. Submit](#7-submit) ·
[8. Troubleshooting](#8-troubleshooting) · [9. Free GPUs for training](#9-free-gpus-for-training-optional)

## 1. What you need

| | |
|---|---|
| **Python** | 3.10, 3.11, 3.12 or 3.13 (`python3 --version`) |
| **OS** | Linux, macOS or Windows 10/11 |
| **Git** | to download the challenge and to submit ([git-scm.com](https://git-scm.com/downloads)) |
| **A GitHub account** | your entry is a public GitHub repository |
| **Docker** (optional) | to test your entry in the exact grading environment ([docker.com](https://www.docker.com/products/docker-desktop/)) |
| **Disk** | about 300 MB |

## 2. Install

Download the challenge and install it into its own virtual environment, so it can't clash with other Python
projects.

**Linux and macOS:**

```bash
git clone https://github.com/shantanu-builds-robots/sprout-challenge
cd sprout-challenge
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[video]"
```

**Windows** (PowerShell):

```powershell
git clone https://github.com/shantanu-builds-robots/sprout-challenge
cd sprout-challenge
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[video]"
```

This installs MuJoCo (the physics simulator), numpy, PyYAML and, for saving videos, imageio. Whenever you open a new
terminal, `cd` into the folder and activate the environment again (the `source` or `Activate.ps1` line) before
running anything.

Using conda instead? `conda create -n sprout python=3.11`, `conda activate sprout`, then the `pip install` line.

## 3. Check it works

Grade the hand-written trot example on the practice track, once:

```bash
python -m sprout_challenge.grade examples/trot --track practice --runs 1
```

You should see something like this within a minute:

```
trot on practice (56.8 m), 1 runs
  run 1/1: fell after 10.6 m (3:stairs, 23.7 s)
score: fell after 10.6 m (median of 1; 0 finished)
```

The trot is meant to fall at the stairs; it's there to show you a simple controller. If you got a score line, the
install works.

## 4. Watch a robot run

Open the 3D viewer and watch the reinforcement-learning baseline finish the practice track:

```bash
python -m sprout_challenge.run examples/ppo_baseline --view
```

Drag with the left mouse button to turn the camera, the right button to move it, and scroll to zoom. Close the
window to stop.

**On macOS** the viewer must be started with `mjpython` instead of `python` (it is installed with MuJoCo):

```bash
mjpython -m sprout_challenge.run examples/ppo_baseline --view
```

**No display** (a server, or WSL without graphics)? Save a video instead and open it afterwards:

```bash
python -m sprout_challenge.run examples/ppo_baseline --video baseline.mp4
```

## 5. Your first entry, step by step

### 5.1 Copy an example

```bash
cp -r examples/trot my_entry          # Windows: Copy-Item -Recurse examples\trot my_entry
```

Every entry is a folder with a `controller.py`. Open `my_entry/controller.py`: it has a `reset(info)` method, called
once before each run, and an `act(obs)` method, called 50 times a second, that returns 12 joint angles. The
[README](README.md#writing-a-controller) explains the joint order and every sensor in `obs`.

Which example to start from:

| Start from | If you want to |
|---|---|
| `examples/stand_still` | write everything yourself, from an empty controller |
| `examples/trot` | hand-tune a gait: about 100 lines, easy to read and change |
| `examples/ppo_baseline` | start from a neural network that already finishes the practice track |

### 5.2 Change something and watch it

Make a small change, then watch the result. In the trot, the gait's numbers are near the top of the file; try
stepping faster, from `freq = 2.35` to `freq = 2.6`:

```bash
python -m sprout_challenge.run my_entry --view
```

The trot's numbers were tuned so that changing any one of them by 10 to 20% still walks; bigger changes, or several
at once, can make it fall. Change one thing at a time and watch what it does.

`run` prints how the run ended, how far the robot got, and on which section. Add `--no-random` to turn off the
randomised robot and start while you debug, and `--seed N` to try a different random robot.

Your controller runs in its own process. `print` works in it, which is the quickest way to see what it's doing.

### 5.3 Score it

```bash
python -m sprout_challenge.grade my_entry --track practice
```

This runs 5 randomised runs and scores the median, exactly as the judges do ([RULES.md](RULES.md#7-scoring)). It also
reports **late steps**: `act` calls that took longer than 20 ms. Any late steps mean your controller is too slow;
the judges' machine runs your code on one core, so aim for a few milliseconds per call.

### 5.4 Practise on tracks you haven't seen

The competition arena has the practice track's sections in a secret order. Copy
`sprout_challenge/tracks/practice.yaml`, shuffle its sections (keep the first and last flat ones in place), and grade
on your copy. Generated tracks are good practice too:

```bash
python -m sprout_challenge.track --seed 7 --difficulty 0.8 --out track_7.yaml
python -m sprout_challenge.grade my_entry --track track_7.yaml
python -m sprout_challenge.run my_entry --track track_7.yaml --view
```

Try several seeds and difficulties between 0 and 1. A track file is plain YAML, so you can also write your own,
for example to practise one section type over and over.

### 5.5 Files your controller needs

Put weights, tables or helper modules in your entry folder and load them relative to `controller.py`:

```python
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
weights = np.load(HERE / "policy.npz")
```

At grading time only the standard library, numpy, scipy, onnxruntime and `sprout_challenge` are available. If you
train with PyTorch, JAX or anything else, export the network to ONNX or numpy arrays first.
[`examples/ppo_baseline`](examples/ppo_baseline/controller.py) shows a network running in plain numpy.

## 6. Check it like the judges (Docker)

The judges grade every entry in a Docker container with no network, 1 CPU and 2 GB of memory. Run the same
container on your entry before you submit; it catches missing files and packages that only exist on your computer.

Build the image once (from the `sprout-challenge` folder):

```bash
docker build -f docker/Dockerfile -t sprout-challenge .
```

Then grade your entry in it:

```bash
docker run --rm --network none --cpus 1 --memory 2g -v "$PWD/my_entry":/submission/my_entry:ro sprout-challenge \
    python -m sprout_challenge.grade /submission/my_entry --track practice
```

On Windows PowerShell, use `${PWD}` instead of `$PWD`. If the container's result differs from the one on your
computer, the container's is the one that counts.

## 7. Submit

1. **Create a new public repository** on GitHub ([github.com/new](https://github.com/new)), for example
   `my-sprout-entry`. Don't fork this repository: a fork carries the whole challenge code with it.
2. **Put your entry in it.** The easiest layout is your entry folder's contents at the top of the repository:

   ```bash
   cd my_entry
   git init -b main
   git add .
   git commit -m "My Sprout Challenge entry"
   git remote add origin https://github.com/<your-username>/my-sprout-entry.git
   git push -u origin main
   ```

   Check on GitHub that `controller.py` and every file it loads are there. Files ignored by a `.gitignore` (weights
   often are) won't be uploaded.
3. **DM the repository link** to [@shantanu.builds.robots](https://www.instagram.com/shantanu.builds.robots/) on
   Instagram.

You can keep improving it: push again any time before **30 November 2026, 23:59 IST**. Whatever is in the
repository at the deadline is what gets graded.

## 8. Troubleshooting

**`pip install` fails building MuJoCo.** Your Python version has no ready-made MuJoCo package. Use Python 3.10 to
3.13 (check with `python3 --version`), and update pip first: `pip install --upgrade pip`.

**`python3: command not found` (Windows).** Use `py` instead, or install Python from
[python.org](https://www.python.org/downloads/) with "Add python.exe to PATH" ticked.

**PowerShell won't run `Activate.ps1`** ("running scripts is disabled"). Run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

**`ModuleNotFoundError: No module named 'sprout_challenge'`.** The virtual environment isn't active in this terminal.
Activate it again (section 2).

**macOS: an error saying `launch_passive` requires the script to be run under `mjpython`.** Use `mjpython` instead of
`python` for `--view` (section 4).

**The viewer doesn't open, or `--video` fails with an OpenGL error, on Linux without a display.** Set an offscreen
renderer for videos: `MUJOCO_GL=egl` with an NVIDIA GPU, or `MUJOCO_GL=osmesa` after installing OSMesa
(`sudo apt install libosmesa6`):

```bash
MUJOCO_GL=egl python -m sprout_challenge.run my_entry --video out.mp4
```

**The grader reports late steps.** Your `act` takes over 20 ms. Do heavy work once in `reset` or at import time,
avoid allocating large arrays every step, and if you run a neural network, keep it small or use onnxruntime.

**`controller error` or `reset:` / `t=…` in the result.** Your controller raised an exception; the message after it
says where. Run with `python -m sprout_challenge.run my_entry` to see the full error.

**Something else?** DM [@shantanu.builds.robots](https://www.instagram.com/shantanu.builds.robots/) with the command
you ran and the full error message.

## 9. Free GPUs for training (optional)

You don't need a GPU to take part: the simulator, your controller and the grading all run on a CPU, and a
hand-written or optimisation-based controller never needs one. A GPU helps only if you train a neural network with
a GPU simulator such as [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) (how the baseline
was trained), where thousands of robots practise in parallel.

If you don't have one, these give you a GPU for free. Limits change often, so check each provider's own page; the
numbers below are approximate as of late 2026.

| Service | What you get for free | Good to know |
|---|---|---|
| [Kaggle Notebooks](https://www.kaggle.com/code) | about 30 GPU hours a week, sessions up to 12 h | the most predictable free quota; phone verification needed to turn on the GPU and internet |
| [Google Colab](https://colab.research.google.com/) | a T4 GPU when available, sessions up to about 12 h | free limits aren't published and vary; idle sessions disconnect, so save checkpoints to Google Drive |
| [Lightning AI](https://lightning.ai/pricing) | a starter block of free GPU hours plus monthly credits | a persistent workspace, closer to a real computer than a notebook |
| [SageMaker Studio Lab](https://studiolab.sagemaker.aws/) | about 4 GPU hours a day | reported closed to new sign-ups in 2026; existing accounts still work |

Many universities and cloud providers also give student or research credits.

**Tips for free GPUs:**

- Train on the GPU, then **export** the network (to ONNX or numpy arrays) and run it on the CPU in your
  `controller.py`. The grader has no GPU.
- Save checkpoints often: free sessions get cut off without warning.
- **Start from the ready-made notebook:** [`training/train_sprout.ipynb`](training/train_sprout.ipynb) trains a
  walking policy for Sprout the way the baseline was trained, on a free Colab or Kaggle GPU, then exports, grades
  and packs it as an entry ([open in Colab](https://colab.research.google.com/github/shantanu-builds-robots/sprout-challenge/blob/main/training/train_sprout.ipynb)).
  See [training/README.md](training/README.md).
