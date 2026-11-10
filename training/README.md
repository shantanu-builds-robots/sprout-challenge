# Training a walking policy on a GPU

Optional. The challenge and its grading run on a CPU; this folder is for people who want to train a neural network
policy the way [`examples/ppo_baseline`](../examples/ppo_baseline) was trained.

| File | What it is |
|---|---|
| [`train_sprout.ipynb`](train_sprout.ipynb) | the notebook: install, train, export, grade, watch, download. Runs on a free Colab or Kaggle GPU. |
| [`sprout_walk.py`](sprout_walk.py) | the training world (thousands of robots on generated tracks, on the GPU), the reward, Brax PPO settings, and the export to `policy.npz` |

**Open it in Colab:**
[colab.research.google.com/github/shantanu-builds-robots/sprout-challenge/blob/main/training/train_sprout.ipynb](https://colab.research.google.com/github/shantanu-builds-robots/sprout-challenge/blob/main/training/train_sprout.ipynb)

On Kaggle: New Notebook → File → Import Notebook → GitHub, and paste this repository's URL.

**On your own computer** (Linux with an NVIDIA GPU), in a separate virtual environment from the challenge's:

```bash
pip install "jax[cuda12]==0.10.2" "brax==0.14.2" "playground==0.2.0" "mujoco==3.17.0" "mujoco-mjx==3.17.0" \
    "mujoco-warp==3.17.0" "warp-lang==1.19.0" pyyaml
PYTHONPATH=. python -c "
from training import sprout_walk as sw
env = sw.WalkEnv()
params = sw.train(env, num_timesteps=120_000_000, checkpoint_dir='checkpoints')
sw.export(params, 'my_entry/policy.npz')"
cp examples/ppo_baseline/controller.py my_entry/
```

The training packages use a newer MuJoCo than the grader (3.17 vs 3.3); grade your entry in the challenge's own
environment (or the Docker image), as the notebook does.

The trained policy reads exactly the inputs `examples/ppo_baseline/controller.py` builds from an `Observation`, so
that controller runs any policy exported here. Change the reward weights (`REWARD_SCALES`), the speeds it
practises, or the training tracks, and see what happens.
