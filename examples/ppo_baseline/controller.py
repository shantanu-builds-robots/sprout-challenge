"""Baseline: a neural-network policy trained with PPO (reinforcement learning) in sprout-gym's GPU simulator.

The policy is a 139 -> 512 -> 256 -> 128 -> 12 MLP saved as numpy arrays in policy.npz, so it needs nothing but
numpy here. Its input is built from the Observation exactly as in training:

  base_lin_vel (3), gyro (3), gravity (3), joint_pos - default_pose (12), joint_vel (12), its own last
  output (12), the steering command (3: forward speed, sideways speed, turn rate), and the terrain scan
  (91: torso height above each scan point minus 0.85 m, clipped to +-1)

and its output a in [-1, 1]^12 becomes joint targets default_pose + 0.5 a. The steering command comes from a
small heading controller that keeps it on the centreline, as in training, except near a barrier, where it steers
into the bypass lane (this policy never learned to jump: the barrier bonus is yours to win).
"""
from pathlib import Path

import numpy as np

from sprout_challenge import Controller

HERE = Path(__file__).resolve().parent


class PPOBaseline(Controller):
    speed = 0.8                 # m/s commanded; the policy was trained on 0.4 .. 0.8

    def reset(self, info):
        z = np.load(HERE / "policy.npz")
        self.mean, self.std = z["mean"], z["std"]
        self.layers = [(z[f"W{i}"], z[f"b{i}"]) for i in range(len([k for k in z.files if k.startswith("W")]))]
        self.pose = np.asarray(info["default_pose"], float)
        self.last = np.zeros(12)
        self.barriers = info.get("barriers", [])

    def lane(self, x):
        """Where to be across the track: the bypass lane from 3 m before a barrier to 1.5 m after it."""
        for b in self.barriers:
            if b["x0"] - 3.0 <= x <= b["x1"] + 1.5:
                return b["bypass_y"]
        return 0.0

    def command(self, obs):
        y, yaw = obs.position[1] - self.lane(obs.position[0]), obs.yaw
        want = np.clip(-0.4 * y, -0.3, 0.3)                 # head back towards the lane's centre
        vy = np.clip(-0.5 * y, -0.2, 0.2) * np.cos(yaw)
        wz = np.clip(-1.5 * (yaw - want), -0.6, 0.6)
        return np.array([self.speed, vy, wz])

    def act(self, obs):
        scan = np.clip(-obs.height_scan - 0.85, -1.0, 1.0).ravel()
        x = np.hstack([obs.base_lin_vel, obs.gyro, obs.gravity, obs.joint_pos - self.pose, obs.joint_vel,
                       self.last, self.command(obs), scan])
        x = (x - self.mean) / self.std
        for i, (W, b) in enumerate(self.layers):
            x = x @ W + b
            if i < len(self.layers) - 1:
                x = x / (1.0 + np.exp(-x))                  # swish
        self.last = np.tanh(x[:12])
        return self.pose + 0.5 * self.last
