"""The interface a submission implements.

    from sprout_challenge import Controller

    class MyController(Controller):
        def reset(self, info):
            self.pose = info["default_pose"]

        def act(self, obs):
            return self.pose            # 12 joint targets [rad]

`act` is called at 50 Hz with an Observation (sprout_challenge.sim.Observation) and returns the 12 joint
position targets for the robot's servos, in the order BR, BL, FL, FR legs x (thigh = abduction, leg = hip
fore-aft, shin = knee). Targets are clipped to the joint limits; a non-finite target keeps the previous one.

`reset` is called once before each run with a dict:
    dt              0.02 (s per act call)
    default_pose    (12,) the crouched standing pose the robot starts in
    joint_lower     (12,) joint limits [rad]
    joint_upper     (12,)
    scan_points     (13, 7, 2) where height_scan samples the terrain, body-heading frame [m]
    finish_x        where the finish line is [m] along the track (the robot starts at x = 1)
    track_width     [m]; the track is centred on y = 0

Each act call has a time budget (20 ms by default). A late answer is discarded: the robot holds its previous
targets for that step, as a real robot would.
"""
from __future__ import annotations

import numpy as np


class Controller:
    def reset(self, info: dict) -> None:
        pass

    def act(self, obs) -> np.ndarray:
        raise NotImplementedError
