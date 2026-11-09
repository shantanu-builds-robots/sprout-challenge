"""The reference simulator: Sprout on a track, in MuJoCo on the CPU. This is what grades submissions.

Physics every 5 ms, control every 20 ms (50 Hz): each control step applies the controller's 12 joint targets to
the robot's position servos (150 N*m, the real motors' limit) and runs four physics steps.

Barriers: where the track splits, a barrier covers one side and a bypass lane stays open on the other. When
the torso passes a barrier's middle, its position across the track decides the route ("over" or "around"). A
crossing is clean if no part of the robot touched the barrier (its top or a face above the ground) on the way;
a clean crossing "over" earns the barrier's bonus.

The run ends when the robot
  - finishes: its torso crosses the finish line;
  - falls: the torso touches the ground, sinks within FALL_HEIGHT of the terrain under it, or tilts past 60 deg;
  - leaves the track: its centre gets within EDGE of the track's side;
  - or runs out of time.

Each run can be randomised (as the grader does, with seeds participants don't see): foot friction, body mass,
motor strength, sensor noise, and a slightly different starting position and heading.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from . import look
from .track import BASE, RES, START_X, Track

ASSETS = Path(__file__).resolve().parent / "assets"
CTRL_DT, SIM_DT = 0.02, 0.005
FALL_HEIGHT = 0.3       # m: torso centre this close to the terrain is a fall
MAX_TILT_COS = 0.5      # cos 60 deg
EDGE = 0.3              # m from the track's side
LEGS = ("back_right", "back_left", "front_left", "front_right")
JOINTS = tuple(f"{l}_{j}_joint" for l in LEGS for j in ("thigh", "leg", "shin"))

# Terrain scan: a 13 x 7 grid around the robot, turned with its heading (x forward, y left), in metres.
SCAN_X = np.linspace(-0.8, 1.6, 13)
SCAN_Y = np.linspace(-0.6, 0.6, 7)
SCAN_POINTS = np.stack(np.meshgrid(SCAN_X, SCAN_Y, indexing="ij"), -1)        # (13, 7, 2)


@dataclass
class Randomization:
    """How much a randomised run varies. The grader uses these defaults."""
    foot_friction: tuple = (0.6, 1.0)
    mass_scale: tuple = (0.95, 1.05)
    motor_scale: tuple = (0.95, 1.05)
    start_offset: float = 0.05          # m, +- in y
    start_yaw: float = 0.05             # rad, +-
    noise: dict = None

    def __post_init__(self):
        if self.noise is None:
            self.noise = {"joint_pos": 0.01, "joint_vel": 0.2, "gyro": 0.05, "accelerometer": 0.2, "gravity": 0.02,
                          "base_lin_vel": 0.05, "position": 0.02, "yaw": 0.01, "height_scan": 0.01}


@dataclass
class Observation:
    """What the controller sees each step. Frames: "body" = the torso (x forward, y left, z up); "world" = the
    track (x along the track towards the finish, y left, z up, ground level 0)."""
    time: float                 # s since the start
    joint_pos: np.ndarray       # (12,) rad; order BR, BL, FL, FR legs x (thigh = abduction, leg = hip, shin = knee)
    joint_vel: np.ndarray       # (12,) rad/s
    gyro: np.ndarray            # (3,) rad/s, body frame
    accelerometer: np.ndarray   # (3,) m/s^2, body frame (includes gravity: about +9.81 z when level)
    gravity: np.ndarray         # (3,) unit vector of gravity in the body frame ((0, 0, -1) when level)
    base_lin_vel: np.ndarray    # (3,) m/s, body frame (as a state estimator would give it)
    position: np.ndarray        # (3,) m, torso centre, world frame (as motion capture would give it)
    yaw: float                  # rad, heading in the world frame (0 = facing the finish)
    foot_contact: np.ndarray    # (4,) bool, BR, BL, FL, FR
    height_scan: np.ndarray     # (13, 7) m: terrain height at SCAN_POINTS minus the torso's height (about -0.86
                                #   on flat ground); a step up ahead reads less negative
    last_action: np.ndarray     # (12,) the joint targets applied last step


def _scene_xml(track: Track, nrow: int, ncol: int, hmin: float, hrange: float, tex_hw: tuple) -> str:
    robot = (ASSETS / "robot.xml").read_text()
    robot = robot.replace('meshdir="meshes"', f'meshdir="{ASSETS / "meshes"}"')
    robot = robot[:robot.rindex("</mujoco>")]
    return robot + f'''
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096"/>
    <map zfar="300"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="ground" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="ground" texture="ground" texuniform="true" texrepeat="2 2" reflectance="0.1"/>
    <!-- Section colours (look.py), painted into this texture's pixels after loading. -->
    <texture name="paint" type="2d" builtin="flat" rgb1="1 1 1" width="{tex_hw[1]}" height="{tex_hw[0]}"/>
    <material name="track" texture="paint" texuniform="false" texrepeat="1 1" specular="0.2" shininess="0.3"/>
    <material name="line" rgba="1 1 1 1" emission="0.5"/>
    <hfield name="track" nrow="{nrow}" ncol="{ncol}" size="{track.length / 2} {track.width / 2} {hrange} {BASE}"/>
  </asset>
  <worldbody>
    <light pos="{track.length / 2} 0 20" dir="0 0 -1" directional="true" castshadow="true"/>
    <geom name="floor" type="hfield" hfield="track" pos="{track.length / 2} 0 {hmin}" material="track"
      contype="1" conaffinity="0"/>
    <geom name="start_line" type="box" pos="{START_X} 0 0.003" size="0.05 {track.width / 2} 0.002" material="line"
      contype="0" conaffinity="0"/>
    <geom name="finish_line" type="box" pos="{track.finish_x} 0 0.003" size="0.05 {track.width / 2} 0.002"
      material="line" contype="0" conaffinity="0"/>
  </worldbody>
  <include file="{ASSETS / "contacts.xml"}"/>
  <include file="{ASSETS / "keyframes.xml"}"/>
</mujoco>
'''


class Sim:
    """One robot on one track. reset() then step(action) until done."""

    def __init__(self, track: Track):
        self.track = track
        h = track.heights()
        self._h = h
        hmin = min(float(h.min()), 0.0)
        hrange = max(float(h.max()) - hmin, 0.01)
        nrow, ncol = h.shape
        paint = look.texture(track)
        self.model = mujoco.MjModel.from_xml_string(_scene_xml(track, nrow, ncol, hmin, hrange, paint.shape[:2]))
        self.model.hfield_data[:] = ((h - hmin) / hrange).ravel()
        t = self.model.texture("paint").id
        a, n = self.model.tex_adr[t], paint.size
        self.model.tex_data[a:a + n] = paint[::-1].ravel()       # MuJoCo's first texture row is the far (+y) edge
        self.model.opt.timestep = SIM_DT
        self.data = mujoco.MjData(self.model)
        m = self.model
        self._torso = m.body("root").id
        self._imu = m.site("imu").id
        self._qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in JOINTS])
        self._vadr = np.array([m.jnt_dofadr[m.joint(j).id] for j in JOINTS])
        self._act = np.array([m.actuator(j).id for j in JOINTS])
        self.joint_lower, self.joint_upper = np.array([m.jnt_range[m.joint(j).id] for j in JOINTS]).T
        self.default_pose = np.array(m.keyframe("home").qpos[7:])
        self._home_z = float(m.keyframe("home").qpos[2])
        self._feet = [m.sensor(f"{l}_foot_floor_found").id for l in LEGS]
        self._torso_contact = m.sensor("torso_floor_found").id
        self._nominal = {"friction": m.geom_friction.copy(), "mass": m.body_mass.copy(),
                         "inertia": m.body_inertia.copy(), "gain": m.actuator_gainprm.copy(),
                         "bias": m.actuator_biasprm.copy(), "force": m.actuator_forcerange.copy()}
        self._foot_geoms = [m.geom(f"{l}_foot").id for l in LEGS]
        self._floor = m.geom("floor").id
        self.barriers = track.barriers()
        self._barrier_state = []
        self.rng = np.random.default_rng(0)
        self.noise = None
        self.last_action = self.default_pose.copy()

    # ------------------------------------------------------------------ terrain

    def ground(self, xy: np.ndarray) -> np.ndarray:
        """Terrain height [m] under each (x, y) (nearest cell); points off the track read 0."""
        xy = np.asarray(xy, float)
        nrow, ncol = self._h.shape
        col = np.round(xy[..., 0] / RES).astype(int)
        row = np.round((xy[..., 1] + self.track.width / 2) / RES).astype(int)
        inside = (col >= 0) & (col < ncol) & (row >= 0) & (row < nrow)
        return np.where(inside, self._h[np.clip(row, 0, nrow - 1), np.clip(col, 0, ncol - 1)], 0.0)

    # ------------------------------------------------------------------ run

    def reset(self, seed: int | None = None, randomization: Randomization | None = None) -> Observation:
        m, d, nom = self.model, self.data, self._nominal
        self.rng = np.random.default_rng(seed)
        r = randomization
        m.geom_friction[:], m.body_mass[:], m.body_inertia[:] = nom["friction"], nom["mass"], nom["inertia"]
        m.actuator_gainprm[:], m.actuator_biasprm[:], m.actuator_forcerange[:] = nom["gain"], nom["bias"], nom["force"]
        y0 = yaw0 = 0.0
        if r is not None:
            u = self.rng.uniform
            for g in self._foot_geoms:
                m.geom_friction[g, 0] = u(*r.foot_friction)
            s = u(*r.mass_scale)
            m.body_mass[self._torso] *= s
            m.body_inertia[self._torso] *= s
            k = u(*r.motor_scale)
            m.actuator_gainprm[:, 0] *= k
            m.actuator_biasprm[:, 1:3] *= k
            m.actuator_forcerange[:] *= k
            y0, yaw0 = u(-r.start_offset, r.start_offset), u(-r.start_yaw, r.start_yaw)
            self.noise = r.noise
        else:
            self.noise = None
        self._barrier_state = [{"route": None, "touched": False, "done": False} for _ in self.barriers]
        mujoco.mj_resetDataKeyframe(m, d, m.keyframe("home").id)
        d.qpos[0:3] = [START_X, y0, self._home_z]
        d.qpos[3:7] = [np.cos(yaw0 / 2), 0, 0, np.sin(yaw0 / 2)]
        d.ctrl[self._act] = self.default_pose
        self.last_action = self.default_pose.copy()
        mujoco.mj_forward(m, d)
        return self.observe()

    def step(self, action: np.ndarray) -> Observation:
        a = np.asarray(action, float).reshape(12)
        if not np.all(np.isfinite(a)):
            a = self.last_action
        a = np.clip(a, self.joint_lower, self.joint_upper)
        self.data.ctrl[self._act] = a
        self.last_action = a
        for _ in range(int(round(CTRL_DT / SIM_DT))):
            mujoco.mj_step(self.model, self.data)
            if self.barriers:
                self._judge_barriers()
        return self.observe()

    def _judge_barriers(self):
        d = self.data
        x, y = d.xpos[self._torso][:2]
        for b, st in zip(self.barriers, self._barrier_state):
            if st["done"]:
                continue
            for c in d.contact[:d.ncon]:
                if self._floor not in (c.geom1, c.geom2):
                    continue
                px, py, pz = c.pos
                if b["x0"] - 0.03 <= px <= b["x1"] + 0.03 and b["y0"] - 0.03 <= py <= b["y1"] + 0.03 and pz > 0.02:
                    st["touched"] = True
            if st["route"] is None and x >= (b["x0"] + b["x1"]) / 2:
                st["route"] = "over" if b["y0"] <= y <= b["y1"] else "around"
            if x >= b["x1"] + 0.8:                  # the hind legs are past it too
                st["done"] = True

    def barrier_results(self) -> list[dict]:
        """Per barrier: the route taken (None if not reached), whether it was clean, and the bonus earned."""
        out = []
        for b, st in zip(self.barriers, self._barrier_state):
            clean = st["route"] is not None and not st["touched"]
            out.append({"route": st["route"], "clean": clean,
                        "bonus": b["bonus"] if st["route"] == "over" and clean and st["done"] else 0.0})
        return out

    @property
    def time(self) -> float:
        return float(self.data.time)

    @property
    def torso_pos(self) -> np.ndarray:
        return self.data.xpos[self._torso].copy()

    def status(self) -> str | None:
        """None while running, else "finished", "fell" or "off track"."""
        d = self.data
        p = d.xpos[self._torso]
        if p[0] >= self.track.finish_x:
            return "finished"
        R = d.xmat[self._torso].reshape(3, 3)
        foot = np.array([[x, y] for x in (-0.45, 0.0, 0.45) for y in (-0.22, 0.0, 0.22)])
        under = p[:2] + foot @ R[:2, :2].T
        if (d.sensordata[self.model.sensor_adr[self._torso_contact]] > 0 or R[2, 2] < MAX_TILT_COS
                or p[2] - float(np.mean(self.ground(under))) < FALL_HEIGHT):
            return "fell"
        if abs(p[1]) > self.track.width / 2 - EDGE:
            return "off track"
        return None

    def observe(self) -> Observation:
        m, d = self.model, self.data
        sens = lambda name: d.sensordata[m.sensor_adr[m.sensor(name).id]:][:m.sensor_dim[m.sensor(name).id]].copy()  # noqa: E731
        R = d.xmat[self._torso].reshape(3, 3)
        p = d.xpos[self._torso].copy()
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        c, s = np.cos(yaw), np.sin(yaw)
        pts = p[:2] + SCAN_POINTS @ np.array([[c, s], [-s, c]])
        scan = self.ground(pts) - p[2]
        o = Observation(
            time=float(d.time), joint_pos=d.qpos[self._qadr].copy(), joint_vel=d.qvel[self._vadr].copy(),
            gyro=sens("gyro"), accelerometer=sens("accelerometer"), gravity=R.T @ np.array([0.0, 0.0, -1.0]),
            base_lin_vel=sens("local_linvel"), position=p, yaw=yaw,
            foot_contact=np.array([d.sensordata[m.sensor_adr[i]] > 0 for i in self._feet]),
            height_scan=scan, last_action=self.last_action.copy())
        if self.noise:
            n, u = self.noise, self.rng.uniform
            for k in ("joint_pos", "joint_vel", "gyro", "accelerometer", "gravity", "base_lin_vel", "position",
                      "height_scan"):
                v = getattr(o, k)
                setattr(o, k, v + u(-n[k], n[k], v.shape))
            o.yaw += u(-n["yaw"], n["yaw"])
        return o
