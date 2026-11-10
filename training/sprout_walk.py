"""Train a walking policy for the Sprout Challenge on a GPU, the way examples/ppo_baseline was trained.

    from training import sprout_walk as sw
    env = sw.WalkEnv()                                    # thousands of robots on generated tracks, on the GPU
    params = sw.train(env, num_timesteps=120_000_000)     # Brax PPO; about 30 min on a desktop GPU
    sw.export(params, "my_entry/policy.npz")              # numpy weights for examples/ppo_baseline/controller.py

Used by training/train_sprout.ipynb (Colab / Kaggle). Needs the GPU packages listed in training/README.md; the
challenge itself (and grading) does not.

What the robots practise on: a heightfield of LANES side-by-side tracks, each 6 m wide like the competition's.
Lane 0 is the practice track (its barrier replaced by flat ground), the rest come from sprout_challenge.track's
generator at a range of difficulties. Episodes are 20 s and start anywhere along a random lane, so every section
type gets practice.

What the policy sees (139 numbers, exactly what examples/ppo_baseline/controller.py builds from an Observation):
body velocity (3), gyro (3), gravity in the body frame (3), joint angles minus the standing pose (12), joint
velocities (12), its own previous action (12), the steering command (3: forward speed, sideways speed, turn rate)
and the terrain scan (91: torso height above each scan point minus 0.85 m, clipped to +-1). Its 12 outputs a in
[-1, 1] become joint targets standing_pose + 0.5 * a.

The command comes from a fixed heading controller (steer) that asks for the episode's speed and pulls the robot
back to its lane's centreline; the controller in examples/ppo_baseline does the same at test time.

Reward: the sum, every step, of the terms in REWARD_SCALES times their weights (see reward_terms for each one):
follow the commanded velocity, make progress along the track, keep the torso level and at its standing height,
take proper steps, and avoid falls, jerky actions and wasted torque. Change the weights to change the walk.
"""
from __future__ import annotations

import functools
import time
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math

from mujoco_playground._src import mjx_env

from sprout_challenge import track as tracks

ASSETS = Path(tracks.__file__).resolve().parent / "assets"
LEGS = ("back_right", "back_left", "front_left", "front_right")          # joint / actuator order
LANE_W = 6.0                                                             # m, the competition's track width
FOOTPRINT = jp.array([[x, y] for x in (-0.45, 0.0, 0.45) for y in (-0.22, 0.0, 0.22)])   # torso box, body frame
SPAWN_CORNERS = jp.array([[0, 0], [0.6, 0.4], [0.6, -0.4], [-0.6, 0.4], [-0.6, -0.4]])

REWARD_SCALES = dict(
    tracking_lin_vel=1.0,       # + match the commanded forward and sideways speed
    tracking_ang_vel=0.5,       # + match the commanded turn rate
    progress=1.0,               # + speed along the track (+x), clipped to [-0.5, 1] m/s
    pose=0.5,                   # + joints near the standing pose
    feet_air_time=0.2,          # + steps that leave the ground long enough
    lin_vel_z=-0.5,             # - bouncing
    ang_vel_xy=-0.05,           # - rocking
    orientation=-5.0,           # - body tilt
    base_height=-20.0,          # - torso height away from 0.85 m above the terrain
    dof_pos_limits=-1.0,        # - joints near their limits
    termination=-1.0,           # - falling
    collision=-1.0,             # - shins or upper legs touching the ground
    self_collision=-1.0,        # - legs touching the body
    torques=-2e-5,              # - motor effort
    action_rate=-0.01,          # - jerky actions
    energy=-1e-4,               # - mechanical power
    feet_clearance=-2.0,        # - swing feet away from the target lift while moving
    feet_height=-0.2,           # - each step's peak lift away from the target
    feet_slip=-0.1,             # - feet sliding while on the ground
    air_time_limit=-0.5,        # - a foot in the air longer than 0.5 s
)


# ---------------------------------------------------------------------------------------------------- terrain

def practice_without_barriers() -> tracks.Track:
    """The practice track with each barrier section replaced by flat ground of the same length (the policy walks
    around barriers in the open lane, which is flat)."""
    t = tracks.Track.load("practice")
    secs = []
    for s, x0 in zip(t.sections, t.starts):
        if s["type"] == "barrier":
            L = tracks.SECTIONS["barrier"][1](s)
            secs.append({"type": "flat", "length": L})
        else:
            secs.append(s)
    return tracks.Track("practice_flat_barrier", secs, t.width)


def training_tracks(n_lanes: int = 8, seed: int = 0) -> list[tracks.Track]:
    """Lane 0: the practice track; the others: generated tracks from easy to hard."""
    out = [practice_without_barriers()]
    for i in range(1, n_lanes):
        d = 0.3 + 0.7 * (i - 1) / max(n_lanes - 2, 1)
        out.append(tracks.generate(seed * 1000 + i, difficulty=d))
    return out


def build_field(lanes: list[tracks.Track]):
    """Stack the lanes across y into one heightfield. Returns (heights (nrow, ncol), half sizes, lane centres,
    lane finish x, narrow mask (n_lanes, ncol): True where the lane is a narrow bridge)."""
    length = max(t.length for t in lanes) + 2.0
    ncol = int(round(length / tracks.RES)) + 1
    per = int(round(LANE_W / tracks.RES))                     # rows per lane (shared edge rows overwrite)
    nrow = per * len(lanes) + 1
    H = np.zeros((nrow, ncol))
    narrow = np.zeros((len(lanes), ncol), bool)
    xs = np.linspace(0, length, ncol)
    for i, t in enumerate(lanes):
        h = t.heights()                                        # (per + 1, t_ncol), x from 0
        H[i * per:(i + 1) * per + 1, :h.shape[1]] = h
        for s, x0 in zip(t.sections, t.starts):
            if s["type"] == "bridge":
                narrow[i] |= (xs >= x0) & (xs < x0 + tracks.SECTIONS["bridge"][1](s))
    centres = (np.arange(len(lanes)) - (len(lanes) - 1) / 2) * LANE_W
    finish = np.array([t.finish_x for t in lanes])
    return H, (length / 2, nrow * tracks.RES / 2 - tracks.RES / 2), centres, finish, narrow


def scene_xml(nrow: int, ncol: int, half_x: float, half_y: float, hmin: float, hrange: float) -> str:
    robot = (ASSETS / "robot.xml").read_text()
    robot = robot.replace('meshdir="meshes"', f'meshdir="{ASSETS / "meshes"}"')
    robot = robot[:robot.rindex("</mujoco>")]
    return robot + f'''
  <asset>
    <hfield name="track" nrow="{nrow}" ncol="{ncol}" size="{half_x} {half_y} {hrange} {tracks.BASE}"/>
  </asset>
  <worldbody>
    <geom name="floor" type="hfield" hfield="track" pos="{half_x} 0 {hmin}" contype="1" conaffinity="0"
      rgba="0.3 0.4 0.5 1"/>
  </worldbody>
  <!-- The torso box does not collide with the heightfield (it covers too many cells for the GPU contact
       budget); a torso this close to the ground already counts as a fall. -->
  <contact>
    <exclude body1="world" body2="root"/>
  </contact>
  <include file="{ASSETS / "contacts.xml"}"/>
  <include file="{ASSETS / "keyframes.xml"}"/>
</mujoco>
'''


# ---------------------------------------------------------------------------------------------------- config

def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        episode_length=1000,             # 20 s
        action_scale=0.5,
        soft_joint_pos_limit_factor=0.95,
        impl="warp",                     # "warp" (MuJoCo Warp: ~40x faster on heightfields) or "jax" (pure MJX)
        naconmax=16 * 8192,              # contact and constraint budget (MuJoCo Warp only)
        njmax=150,
        n_lanes=8,
        track_seed=0,
        noise_level=1.0,
        noise=config_dict.create(joint_pos=0.03, joint_vel=1.5, gyro=0.2, gravity=0.05, linvel=0.1, scan=0.02),
        reward=config_dict.create(
            scales=config_dict.create(**REWARD_SCALES),
            tracking_sigma=0.25,
            base_height_target=0.85,
            max_foot_height=0.12,
            air_time_min=0.25,
            air_time_max=0.5,
        ),
        course=config_dict.create(
            speed=[0.4, 0.8],    # m/s: each episode's forward speed command
            steer_y=0.5, max_vy=0.2, lean=0.4, max_lean=0.3, steer_yaw=1.5, max_yaw=0.6,
            spawn_y=0.3,         # m +- from the lane centre (0.05 on a bridge)
            spawn_yaw=0.3,       # rad +-
            off_course=2.7,      # m from the lane centre ends the episode (the track edge is at 3 m)
        ),
        fall_height=0.3,
    )


# ---------------------------------------------------------------------------------------------------- env

class WalkEnv(mjx_env.MjxEnv):
    """Sprout walking along generated tracks, with a steering command; thousands in parallel on the GPU."""

    def __init__(self, config: config_dict.ConfigDict | None = None, config_overrides=None):
        super().__init__(config or default_config(), config_overrides)
        c = self._config
        lanes = training_tracks(c.n_lanes, c.track_seed)
        H, (half_x, half_y), centres, finish, narrow = build_field(lanes)
        hmin = min(float(H.min()), 0.0)
        hrange = max(float(H.max()) - hmin, 0.01)
        nrow, ncol = H.shape
        m = mujoco.MjModel.from_xml_string(scene_xml(nrow, ncol, half_x, half_y, hmin, hrange))
        m.hfield_data[:] = ((H - hmin) / hrange).ravel()
        m.opt.timestep = c.sim_dt
        self._mj_model = m
        graph_mode = None
        if c.impl == "warp":
            # No CUDA-graph capture: Warp-mode capture fails to allocate heightfield contact buffers in the
            # second compiled function that steps the physics (PPO compiles several).
            import warp as wp
            graph_mode = wp.JaxCallableGraphMode.NONE
        self._mjx_model = mjx.put_model(m, impl=c.impl, graph_mode=graph_mode)
        self.lanes = lanes

        self._H = jp.array(H)
        self._half = (half_x, half_y)
        self._shape = H.shape
        self._lane_y = jp.array(centres)
        self._finish = jp.array(finish)
        self._narrow = jp.array(narrow)
        self._torso = m.body("root").id
        self._imu = m.site("imu").id
        self._floor = m.geom("floor").id
        self._feet = np.array([m.site(f"{l}_foot").id for l in LEGS])
        self._init_q = jp.array(m.keyframe("home").qpos)
        self._pose = jp.array(m.keyframe("home").qpos[7:])
        lo, hi = m.jnt_range[1:].T
        self._soft_lo, self._soft_hi = jp.array(lo) * c.soft_joint_pos_limit_factor, jp.array(hi) * c.soft_joint_pos_limit_factor

        def adr(name):
            return int(m.sensor_adr[m.sensor(name).id])
        self._foot_adr = np.array([adr(f"{l}_foot_floor_found") for l in LEGS])
        self._limb_adr = np.array([adr(f"{l}_{p}_floor_found") for l in LEGS for p in ("shin", "leg")])
        self._self_adr = np.array([adr(f"{l}_{p}_torso_found") for l in LEGS for p in ("leg", "shin")])
        self._foot_vel_adr = np.array([[adr(f"{l}_foot_global_linvel") + k for k in range(3)] for l in LEGS])

        from sprout_challenge.sim import SCAN_POINTS        # the competition's 13 x 7 scan grid, body frame
        self._scan_pts = jp.array(SCAN_POINTS.reshape(-1, 2))                               # (91, 2)

    # -------------------------------------------------------------------------------------------- terrain

    def ground_height(self, xy):
        """Terrain height [m] under each point (..., 2); nearest heightfield cell."""
        nrow, ncol = self._shape
        hx, hy = self._half
        col = jp.clip(jp.round(xy[..., 0] / (2 * hx) * (ncol - 1)), 0, ncol - 1).astype(jp.int32)
        row = jp.clip(jp.round((xy[..., 1] / hy + 1) * 0.5 * (nrow - 1)), 0, nrow - 1).astype(jp.int32)
        return self._H[row, col]

    def _heading(self, data):
        fwd = data.xmat[self._torso][:2, 0]
        fwd = fwd / jp.maximum(jp.linalg.norm(fwd), 1e-6)
        return fwd, jp.arctan2(fwd[1], fwd[0])

    def _to_world(self, data, pts):
        fwd, _ = self._heading(data)
        left = jp.array([-fwd[1], fwd[0]])
        return data.qpos[0:2] + pts[:, :1] * fwd + pts[:, 1:2] * left

    def base_height(self, data):
        """Torso height above the mean terrain under its footprint."""
        return data.qpos[2] - jp.mean(self.ground_height(self._to_world(data, FOOTPRINT)))

    def scan(self, data):
        h = self.ground_height(self._to_world(data, self._scan_pts))
        return jp.clip(data.qpos[2] - h - self._config.reward.base_height_target, -1.0, 1.0)

    def steer(self, data, speed, lane_y):
        """The command: the episode's forward speed, plus a pull back to the lane centre and to facing +x."""
        cc = self._config.course
        y = data.qpos[1] - lane_y
        _, yaw = self._heading(data)
        want = jp.clip(-cc.lean * y, -cc.max_lean, cc.max_lean)
        vy = jp.clip(-cc.steer_y * y, -cc.max_vy, cc.max_vy) * jp.cos(yaw)
        wz = jp.clip(-cc.steer_yaw * (yaw - want), -cc.max_yaw, cc.max_yaw)
        return jp.array([speed, vy, wz])

    # -------------------------------------------------------------------------------------------- sensors

    def _sensor(self, data, name):
        return mjx_env.get_sensor_data(self.mj_model, data, name)

    def _gravity(self, data):
        return data.site_xmat[self._imu].T @ jp.array([0.0, 0.0, -1.0])

    def _foot_contact(self, data):
        return data.sensordata[self._foot_adr] > 0

    # -------------------------------------------------------------------------------------------- reset / step

    def reset(self, rng):
        c, cc = self._config, self._config.course
        rng, k_lane, k_x, k_y, k_yaw, k_speed, k_v = jax.random.split(rng, 7)
        lane = jax.random.randint(k_lane, (), 0, len(self.lanes))
        lane_y = self._lane_y[lane]
        x = jax.random.uniform(k_x, minval=tracks.START_X, maxval=self._finish[lane] - 2.0)
        col = jp.clip(jp.round(x / tracks.RES).astype(jp.int32), 0, self._narrow.shape[1] - 1)
        on_bridge = jp.any(self._narrow[lane, jp.clip(col + jp.arange(-20, 21), 0, self._narrow.shape[1] - 1)])
        y = lane_y + jax.random.uniform(k_y, minval=-1.0, maxval=1.0) * jp.where(on_bridge, 0.05, cc.spawn_y)
        yaw = jax.random.uniform(k_yaw, minval=-cc.spawn_yaw, maxval=cc.spawn_yaw)
        xy = jp.array([x, y])
        ground = jp.max(self.ground_height(xy + SPAWN_CORNERS))
        qpos = self._init_q.at[0:2].set(xy).at[2].add(ground)
        qpos = qpos.at[3:7].set(math.quat_mul(qpos[3:7], math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), yaw)))
        qvel = jp.zeros(self.mjx_model.nv).at[0:6].set(jax.random.uniform(k_v, (6,), minval=-0.3, maxval=0.3))
        data = mjx_env.make_data(self.mj_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:], impl=self.mjx_model.impl.value,
                                 naconmax=c.naconmax, njmax=c.njmax)
        data = mjx.forward(self.mjx_model, data)

        speed = jax.random.uniform(k_speed, minval=cc.speed[0], maxval=cc.speed[1])
        info = {
            "rng": rng,
            "lane": lane,
            "lane_y": lane_y,
            "speed": speed,
            "command": self.steer(data, speed, lane_y),
            "last_act": jp.zeros(12),
            "feet_air_time": jp.zeros(4),
            "last_contact": jp.zeros(4, dtype=bool),
            "swing_peak": jp.zeros(4),
        }
        metrics = {f"reward/{k}": jp.zeros(()) for k in c.reward.scales.keys()}
        metrics.update({"distance": jp.zeros(()), "finished": jp.zeros(()), "off_course": jp.zeros(())})
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state, action):
        c, cc = self._config, self._config.course
        info = state.info
        info["command"] = self.steer(state.data, info["speed"], info["lane_y"])
        x0 = state.data.qpos[0]
        data = mjx_env.step(self.mjx_model, state.data, self._pose + action * c.action_scale, self.n_substeps)

        contact = self._foot_contact(data)
        contact_filt = contact | info["last_contact"]
        first_contact = (info["feet_air_time"] > 0.0) * contact_filt
        info["feet_air_time"] += self.dt
        foot_pos = data.site_xpos[self._feet]
        foot_clear = foot_pos[:, 2] - self.ground_height(foot_pos[:, :2])
        info["swing_peak"] = jp.maximum(info["swing_peak"], foot_clear)

        base_h = self.base_height(data)
        x1, y1 = data.qpos[0], data.qpos[1]
        fell = (base_h < c.fall_height) | (self._sensor(data, "upvector")[-1] < 0.5)
        finished = x1 > self._finish[info["lane"]]
        off = jp.abs(y1 - info["lane_y"]) > cc.off_course
        done = fell | finished | off

        terms = self.reward_terms(data, action, info, fell, first_contact, contact, foot_clear, base_h, x0)
        terms = {k: v * c.reward.scales[k] for k, v in terms.items()}
        reward = jp.clip(sum(terms.values()) * self.dt, 0.0, 10000.0)

        info["last_act"] = action
        info["feet_air_time"] *= ~contact
        info["last_contact"] = contact
        info["swing_peak"] *= ~contact
        obs = self._get_obs(data, info)
        for k, v in terms.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["distance"] = x1 - x0              # summed over an episode: metres walked along the track
        state.metrics["finished"] = finished.astype(jp.float32)
        state.metrics["off_course"] = off.astype(jp.float32)
        return state.replace(data=data, obs=obs, reward=reward, done=done.astype(jp.float32), info=info)

    # -------------------------------------------------------------------------------------------- obs / reward

    def _noisy(self, info, x, scale):
        info["rng"], k = jax.random.split(info["rng"])
        return x + (2 * jax.random.uniform(k, x.shape) - 1) * self._config.noise_level * scale

    def _get_obs(self, data, info):
        ns = self._config.noise
        gyro, gravity = self._sensor(data, "gyro"), self._gravity(data)
        linvel = self._sensor(data, "local_linvel")
        q, qd = data.qpos[7:], data.qvel[6:]
        scan = self.scan(data)
        state = jp.hstack([
            self._noisy(info, linvel, ns.linvel),                  # 3
            self._noisy(info, gyro, ns.gyro),                      # 3
            self._noisy(info, gravity, ns.gravity),                # 3
            self._noisy(info, q, ns.joint_pos) - self._pose,       # 12
            self._noisy(info, qd, ns.joint_vel),                   # 12
            info["last_act"],                                      # 12
            info["command"],                                       # 3
            self._noisy(info, scan, ns.scan),                      # 91
        ])
        privileged = jp.hstack([                                   # the critic's extra view (training only)
            state, gyro, self._sensor(data, "accelerometer"), gravity, linvel, self._sensor(data, "global_angvel"),
            q - self._pose, qd, data.actuator_force, info["last_contact"], data.sensordata[self._foot_vel_adr].ravel(),
            info["feet_air_time"], self.base_height(data), scan,
        ])
        return {"state": state, "privileged_state": privileged}

    def reward_terms(self, data, action, info, fell, first_contact, contact, foot_clear, base_h, x0):
        """Each term before its weight (REWARD_SCALES); the step's reward is the weighted sum times dt."""
        rc = self._config.reward
        cmd = info["command"]
        local_vel, gyro = self._sensor(data, "local_linvel"), self._sensor(data, "gyro")
        up = self._sensor(data, "upvector")
        q, qd, tau = data.qpos[7:], data.qvel[6:], data.actuator_force
        feet_vxy = data.sensordata[self._foot_vel_adr][:, :2]
        out_of_limits = -jp.clip(q - self._soft_lo, None, 0.0) + jp.clip(q - self._soft_hi, 0.0, None)
        return {
            "tracking_lin_vel": jp.exp(-jp.sum(jp.square(cmd[:2] - local_vel[:2])) / rc.tracking_sigma),
            "tracking_ang_vel": jp.exp(-jp.square(cmd[2] - gyro[2]) / rc.tracking_sigma),
            "progress": jp.clip((data.qpos[0] - x0) / self.dt, -0.5, 1.0),
            "pose": jp.exp(-jp.sum(jp.square(q - self._pose) * jp.array([1.0, 1.0, 0.1] * 4))),
            "feet_air_time": jp.sum((info["feet_air_time"] - rc.air_time_min) * first_contact),
            "lin_vel_z": jp.square(self._sensor(data, "global_linvel")[2]),
            "ang_vel_xy": jp.sum(jp.square(self._sensor(data, "global_angvel")[:2])),
            "orientation": jp.sum(jp.square(up[:2])),
            "base_height": jp.square(base_h - rc.base_height_target),
            "dof_pos_limits": jp.sum(out_of_limits),
            "termination": fell.astype(jp.float32),
            "collision": jp.sum(data.sensordata[self._limb_adr] > 0).astype(jp.float32),
            "self_collision": jp.sum(data.sensordata[self._self_adr] > 0).astype(jp.float32),
            "torques": jp.sqrt(jp.sum(jp.square(tau))) + jp.sum(jp.abs(tau)),
            "action_rate": jp.sum(jp.square(action - info["last_act"])),
            "energy": jp.sum(jp.abs(qd) * jp.abs(tau)),
            "feet_clearance": jp.sum(jp.abs(foot_clear - rc.max_foot_height) * jp.sqrt(jp.linalg.norm(feet_vxy, axis=-1))),
            "feet_height": jp.sum(jp.square(info["swing_peak"] / rc.max_foot_height - 1.0) * first_contact),
            "feet_slip": jp.sum(jp.sum(jp.square(feet_vxy), axis=-1) * contact),
            "air_time_limit": jp.sum(info["feet_air_time"] > rc.air_time_max).astype(jp.float32),
        }

    # -------------------------------------------------------------------------------------------- accessors

    @property
    def xml_path(self) -> str:
        return ""

    @property
    def action_size(self) -> int:
        return 12

    @property
    def mj_model(self):
        return self._mj_model

    @property
    def mjx_model(self):
        return self._mjx_model


# ---------------------------------------------------------------------------------------------------- randomisation

def domain_randomize(model, rng, floor_geom_id: int, torso_body_id: int):
    """Per-robot physics variation, fixed for a robot's lifetime: ground friction, joint friction and armature,
    torso centre of mass, link masses, a payload, joint zero offsets and motor strength. The grader runs the
    nominal robot; training on variations makes the policy robust to small modelling differences."""
    @jax.vmap
    def rand(rng):
        ks = jax.random.split(rng, 8)
        friction = model.geom_friction.at[floor_geom_id, 0].set(jax.random.uniform(ks[0], minval=0.4, maxval=1.0))
        frictionloss = model.dof_frictionloss.at[6:].multiply(jax.random.uniform(ks[1], (12,), minval=0.9, maxval=1.1))
        armature = model.dof_armature.at[6:].multiply(jax.random.uniform(ks[2], (12,), minval=1.0, maxval=1.05))
        ipos = model.body_ipos.at[torso_body_id].add(jax.random.uniform(ks[3], (3,), minval=-0.05, maxval=0.05))
        mass = model.body_mass * jax.random.uniform(ks[4], (model.nbody,), minval=0.9, maxval=1.1)
        mass = mass.at[torso_body_id].add(jax.random.uniform(ks[5], minval=-2.0, maxval=2.0))
        qpos0 = model.qpos0.at[7:].add(jax.random.uniform(ks[6], (12,), minval=-0.05, maxval=0.05))
        k = jax.random.uniform(ks[7], minval=0.93, maxval=1.07)
        gain = model.actuator_gainprm.at[:, 0].multiply(k)
        bias = model.actuator_biasprm.at[:, 1:3].multiply(k)
        force = model.actuator_forcerange * k
        return friction, frictionloss, armature, ipos, mass, qpos0, gain, bias, force

    out = rand(rng)
    names = ("geom_friction", "dof_frictionloss", "dof_armature", "body_ipos", "body_mass", "qpos0",
             "actuator_gainprm", "actuator_biasprm", "actuator_forcerange")
    fields = dict(zip(names, out))
    in_axes = jax.tree_util.tree_map(lambda x: None, model).tree_replace({k: 0 for k in fields})
    return model.tree_replace(fields), in_axes


# ---------------------------------------------------------------------------------------------------- training

def ppo_config(num_timesteps=120_000_000, num_envs=8192):
    """Brax PPO settings used for examples/ppo_baseline (asymmetric actor-critic: the critic gets privileged_state)."""
    return dict(
        num_timesteps=int(num_timesteps), num_evals=20, reward_scaling=1.0, episode_length=1000,
        normalize_observations=True, action_repeat=1, unroll_length=20, num_minibatches=32, num_updates_per_batch=4,
        discounting=0.97, learning_rate=3e-4, entropy_cost=1e-2, num_envs=int(num_envs), batch_size=256,
        max_grad_norm=1.0, num_resets_per_eval=10, num_eval_envs=128,
        # Progress comes from evaluations (deterministic policy, fresh robots), 20 per run: with Playground's default
        # auto-reset, Brax's statistics of the training episodes are not meaningful.
        run_evals=True, log_training_metrics=False,
    )


NETWORK = dict(policy_hidden_layer_sizes=(512, 256, 128), value_hidden_layer_sizes=(512, 256, 128),
               policy_obs_key="state", value_obs_key="privileged_state")


def _patch_jax():
    """Brax 0.14's PPO calls jax.device_put_replicated, which newer JAX removed; put back a version of it."""
    if "device_put_replicated" in jax.__dict__:
        return

    def device_put_replicated(x, devices):
        devices = list(devices)

        def rep(a):
            shards = [jax.device_put(jp.asarray(a)[None], d) for d in devices]
            if len(devices) == 1:
                return shards[0]
            mesh = jax.sharding.Mesh(devices, ("d",))
            spec = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("d"))
            return jax.make_array_from_single_device_arrays((len(devices),) + jp.shape(a), spec, shards)
        return jax.tree_util.tree_map(rep, x)
    jax.device_put_replicated = device_put_replicated


def train(env: WalkEnv, num_timesteps=120_000_000, num_envs=8192, seed=0, checkpoint_dir=None, restore_dir=None,
          progress=None):
    """Train with Brax PPO. progress(step, row) is called with each training report (row: dict of floats,
    one per evaluation: 'reward', 'seconds' (how long robots last, max 20), 'distance' (m walked), 'finished'
    (share reaching the finish), 'minutes', 'sps'). Returns the policy params (normaliser, policy MLP, value MLP).
    checkpoint_dir: save a checkpoint at every report (restore_dir resumes from one)."""
    _patch_jax()
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    t0 = time.time()

    def on_progress(step, metrics):
        """One row per evaluation: 128 fresh robots, deterministic policy, up to 20 s each."""
        if "eval/episode_reward" not in metrics:
            return
        row = {"minutes": (time.time() - t0) / 60, "reward": float(metrics["eval/episode_reward"]),
               "seconds": float(metrics.get("eval/avg_episode_length", np.nan)) * env.dt,   # how long they last
               "distance": float(metrics.get("eval/episode_distance", np.nan)),             # m walked
               "finished": float(metrics.get("eval/episode_finished", np.nan)),             # share that finish
               "sps": float(metrics.get("training/sps", np.nan))}
        if progress:
            progress(int(step), row)
        else:
            print(f"[{row['minutes']:6.1f} min] step {int(step):>12,}  reward {row['reward']:7.2f}  "
                  f"lasts {row['seconds']:4.1f} s  walks {row['distance']:5.1f} m", flush=True)

    rand = functools.partial(domain_randomize, floor_geom_id=env._floor, torso_body_id=env._torso)
    make_inference, params, _ = ppo.train(
        environment=env, eval_env=WalkEnv(env._config), **ppo_config(num_timesteps, num_envs),
        network_factory=functools.partial(ppo_networks.make_ppo_networks, **NETWORK), seed=seed,
        randomization_fn=rand, progress_fn=on_progress,
        save_checkpoint_path=checkpoint_dir, restore_checkpoint_path=restore_dir,
        # Playground's default auto-reset (restore each robot's first start state) as for the baseline; with
        # full_reset=True Brax stops reporting the training episodes' statistics.
        wrap_env_fn=wrapper.wrap_for_brax_training)
    return params


# ---------------------------------------------------------------------------------------------------- export

def numpy_policy(z, obs):
    """The trained policy in plain numpy: normalise, MLP with swish, tanh of the first 12 outputs (the means)."""
    x = (np.asarray(obs, np.float64) - z["mean"]) / z["std"]
    n = len([k for k in z if k.startswith("W")])
    for i in range(n):
        x = x @ z[f"W{i}"] + z[f"b{i}"]
        if i < n - 1:
            x = x / (1.0 + np.exp(-x))
    return np.tanh(x[..., :12])


def export(params, path):
    """Save the policy as numpy arrays in the format examples/ppo_baseline/controller.py loads (policy.npz)."""
    norm, pol = params[0], params[1]
    mean, std = norm.mean, norm.std
    if isinstance(mean, dict):
        mean, std = mean["state"], std["state"]
    layers = pol["params"]
    z = {"mean": np.asarray(mean, np.float64), "std": np.asarray(std, np.float64)}
    for i in range(len(layers)):
        z[f"W{i}"] = np.asarray(layers[f"hidden_{i}"]["kernel"], np.float64)
        z[f"b{i}"] = np.asarray(layers[f"hidden_{i}"]["bias"], np.float64)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **z)
    return z


def check_export(env: WalkEnv, params, z, n=64, seed=0):
    """Compare the numpy policy with Brax's on observations from fresh resets; returns the largest difference."""
    _patch_jax()
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    nets = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                          preprocess_observations_fn=running_statistics.normalize, **NETWORK)
    infer = ppo_networks.make_inference_fn(nets)(params, deterministic=True)
    states = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(seed), n))
    a_brax, _ = infer(states.obs, jax.random.PRNGKey(0))
    a_np = numpy_policy(z, np.asarray(states.obs["state"]))
    return float(np.max(np.abs(np.asarray(a_brax) - a_np)))
