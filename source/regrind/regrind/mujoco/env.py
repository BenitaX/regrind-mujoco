"""MuJoCo trajectory-tracking tasks for WujiHand and LeapHand."""

import tempfile
import os
from pathlib import Path

import gymnasium as gym
import h5py
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium.spaces import Box

from .scene import build_scene_xml, WUJI_JOINTS, LEAP_JOINTS

ROOT = Path(__file__).resolve().parents[4]
DATA = Path(os.environ.get("REGRIND_DATA_DIR", ROOT / "data"))


def _id(model, kind, name):
    value = mujoco.mj_name2id(model, kind, name)
    if value < 0:
        raise ValueError(f"MuJoCo model is missing {name!r}")
    return value


def _rot6(quat):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3)[:, :2].reshape(-1)


def _quat_error(target, actual):
    local_error = np.empty(3)
    mujoco.mju_subQuat(local_error, target, actual)
    world_error = np.empty(3)
    mujoco.mju_rotVecQuat(world_error, local_error, actual)
    return world_error


def _quat_from_rotvec(rotvec):
    angle = np.linalg.norm(rotvec)
    if angle < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.r_[np.cos(angle / 2), rotvec * (np.sin(angle / 2) / angle)]


def _rotate(quat, vector):
    result = np.empty(3)
    mujoco.mju_rotVecQuat(result, vector, quat)
    return result


class RegrindEnv(gym.Env):
    """Actions are six wrist deltas plus robot-specific relative finger targets."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, robot="wujihand", object_name="scissors", play=False,
                 trajectory_path=None, render_mode=None, frame_skip=4,
                 randomization=True, observation_noise=True, observation_delay=2,
                 gravity=None, max_episode_steps=270):
        if robot not in ("wujihand", "leaphand") or object_name not in ("scissors", "screwdriver"):
            raise ValueError("Expected wujihand/leaphand and scissors/screwdriver")
        if gravity is not None and (not np.isfinite(gravity) or gravity < 0):
            raise ValueError("gravity must be finite and nonnegative")
        self.robot, self.object_name, self.play = robot, object_name, play
        self.joint_names = WUJI_JOINTS if robot == "wujihand" else LEAP_JOINTS
        self.nj = len(self.joint_names)
        self.articulated = object_name == "scissors"
        self.randomization = randomization and not play
        self.observation_noise = observation_noise and not play
        if not isinstance(observation_delay, int) or observation_delay < 0:
            raise ValueError("observation_delay must be a nonnegative integer")
        self.observation_delay = observation_delay if not play else 0
        self.gravity = gravity
        self.total_steps = 0
        self.max_episode_steps = max_episode_steps
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        suffix = ("_2x" if self.articulated else "_1.5x") if robot == "leaphand" else ""
        trajectory_path = trajectory_path or DATA / "retargeted_traj" / robot / (object_name + suffix) / "retargeted_120fps.h5"
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        if not isinstance(frame_skip, int) or frame_skip < 1:
            raise ValueError("frame_skip must be positive")
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        with h5py.File(trajectory_path, "r") as h5:
            keys = ["robot_pos", "robot_quat", "robot_joints", "object_pos", "object_quat"]
            if self.articulated:
                keys.append("object_joint")
            raw = {key: h5[key][:] for key in keys}
        for key, value in raw.items():
            if not np.all(np.isfinite(value)):
                raise ValueError(f"Non-finite trajectory values: {key}")
        # Isaac uses 30 Hz policy steps with a 120 Hz demo and four physics steps per action.
        self.trajectory = {key: value[::frame_skip].astype(np.float64) for key, value in raw.items()}
        if any(len(value) != len(self.trajectory["robot_pos"]) for value in self.trajectory.values()):
            raise ValueError("Trajectory datasets have inconsistent lengths")
        if self.trajectory["robot_joints"].shape[1] != self.nj:
            raise ValueError(f"Expected {self.nj} finger joints")
        self.base_trajectory = {k: v.copy() for k, v in self.trajectory.items()}
        self.length = len(self.trajectory["robot_pos"])
        if self.length < 2:
            raise ValueError("Trajectory must contain at least two policy frames")
        self.dt = frame_skip / 120.0

        with tempfile.TemporaryDirectory() as temp_dir:
            xml_path = Path(temp_dir) / "scene.xml"
            xml_path.write_text(build_scene_xml(robot, object_name), encoding="utf-8")
            self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.model.opt.timestep = 1.0 / 120.0
        self.data = mujoco.MjData(self.model)

        self.robot_body = _id(self.model, mujoco.mjtObj.mjOBJ_BODY, "root")
        self.object_body = _id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_name)
        self.object_top_body = _id(self.model, mujoco.mjtObj.mjOBJ_BODY, "scissors_top") if self.articulated else None
        self.wrist_joint = _id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "wrist_free")
        self.object_free_joint = _id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{object_name}_free")
        self.object_hinge = _id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "scissors_joint") if self.articulated else None
        self.wrist_qpos = self.model.jnt_qposadr[self.wrist_joint]
        self.object_qpos = self.model.jnt_qposadr[self.object_free_joint]
        self.hinge_qpos = self.model.jnt_qposadr[self.object_hinge] if self.articulated else None
        self.finger_qpos = np.array([
            self.model.jnt_qposadr[_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in self.joint_names
        ])
        self.finger_dof = np.array([
            self.model.jnt_dofadr[_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in self.joint_names
        ])
        self.actuators = np.array([
            _id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_actuator") for name in self.joint_names
        ])
        scale = (2.0 if self.articulated else 1.5) if robot == "leaphand" else 1.0
        self.bottom_points = np.load(DATA / "keypoints" / ("scissors_bottom_25.npy" if self.articulated else "screwdriver_50.npy")).astype(np.float64) * scale
        self.top_points = np.load(DATA / "keypoints/scissors_top_25.npy").astype(np.float64) * scale if self.articulated else None
        self.actions = np.zeros(6 + self.nj, dtype=np.float64)
        self.frame = 0
        self.viewer = None
        self.action_space = Box(-1.0, 1.0, shape=(6 + self.nj,), dtype=np.float32)
        self.observation_space = Box(-np.inf, np.inf, shape=(51 + 4 * self.nj - (0 if self.articulated else 2),), dtype=np.float32)
        self._friction = self.model.geom_friction.copy()
        self._mass = self.model.body_mass.copy()
        self._inertia = self.model.body_inertia.copy()
        self._history = []

        self.object_lin_vel = np.gradient(self.trajectory["object_pos"], self.dt, axis=0)
        self.object_ang_vel = self._angular_velocity(self.trajectory["object_quat"])

    def _angular_velocity(self, quats):
        result = np.zeros((len(quats), 3))
        for i in range(len(quats) - 1):
            result[i] = _quat_error(quats[i + 1], quats[i]) / self.dt
        result[-1] = result[-2]
        return result

    def _pose_points(self, body_id, points):
        pos, quat = self.data.xpos[body_id], self.data.xquat[body_id]
        return np.array([pos + _rotate(quat, point) for point in points])

    def _target_points(self, frame):
        pos, quat = self.trajectory["object_pos"][frame], self.trajectory["object_quat"][frame]
        if not self.articulated:
            return np.array([pos + _rotate(quat, point) for point in self.bottom_points])
        angle = float(np.asarray(self.trajectory["object_joint"][frame]).reshape(-1)[0])
        hinge_quat = _quat_from_rotvec(np.array([0.0, 0.0, -angle]))
        top_quat = np.empty(4)
        mujoco.mju_mulQuat(top_quat, quat, hinge_quat)
        bottom = np.array([pos + _rotate(quat, point) for point in self.bottom_points])
        top = np.array([pos + _rotate(top_quat, point) for point in self.top_points])
        return np.concatenate([bottom, top])

    def _observation(self):
        t = self.frame
        root_q = self.data.xquat[self.robot_body]
        object_q = self.data.xquat[self.object_body]
        object_velocity = np.empty(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.object_body, object_velocity, 0)
        obs = np.concatenate([
            self.data.xpos[self.object_body], _rot6(object_q), [self.data.qpos[self.hinge_qpos]] if self.articulated else [],
            self.data.xpos[self.robot_body], _rot6(root_q), self.data.qpos[self.finger_qpos],
            object_velocity, self.data.qvel[self.finger_dof],
            self.trajectory["object_pos"][t], _rot6(self.trajectory["object_quat"][t]),
            np.asarray(self.trajectory["object_joint"][t]).reshape(-1)[:1] if self.articulated else [],
            self.trajectory["robot_pos"][t], _rot6(self.trajectory["robot_quat"][t]),
            self.trajectory["robot_joints"][t], self.actions, [t / (self.length - 1)],
        ]).astype(np.float32)
        if self.observation_noise:
            obs[:3] += self.np_random.uniform(-0.002, 0.002, 3).astype(np.float32)
        self._history.append(obs.copy())
        self._history = self._history[-(self.observation_delay + 1):]
        delayed = obs.copy()
        state_end = 18 + self.nj + int(self.articulated)
        delayed[:state_end] = self._history[max(0, len(self._history) - 1 - self._lag)][:state_end]
        return delayed

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_steps = 0
        self._history = []
        self._lag = int(self.np_random.integers(self.observation_delay + 1))
        self.trajectory = {k: v.copy() for k, v in self.base_trajectory.items()}
        self.model.geom_friction[:] = self._friction
        self.model.body_mass[:] = self._mass
        self.model.body_inertia[:] = self._inertia
        if self.randomization:
            yaw = _quat_from_rotvec(np.array([0., 0., self.np_random.uniform(-np.pi / 6, np.pi / 6)]))
            offset = np.r_[self.np_random.uniform(-0.05, 0.05, 2), 0.]
            for prefix in ("robot", "object"):
                self.trajectory[f"{prefix}_pos"] = np.array([_rotate(yaw, p) + offset for p in self.trajectory[f"{prefix}_pos"]])
                for quat in self.trajectory[f"{prefix}_quat"]:
                    original = quat.copy()
                    mujoco.mju_mulQuat(quat, yaw, original)
            self.model.geom_friction[:, 0] *= self.np_random.uniform(0.7, 1.3)
            factor = self.np_random.uniform(0.8, 1.2)
            ids = [self.object_body] + ([self.object_top_body] if self.articulated else [])
            self.model.body_mass[ids] *= factor
            self.model.body_inertia[ids] *= factor
        # Curriculum step counts are policy steps per worker, matching Isaac's environment clock.
        stage = min(11, (self.total_steps - 20000) // 10000)
        ranges = [(0, 1), (0.5, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 9.81), (9, 9.81), (9.81, 9.81)]
        g = self.gravity if self.gravity is not None else (9.81 if self.play else (0 if stage < 0 else self.np_random.uniform(*ranges[stage])))
        self.model.opt.gravity[:] = [0, 0, -g]
        mujoco.mj_setConst(self.model, self.data)
        self.object_lin_vel = np.gradient(self.trajectory["object_pos"], self.dt, axis=0)
        self.object_ang_vel = self._angular_velocity(self.trajectory["object_quat"])
        mujoco.mj_resetData(self.model, self.data)
        self.frame = int(options["start_index"]) if options and "start_index" in options else (0 if self.play else int(self.np_random.integers(self.length - 1)))
        if not 0 <= self.frame < self.length - 1:
            raise ValueError("start_index must be in [0, trajectory length - 2]")
        self.actions.fill(0)
        self.previous_action = self.actions.copy()
        self.data.qpos[self.wrist_qpos:self.wrist_qpos + 7] = np.r_[
            self.trajectory["robot_pos"][self.frame], self.trajectory["robot_quat"][self.frame]
        ]
        self.data.qpos[self.object_qpos:self.object_qpos + 7] = np.r_[
            self.trajectory["object_pos"][self.frame], self.trajectory["object_quat"][self.frame]
        ]
        if self.articulated:
            self.data.qpos[self.hinge_qpos] = float(np.asarray(self.trajectory["object_joint"][self.frame]).reshape(-1)[0])
        self.data.qpos[self.finger_qpos] = self.trajectory["robot_joints"][self.frame]
        self.data.ctrl[self.actuators] = self.trajectory["robot_joints"][self.frame]
        for joint, prefix in ((self.wrist_joint, "robot"), (self.object_free_joint, "object")):
            address = self.model.jnt_dofadr[joint]
            positions = self.trajectory[f"{prefix}_pos"]
            quats = self.trajectory[f"{prefix}_quat"]
            self.data.qvel[address:address + 3] = np.gradient(positions, self.dt, axis=0)[self.frame]
            # Free-joint angular velocity is expressed in the body's local frame.
            angular = np.empty(3)
            mujoco.mju_subQuat(angular, quats[self.frame + 1], quats[self.frame])
            self.data.qvel[address + 3:address + 6] = angular / self.dt
        self.data.qvel[self.finger_dof] = np.gradient(self.trajectory["robot_joints"], self.dt, axis=0)[self.frame]
        if self.articulated:
            self.data.qvel[self.model.jnt_dofadr[self.object_hinge]] = np.asarray(np.gradient(self.trajectory["object_joint"], self.dt, axis=0)[self.frame]).item()
        mujoco.mj_forward(self.model, self.data)
        if self.render_mode == "human":
            self.render()
        return self._observation(), {"frame": self.frame}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action_space.shape or not np.all(np.isfinite(action)):
            raise ValueError(f"action must be a finite vector of shape {self.action_space.shape}")
        bounds_reward = np.exp(-np.mean((action - np.clip(action, -1.0, 1.0)) ** 2))
        self.total_steps += 1
        self.episode_steps += 1
        action = np.clip(action, -1.0, 1.0)
        target_pos = self.trajectory["robot_pos"][self.frame] + action[:3] * self.dt
        rot_delta = _quat_from_rotvec(action[3:6] * (3.2 * self.dt))
        target_quat = np.empty(4)
        mujoco.mju_mulQuat(target_quat, rot_delta, self.trajectory["robot_quat"][self.frame])
        finger_target = self.trajectory["robot_joints"][self.frame] + action[6:] * (3.2 * self.dt)
        ctrl_ranges = self.model.actuator_ctrlrange[self.actuators]
        self.data.ctrl[self.actuators] = np.clip(finger_target, ctrl_ranges[:, 0], ctrl_ranges[:, 1])

        for _ in range(self.frame_skip):
            self.data.xfrc_applied.fill(0)
            root_velocity = np.empty(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.robot_body, root_velocity, 0)
            self.data.xfrc_applied[self.robot_body, :3] = 300.0 * (target_pos - self.data.xpos[self.robot_body]) - 30.0 * root_velocity[3:]
            rot_error = _quat_error(target_quat, self.data.xquat[self.robot_body])
            self.data.xfrc_applied[self.robot_body, 3:] = 3.0 * rot_error - 0.3 * root_velocity[:3]
            mujoco.mj_step(self.model, self.data)

        mujoco.mj_forward(self.model, self.data)

        previous_action = self.actions.copy()
        self.actions = action
        self.frame = min(self.frame + 1, self.length - 1)
        target_points = self._target_points(self.frame)
        current_points = np.concatenate([
            self._pose_points(self.object_body, self.bottom_points),
            self._pose_points(self.object_top_body, self.top_points) if self.articulated else np.empty((0, 3)),
        ])
        keypoint_error = float(np.linalg.norm(current_points - target_points, axis=-1).mean())
        object_velocity = np.empty(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.object_body, object_velocity, 0)
        lin_error = np.sum((object_velocity[3:] - self.object_lin_vel[self.frame]) ** 2)
        ang_error = np.sum((object_velocity[:3] - self.object_ang_vel[self.frame]) ** 2)
        wrist_pos_error = np.linalg.norm(self.data.xpos[self.robot_body] - self.trajectory["robot_pos"][self.frame])
        wrist_rot_error = np.linalg.norm(_quat_error(self.trajectory["robot_quat"][self.frame], self.data.xquat[self.robot_body]))
        action_rate = np.mean((action - previous_action) ** 2)
        reward = (
            1.5 * np.exp(-keypoint_error / 0.02)
            + np.exp(-lin_error)
            + np.exp(-ang_error / (3.14 ** 2))
            + 0.05 * np.exp(-wrist_pos_error / 0.02)
            + 0.05 * np.exp(-wrist_rot_error / 0.2)
            + (0.5 if self.robot == "wujihand" else 0.0) * np.exp(-np.mean(action ** 2))
            + np.exp(-action_rate / (0.5 ** 2))
            + bounds_reward
        )
        object_pos = self.data.xpos[self.object_body]
        failure_reasons = []
        if keypoint_error > 0.15:
            failure_reasons.append("object_deviation")
        lower, upper = ((-0.25, -0.25, 0.95), (0.25, 0.25, 1.5)) if self.articulated else ((-0.4, -0.25, 0.5 if self.robot == "leaphand" else 0.0), (0.1, 0.25, 1.0 if self.robot == "leaphand" else 0.5))
        if np.any(object_pos < lower) or np.any(object_pos > upper):
            failure_reasons.append("object_out_of_bounds")
        if np.linalg.norm(self.data.xpos[self.robot_body] - object_pos) > 0.5:
            failure_reasons.append("hand_far_from_object")
        failed = bool(failure_reasons)
        terminated = failed or self.frame == self.length - 1
        if failed:
            reward -= 10.0
        info = {
            "frame": self.frame,
            "keypoint_error": keypoint_error,
            "wrist_pos_error": float(wrist_pos_error),
            "wrist_rot_error": float(wrist_rot_error),
            "contacts": int(self.data.ncon),
            "failure_reason": ",".join(failure_reasons),
            "success": bool(not failed and self.frame == self.length - 1),
        }
        if self.render_mode == "human":
            self.render()
        return self._observation(), float(reward * self.dt), bool(terminated), bool(self.episode_steps >= self.max_episode_steps and not terminated), info

    def set_training_steps(self, steps):
        """Restore the per-worker curriculum clock when resuming a PPO run."""
        self.total_steps = int(steps)

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
