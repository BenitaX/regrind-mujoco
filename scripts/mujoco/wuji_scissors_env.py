"""Single-environment Gymnasium port of REGRIND's WujiHand scissors task."""

import tempfile
from pathlib import Path

import gymnasium as gym
import h5py
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium.spaces import Box

if __package__:
    from .replay_wuji_scissors import ROOT, build_scene_xml
else:
    from replay_wuji_scissors import ROOT, build_scene_xml


TRAJECTORY = ROOT / "data/retargeted_traj/wujihand/scissors/retargeted_120fps.h5"
JOINT_NAMES = [f"right_finger{i}_joint{j}" for i in range(1, 6) for j in range(1, 5)]
OBS_SIZE = 131
POLICY_DT = 1.0 / 30.0


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


class WujiScissorsEnv(gym.Env):
    """Actions are 6 wrist deltas plus 20 relative finger targets; no Isaac dependency."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, trajectory_path=TRAJECTORY, render_mode=None, frame_skip=4):
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        if frame_skip < 1:
            raise ValueError("frame_skip must be positive")
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        with h5py.File(trajectory_path, "r") as h5:
            raw = {key: h5[key][:] for key in (
                "robot_pos", "robot_quat", "robot_joints", "object_pos", "object_quat", "object_joint"
            )}
        # Isaac uses 30 Hz policy steps with a 120 Hz demo and four physics steps per action.
        self.trajectory = {key: value[::frame_skip].astype(np.float64) for key, value in raw.items()}
        if any(len(value) != len(self.trajectory["robot_pos"]) for value in self.trajectory.values()):
            raise ValueError("Trajectory datasets have inconsistent lengths")
        if self.trajectory["robot_joints"].shape[1] != 20:
            raise ValueError("Expected 20 WujiHand finger joints")
        self.length = len(self.trajectory["robot_pos"])
        if self.length < 2:
            raise ValueError("Trajectory must contain at least two policy frames")
        self.dt = POLICY_DT

        with tempfile.TemporaryDirectory() as temp_dir:
            xml_path = Path(temp_dir) / "scene.xml"
            xml_path.write_text(build_scene_xml(), encoding="utf-8")
            self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.model.opt.timestep = 1.0 / 120.0
        self.data = mujoco.MjData(self.model)

        self.robot_body = _id(self.model, mujoco.mjtObj.mjOBJ_BODY, "root")
        self.object_body = _id(self.model, mujoco.mjtObj.mjOBJ_BODY, "scissors")
        self.object_top_body = _id(self.model, mujoco.mjtObj.mjOBJ_BODY, "scissors_top")
        self.wrist_joint = _id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "wrist_free")
        self.object_free_joint = _id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "scissors_free")
        self.object_hinge = _id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "scissors_joint")
        self.wrist_qpos = self.model.jnt_qposadr[self.wrist_joint]
        self.object_qpos = self.model.jnt_qposadr[self.object_free_joint]
        self.hinge_qpos = self.model.jnt_qposadr[self.object_hinge]
        self.finger_qpos = np.array([
            self.model.jnt_qposadr[_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in JOINT_NAMES
        ])
        self.finger_dof = np.array([
            self.model.jnt_dofadr[_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)] for name in JOINT_NAMES
        ])
        self.actuators = np.array([
            _id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_actuator") for name in JOINT_NAMES
        ])
        self.bottom_points = np.load(ROOT / "data/keypoints/scissors_bottom_25.npy").astype(np.float64)
        self.top_points = np.load(ROOT / "data/keypoints/scissors_top_25.npy").astype(np.float64)
        self.actions = np.zeros(26, dtype=np.float64)
        self.frame = 0
        self.viewer = None
        self.action_space = Box(-1.0, 1.0, shape=(26,), dtype=np.float32)
        self.observation_space = Box(-np.inf, np.inf, shape=(OBS_SIZE,), dtype=np.float32)

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
            self.data.xpos[self.object_body], _rot6(object_q), [self.data.qpos[self.hinge_qpos]],
            self.data.xpos[self.robot_body], _rot6(root_q), self.data.qpos[self.finger_qpos],
            object_velocity, self.data.qvel[self.finger_dof],
            self.trajectory["object_pos"][t], _rot6(self.trajectory["object_quat"][t]),
            np.asarray(self.trajectory["object_joint"][t]).reshape(-1)[:1],
            self.trajectory["robot_pos"][t], _rot6(self.trajectory["robot_quat"][t]),
            self.trajectory["robot_joints"][t], self.actions, [t / (self.length - 1)],
        ]).astype(np.float32)
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.frame = int(options["start_index"]) if options and "start_index" in options else int(self.np_random.integers(self.length - 1))
        self.actions.fill(0)
        self.previous_action = self.actions.copy()
        self.data.qpos[self.wrist_qpos:self.wrist_qpos + 7] = np.r_[
            self.trajectory["robot_pos"][self.frame], self.trajectory["robot_quat"][self.frame]
        ]
        self.data.qpos[self.object_qpos:self.object_qpos + 7] = np.r_[
            self.trajectory["object_pos"][self.frame], self.trajectory["object_quat"][self.frame]
        ]
        self.data.qpos[self.hinge_qpos] = float(np.asarray(self.trajectory["object_joint"][self.frame]).reshape(-1)[0])
        self.data.qpos[self.finger_qpos] = self.trajectory["robot_joints"][self.frame]
        self.data.ctrl[self.actuators] = self.trajectory["robot_joints"][self.frame]
        mujoco.mj_forward(self.model, self.data)
        if self.render_mode == "human":
            self.render()
        return self._observation(), {"frame": self.frame}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (26,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be a finite vector of shape (26,)")
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

        previous_action = self.actions.copy()
        self.actions = action
        self.frame = min(self.frame + 1, self.length - 1)
        target_points = self._target_points(self.frame)
        current_points = np.concatenate([
            self._pose_points(self.object_body, self.bottom_points),
            self._pose_points(self.object_top_body, self.top_points),
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
            1.5 * np.exp(-keypoint_error / 0.015)
            + np.exp(-lin_error)
            + np.exp(-ang_error / (3.14 ** 2))
            + 0.05 * np.exp(-wrist_pos_error / 0.02)
            + 0.05 * np.exp(-wrist_rot_error / 0.2)
            + 0.5 * np.exp(-np.mean(action ** 2))
            + np.exp(-action_rate / (0.5 ** 2))
        )
        object_pos = self.data.xpos[self.object_body]
        failure_reasons = []
        if keypoint_error > 0.15:
            failure_reasons.append("object_deviation")
        if np.any(object_pos < (-0.25, -0.25, 0.95)) or np.any(object_pos > (0.25, 0.25, 1.5)):
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
        return self._observation(), float(reward), bool(terminated), False, info

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
