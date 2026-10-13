"""Script to replay a retargeted trajectory."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Agent configuration entry point (required by hydra_task_config).",
)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=270, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--step_dynamics",
    action="store_true",
    default=False,
    help="Replay in open loop by stepping physics with replayed joint position targets instead of hard-setting state each timestep.",
)
parser.add_argument(
    "--replay_debug",
    action="store_true",
    default=False,
    help="Manager + --step_dynamics: log demo vs commanded vs reached finger pose (env 0) and termination flags.",
)
parser.add_argument(
    "--replay_log_interval",
    type=int,
    default=15,
    help="With --replay_debug, log every N environment steps (env 0).",
)
parser.add_argument(
    "--replay_use_rsi",
    action="store_true",
    default=False,
    help="Keep MotionCommand rsi_enabled (random demo index on reset). Default: False — disable RSI so resets align "
    "with segment starts / explicit replay sync (see Leap *_PLAY cfgs which also set rsi_enabled=False).",
)
parser.add_argument(
    "--retargeted_traj_path",
    type=str,
    default=None,
    help="Override env commands.motion.retargeted_traj_path (manager-based tasks).",
)
parser.add_argument(
    "--demo_path",
    type=str,
    default=None,
    help="Override env commands.motion.demo_path (manager-based tasks).",
)
parser.add_argument(
    "--demo_dt",
    type=float,
    default=None,
    help="Override env commands.motion.demo_dt (e.g. 0.00833333 for 120 Hz).",
)
parser.add_argument(
    "--visualize_mano",
    action="store_true",
    default=False,
    help="Visualize MANO hand skeleton overlay (blue/green joints, yellow bones).",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra (see scripts/rsl_rl/play.py)
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import imageio.v2 as imageio
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import regrind.tasks  # noqa: F401
from regrind.data.arctic import load_retargeted_traj
from regrind.data.utils import read_h5_to_dict
from regrind.utils.markers import MANO_TO_MANUS_MAPPING, ManoMarker
from regrind.utils.motion_interp import rescale_motion
from regrind.utils.math import quat_from_euler_xyz_intrinsic
from scipy.spatial.transform import Rotation as R


from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg


def _replay_never_terminate(env: ManagerBasedRLEnv, **kwargs) -> torch.Tensor:
    """Replacement termination: never fire (reference replay / debugging)."""
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg):
    """Random actions agent with Isaac Lab environment."""
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
        if args_cli.retargeted_traj_path is not None:
            env_cfg.commands.motion.retargeted_traj_path = args_cli.retargeted_traj_path
        if args_cli.demo_path is not None:
            env_cfg.commands.motion.demo_path = args_cli.demo_path
        if args_cli.demo_dt is not None:
            env_cfg.commands.motion.demo_dt = args_cli.demo_dt
    # override configurations with non-hydra CLI arguments (matches play.py behavior)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    env_cfg.normalize_action = False

    if isinstance(env_cfg, ManagerBasedRLEnvCfg) and hasattr(env_cfg.commands, "motion"):
        if not args_cli.replay_use_rsi:
            env_cfg.commands.motion.rsi_enabled = False
            print(
                "[INFO] replay: commands.motion.rsi_enabled=False (use --replay_use_rsi for training-like random starts)."
            )

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    _env = env.unwrapped
    assert isinstance(_env, ManagerBasedRLEnv), "Environment must be a ManagerBasedRLEnv"

    _REPLAY_PATCH_TERM_NAMES = (
        "time_out",
        "demo_end_reached",
        "object_deviation",
        "object_out_of_bounds",
        "object_rotation_error",
        "hand_far_from_object",
    )
    if args_cli.step_dynamics:
        # patch replay termination terms to never fire
        tm0 = _env.termination_manager
        patched = []
        for _tn in _REPLAY_PATCH_TERM_NAMES:
            if _tn not in tm0.active_terms:
                continue
            _old = tm0.get_term_cfg(_tn)
            tm0.set_term_cfg(
                _tn,
                TerminationTermCfg(
                    func=_replay_never_terminate,
                    params=dict(_old.params),
                    time_out=_old.time_out,
                ),
            )
            patched.append(_tn)
        print(f"[INFO] replay: patched termination terms (no env reset during replay): {patched}")

    motion_cmd = _env.command_manager.get_term("motion")
    object_asset = motion_cmd.object
    hand_asset = motion_cmd.robot
    hand_joint_ids = motion_cmd.actuated_dof_indices

    mano_marker = None
    if args_cli.visualize_mano:
        mano_marker = ManoMarker(prim_path="/World/Visuals/replay_mano_marker", device=_env.device)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # load retargeted trajectory
    if args_cli.retargeted_traj_path is not None:
        retargeted_traj_path = args_cli.retargeted_traj_path
    else:
        retargeted_traj_path = env_cfg.commands.motion.retargeted_traj_path

    # Reuse the exact rescaled trajectory prepared by MotionCommand.
    demo_qpos = motion_cmd.motion.demo_qpos
    demo_mano_joint_coord = motion_cmd.motion.demo_mano_joint_coord
    traj_len = demo_qpos.shape[0]
    demo_object_pos = demo_qpos[:, 0:3]
    demo_object_quat = demo_qpos[:, 3:7]
    if hasattr(object_asset, "num_joints") and object_asset.num_joints > 0:
        demo_object_joint = demo_qpos[:, 7:8]
    else:
        demo_object_joint = None
    robot_pos = demo_qpos[:, 8:11]
    robot_euler = demo_qpos[:, 11:14]
    fidx = motion_cmd.motion.demo_qpos_index["fingers"]
    robot_joints = demo_qpos[:, fidx]
    print(f"[INFO] Using manager MotionCommand trajectory with {traj_len} rescaled steps ({fidx.numel()} finger DoF).")

    demo_actions = torch.zeros((traj_len, *env.action_space.shape), dtype=torch.float32, device=_env.device)
    demo_actions[:, :, :3] = robot_pos.unsqueeze(1).expand(-1, _env.num_envs, -1) if torch.is_tensor(robot_pos) else torch.tensor(robot_pos, dtype=torch.float32, device=_env.device).unsqueeze(1).expand(-1, _env.num_envs, -1)  # (traj_len, num_envs, 3)
    demo_actions[:, :, 3:6] = robot_euler.unsqueeze(1).expand(-1, _env.num_envs, -1) if torch.is_tensor(robot_euler) else torch.tensor(robot_euler, dtype=torch.float32, device=_env.device).unsqueeze(1).expand(-1, _env.num_envs, -1)  # (traj_len, num_envs, 3)
    demo_actions[:, :, 6:] = robot_joints.unsqueeze(1).expand(-1, _env.num_envs, -1) if torch.is_tensor(robot_joints) else torch.tensor(robot_joints, dtype=torch.float32, device=_env.device).unsqueeze(1).expand(-1, _env.num_envs, -1)  # (traj_len, num_envs, n_finger)

    # --step_dynamics: env.step with zero residual (same as a silent policy). MotionCommand must supply
    # the nominal pose: base_action_source stays "motion_target" (Leap cfgs), and we sync time_steps each frame.
    # Do NOT use base_action_source="zero" + inverted absolute pose: scales are ~dt*decimation so pose/scale hits
    # raw_clip (-1,1) and targets saturate — hand and fingers look frozen.
    if args_cli.step_dynamics:
        print(
            "[INFO] Manager open-loop: env.step with zero actions and motion_target bases (MotionCommand.time_steps "
            "synced each step; matches training: target = demo + scale*raw). Object is not teleported each step."
        )

    def _sync_manager_hand(timestep: int, *, open_loop_pd: bool) -> None:
        """Apply demo hand pose for manager env: joint-only (floating) or root + fingers (free root)."""
        a = demo_actions[timestep]
        pos_w = a[:, :3] + _env.scene.env_origins
        euler = a[:, 3:6]
        quat = quat_from_euler_xyz_intrinsic(euler[:, 0], euler[:, 1], euler[:, 2])
        z6 = torch.zeros(_env.num_envs, 6, dtype=torch.float32, device=_env.device)
        root_state = torch.cat([pos_w, quat, z6], dim=-1)
        fingers = a[:, 6:]
        hand_asset.write_root_state_to_sim(root_state)
        if open_loop_pd:
            hand_asset.set_joint_position_target(fingers, joint_ids=hand_joint_ids)
        else:
            zf = torch.zeros_like(fingers, dtype=torch.float32, device=_env.device)
            hand_asset.write_joint_state_to_sim(fingers, zf, joint_ids=hand_joint_ids)

    # prepare video recording (manual capture to avoid gym/gymnasium API mismatch)
    video_writer = None
    if args_cli.video:
        video_folder = os.path.join(os.path.dirname(retargeted_traj_path), "videos")
        os.makedirs(video_folder, exist_ok=True)
        mode_suffix = "_step_dynamics" if args_cli.step_dynamics else "_hard_set"
        video_path = os.path.join(video_folder, retargeted_traj_path.split("/")[-1].split(".")[0] + mode_suffix + ".mp4")
        print(f"[INFO] Recording video to: {video_path}")

    # reset environment
    env.reset()
    t = 0
    total_timestep = 0

    def _sync_motion_command_timestep(timestep: int) -> None:
        """Align MotionCommand.time_steps with replay index so env.step does not advance a separate clock.

        Without this, command_manager increments time_steps each step, demo_end / targets desync from
        script frame `t`, and termination resets teleport the scene (visible jumps in video).
        """
        if not args_cli.step_dynamics:
            return
        max_i = motion_cmd.motion.total_demo_len - 1
        ts = int(min(max(timestep, 0), max_i))
        motion_cmd.time_steps.fill_(ts)

    def _write_demo_object_state(timestep: int) -> None:
        """Teleport object to demo frame ``timestep`` (must match synced MotionCommand time for terminations)."""
        object_root_pose = (
            torch.cat([demo_object_pos[timestep], demo_object_quat[timestep]], dim=-1).unsqueeze(0).expand(_env.num_envs, -1)
        )
        object_root_vel = torch.zeros(_env.num_envs, 6, dtype=torch.float32, device=_env.device)
        object_root_state = torch.cat(
            [object_root_pose[:, :3] + _env.scene.env_origins, object_root_pose[:, 3:], object_root_vel], dim=-1
        )
        object_asset.write_root_state_to_sim(object_root_state)
        if hasattr(object_asset, "num_joints") and object_asset.num_joints > 0:
            object_joint_pos = demo_object_joint[timestep]
            if object_joint_pos.ndim == 0:
                object_joint_pos = object_joint_pos.view(1, 1)
            if object_joint_pos.ndim == 1:
                object_joint_pos = object_joint_pos.unsqueeze(0)
            object_joint_pos = object_joint_pos.expand(_env.num_envs, -1)
            object_joint_vel = torch.zeros_like(object_joint_pos, dtype=torch.float32, device=_env.device)
            object_asset.write_joint_state_to_sim(object_joint_pos, object_joint_vel)

    def _replay_eval_firing_terminations(env: ManagerBasedRLEnv, command_timestep: int, env_idx: int = 0):
        """Re-evaluate each termination func with MotionCommand.time_steps = command_timestep (matches pre-command-update step)."""
        ts_bak = motion_cmd.time_steps.clone()
        max_i = motion_cmd.motion.total_demo_len - 1
        motion_cmd.time_steps.fill_(int(min(max(command_timestep, 0), max_i)))
        try:
            term_names, timeout_names = [], []
            tm = env.termination_manager
            for name in tm.active_terms:
                cfg = tm.get_term_cfg(name)
                val = cfg.func(env, **cfg.params)
                if bool(val[env_idx].item()):
                    (timeout_names if cfg.time_out else term_names).append(name)
            ti = int(motion_cmd.timesteps_in_current_demo[env_idx].item())
            dl = int(motion_cmd.current_demo_length[env_idx].item())
            return term_names, timeout_names, ti, dl
        finally:
            motion_cmd.time_steps.copy_(ts_bak)

    def _apply_replay_state(timestep: int):
        max_i = motion_cmd.motion.total_demo_len - 1
        motion_cmd.time_steps.fill_(int(min(max(timestep, 0), max_i)))
        _write_demo_object_state(timestep)
        _sync_manager_hand(timestep, open_loop_pd=False)
        _env.scene.update(dt=_env.physics_dt)

    def _step_open_loop_dynamics(timestep: int):
        # Apply replayed joint positions as low-level targets, bypassing env action semantics.
        if args_cli.step_dynamics:
            _sync_motion_command_timestep(timestep)
            act = torch.zeros_like(demo_actions[timestep])
            step_out = _env.step(act)
            if args_cli.replay_debug and (
                timestep % max(args_cli.replay_log_interval, 1) == 0 or timestep < 3
            ):
                demo_f = demo_actions[timestep, 0, 6:]
                reached = hand_asset.data.joint_pos[0, hand_joint_ids]
                jt = _env.action_manager.get_term("joint_pos")
                proc = jt.processed_actions[0].clone()
                cmd = motion_cmd.target_hand_joint_pos[0, :]
                err = torch.norm(demo_f - reached).item()
                term = step_out[2] if len(step_out) > 2 else None
                tout = step_out[3] if len(step_out) > 3 else None
                term0 = bool(term[0].item()) if term is not None and term.numel() > 0 else False
                tout0 = bool(tout[0].item()) if tout is not None and tout.numel() > 0 else False
                fired_t, fired_to, ti, dl = _replay_eval_firing_terminations(_env, timestep, env_idx=0)
                print(
                    f"[replay_debug] t={timestep} finger L2(demo,reached)={err:.5f} "
                    f"L2(cmd,processed)={torch.norm(cmd - proc).item():.5f} "
                    f"terminated[0]={term0} time_out[0]={tout0} "
                    f"demo_in_demo={ti}/{dl} firing_terms={fired_t} firing_timeouts={fired_to}"
                )
        else:
            print(demo_actions[timestep, 0, 3:6])
            _sync_manager_hand(timestep, open_loop_pd=True)
            hand_asset.write_data_to_sim()
            _env.sim.step()
            _env.scene.update(dt=_env.physics_dt)

    def _visualize_replay_mano(timestep: int) -> None:
        if mano_marker is None:
            return
        mano_target = demo_mano_joint_coord[timestep].unsqueeze(0).expand(_env.num_envs, -1, -1)
        mano_marker.visualize(mano_target + _env.scene.env_origins.unsqueeze(1))

    # In open-loop dynamics mode, initialize the scene once, then step actions.
    if args_cli.step_dynamics:
        _sync_motion_command_timestep(0)
        _apply_replay_state(0)
        _visualize_replay_mano(0)
        frame = env.render()
        if args_cli.video:
            vfps = int(round(getattr(_env, "metadata", {}).get("render_fps", 30)))
            video_writer = imageio.get_writer(video_path, fps=max(vfps, 1))
            frame_to_write = frame[0] if isinstance(frame, (list, tuple)) else frame
            video_writer.append_data(frame_to_write)
            total_timestep += 1

    # simulate environment
    while simulation_app.is_running():
        print(f"[INFO]: Step {t}")
        # run everything in inference mode
        with torch.inference_mode():
            if args_cli.step_dynamics:
                _step_open_loop_dynamics(t)
            else:
                _apply_replay_state(t)

            _visualize_replay_mano(t)

            frame = env.render()
            if args_cli.video:
                # lazily create writer using first frame to ensure correct size
                if video_writer is None:
                    vfps = int(round(getattr(_env, "metadata", {}).get("render_fps", 30)))
                    video_writer = imageio.get_writer(video_path, fps=max(vfps, 1))
                # handle vectorized return (if list/tuple of frames)
                if isinstance(frame, (list, tuple)):
                    frame_to_write = frame[0]
                else:
                    frame_to_write = frame
                video_writer.append_data(frame_to_write)

            t += 1
            if t >= traj_len:
                # loop the demo
                env.reset()
                t = 0
                if args_cli.step_dynamics:
                    _sync_motion_command_timestep(0)
                    _apply_replay_state(0)

            if args_cli.video:
                total_timestep += 1
                # Exit the play loop after recording one video
                if total_timestep == args_cli.video_length:
                    break

    # close the simulator
    if video_writer is not None:
        video_writer.close()
        print(f"Saved video to {video_path}")
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
