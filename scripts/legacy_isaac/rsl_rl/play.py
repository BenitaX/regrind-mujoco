# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=270, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--zero_actions", action="store_true", default=False, help="Use zero actions to step the environment.")
parser.add_argument(
    "--auto_gravity_from_ckpt",
    action="store_true",
    default=False,
    help=(
        "Infer the gravity the checkpoint was trained under from its iteration number and the run's "
        "gravity curriculum, and fix sim.gravity to that value for eval (overrides the cfg gravity)."
    ),
)
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time, if possible.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnv,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import regrind.tasks  # noqa: F401
from regrind.modules.actor_critic import CustomActorCritic  # noqa: F401

# Register the custom policy class in RSL-RL runner's namespace so eval() can resolve it
import rsl_rl.runners.on_policy_runner as _rsl_rl_on_policy_runner  # noqa: E402
_rsl_rl_on_policy_runner.CustomActorCritic = CustomActorCritic  # type: ignore[attr-defined]


# PLACEHOLDER: Extension template (do not remove this comment)

def _infer_gravity_from_checkpoint(resume_path: str, num_steps_per_env: int) -> tuple[float, float, float] | None:
    """Infer the gravity a checkpoint was trained under from its iteration and the run's curriculum.

    The gravity curriculum (``curriculum_gravity_with_randomization``) maps ``env.common_step_counter``
    to a gravity range, and ``common_step_counter == iteration * num_steps_per_env``. The iteration is
    encoded in the checkpoint filename (``model_<iter>.pt``) and the exact schedule (including any
    train-time Hydra overrides) is dumped to ``<run>/params/env.yaml``. We reproduce the same stage
    lookup the curriculum uses and collapse the resulting range to its midpoint.

    Returns the ``(gx, gy, gz)`` gravity, or ``None`` if it cannot be determined.
    """
    import re
    import yaml

    model_file = os.path.basename(resume_path)
    m = re.fullmatch(r"model_(\d+)\.pt", model_file)
    if m is None:
        print(f"[WARN] --auto_gravity_from_ckpt: cannot parse iteration from {model_file!r}; skipping.")
        return None
    iteration = int(m.group(1))

    env_yaml_path = os.path.join(os.path.dirname(resume_path), "params", "env.yaml")
    if not os.path.isfile(env_yaml_path):
        print(f"[WARN] --auto_gravity_from_ckpt: no dumped config at {env_yaml_path}; skipping.")
        return None

    # env.yaml is dumped with !!python/tuple tags, so use unsafe_load.
    with open(env_yaml_path) as f:
        env_dump = yaml.unsafe_load(f)
    try:
        gravity_stages = env_dump["events"]["curriculum_gravity"]["params"]["gravity_stages"]
    except (KeyError, TypeError):
        print(f"[WARN] --auto_gravity_from_ckpt: no curriculum_gravity in {env_yaml_path}; skipping.")
        return None

    current_step = iteration * num_steps_per_env

    # Mirror curriculum_gravity_with_randomization: take the last stage whose threshold <= current_step.
    g_min: tuple[float, float, float] = (0.0, 0.0, -9.81)
    g_max: tuple[float, float, float] = (0.0, 0.0, -9.81)
    for step_threshold, gravity_value_min, gravity_value_max in sorted(gravity_stages, key=lambda s: s[0]):
        if current_step >= step_threshold:
            g_min = tuple(gravity_value_min)
            g_max = tuple(gravity_value_max)
        else:
            break

    gravity = tuple((a + b) / 2.0 for a, b in zip(g_min, g_max))
    print(
        f"[INFO] --auto_gravity_from_ckpt: iter={iteration} x {num_steps_per_env} steps = {current_step} -> "
        f"gravity range {g_min}..{g_max} -> midpoint {gravity}"
    )
    return gravity  # type: ignore[return-value]


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    model_name = resume_path.split('/')[-1].split('.')[0]

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # optionally fix gravity to the value the checkpoint was trained under (gravity curriculum)
    if args_cli.auto_gravity_from_ckpt:
        inferred_gravity = _infer_gravity_from_checkpoint(resume_path, agent_cfg.num_steps_per_env)
        if inferred_gravity is not None:
            env_cfg.sim.gravity = inferred_gravity

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": model_name,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename=f"{model_name}_policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename=f"{model_name}_policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            actions = policy(obs)
            if args_cli.zero_actions:
                actions.zero_()
            obs, rewards, dones, extras = env.step(actions)

        timestep += 1
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
