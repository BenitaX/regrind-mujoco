"""Task listing, rollouts, training, evaluation and trajectory replay on MuJoCo."""

import argparse
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import regrind


def make_env(task, seed, trajectory=None, render_mode=None):
    def create():
        env = gym.make(task, trajectory_path=trajectory, render_mode=render_mode)
        env.reset(seed=seed)
        return env
    return create


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "zero", "random", "train", "play", "replay"])
    parser.add_argument("--task", default="Regrind-WujiHand-Scissors-v0")
    parser.add_argument("--suite", choices=["WujiHand", "LeapHand"])
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num-envs", "--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trajectory", "--retargeted_traj_path", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=Path("models/mujoco"))
    args = parser.parse_args(argv)
    if args.command == "list":
        for task in sorted(gym.registry):
            if task.startswith("Regrind-") and (args.suite is None or args.suite in task):
                print(task)
        return
    if args.num_envs < 1 or args.steps < 1 or args.total_timesteps < 1 or args.n_steps < 2 or args.batch_size < 2:
        parser.error("Environment counts and step counts must be positive; PPO rollout and batch sizes must be >= 2")
    if args.command == "train":
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
        factories = [make_env(args.task, args.seed + i, args.trajectory) for i in range(args.num_envs)]
        vector = VecMonitor((DummyVecEnv if args.num_envs == 1 else SubprocVecEnv)(factories))
        args.out.mkdir(parents=True, exist_ok=True)
        try:
            policy = PPO.load(args.checkpoint, env=vector) if args.checkpoint else PPO(
                "MlpPolicy", vector, seed=args.seed, verbose=1, n_steps=args.n_steps,
                batch_size=args.batch_size, policy_kwargs={"net_arch": [256, 256, 128]}, device="cpu")
            if args.checkpoint:
                vector.env_method("set_training_steps", policy.num_timesteps // args.num_envs)
                vector.reset()
            policy.learn(args.total_timesteps, reset_num_timesteps=args.checkpoint is None,
                         callback=CheckpointCallback(max(1, 100_000 // args.num_envs), str(args.out), args.task))
            policy.save(args.out / args.task)
            print(f"Saved {args.out / args.task}.zip")
        finally:
            vector.close()
        return
    policy = None
    if args.command == "play":
        if args.checkpoint is None:
            parser.error("play requires --checkpoint")
        from stable_baselines3 import PPO
        policy = PPO.load(args.checkpoint)
    task = args.task
    if args.command in ("play", "replay") and "-Play-" not in task:
        task = task.replace("-v0", "-Play-v0")
    env = gym.make(task, trajectory_path=args.trajectory, render_mode=None if args.headless else "human")
    try:
        obs, _ = env.reset(seed=args.seed, options={"start_index": 0})
        env.action_space.seed(args.seed)
        if args.command == "replay":
            raw = env.unwrapped
            for frame in range(raw.length):
                raw.frame = frame
                for address, prefix in ((raw.wrist_qpos, "robot"), (raw.object_qpos, "object")):
                    raw.data.qpos[address:address + 7] = np.r_[raw.trajectory[f"{prefix}_pos"][frame], raw.trajectory[f"{prefix}_quat"][frame]]
                raw.data.qpos[raw.finger_qpos] = raw.trajectory["robot_joints"][frame]
                if raw.articulated:
                    raw.data.qpos[raw.hinge_qpos] = np.asarray(raw.trajectory["object_joint"][frame]).item()
                mujoco.mj_forward(raw.model, raw.data)
                if not args.headless:
                    env.render()
                    time.sleep(raw.dt)
            print(f"Replayed {raw.length} policy frames")
            return
        total_reward = 0.0
        for _ in range(args.steps):
            action = policy.predict(obs, deterministic=True)[0] if policy else (env.action_space.sample() if args.command == "random" else np.zeros(env.action_space.shape))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if not args.headless:
                time.sleep(env.unwrapped.dt)
            if terminated or truncated:
                obs, _ = env.reset()
        print(f"Completed {args.steps} steps; reward={total_reward:.3f}; last info={info}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
