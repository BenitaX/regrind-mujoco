"""Train a PPO policy on the standalone MuJoCo WujiHand scissors task."""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from wuji_scissors_env import ROOT, WujiScissorsEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "models/mujoco/wuji_scissors_ppo")
    parser.add_argument("--resume", type=Path, help="Resume from a saved PPO .zip checkpoint")
    args = parser.parse_args()
    if args.num_envs < 1 or args.total_timesteps < 1:
        parser.error("--num-envs and --total-timesteps must be positive")

    env = VecMonitor(SubprocVecEnv([WujiScissorsEnv for _ in range(args.num_envs)], start_method="spawn"))
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.resume:
            model = PPO.load(args.resume, env=env, device="auto")
        else:
            model = PPO("MlpPolicy", env, verbose=1, seed=args.seed)
        checkpoint = CheckpointCallback(
            save_freq=max(1, 100_000 // args.num_envs),
            save_path=str(args.out.parent),
            name_prefix=args.out.name,
        )
        try:
            model.learn(total_timesteps=args.total_timesteps, callback=checkpoint)
        except KeyboardInterrupt:
            print("Training interrupted; saving the current policy.")
        model.save(str(args.out))
        print(f"Saved policy to {args.out}.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
