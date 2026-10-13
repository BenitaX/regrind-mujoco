"""Run one headless zero-residual WujiHand scissors episode and save step metrics."""

import argparse
import csv
from pathlib import Path

import numpy as np

from wuji_scissors_env import ROOT, WujiScissorsEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "logs/mujoco/wuji_scissors_zero_action.csv")
    args = parser.parse_args()

    env = WujiScissorsEnv()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ("step", "frame", "reward", "keypoint_error", "wrist_pos_error", "wrist_rot_error", "contacts", "terminated", "success", "failure_reason")
    rewards, errors, contacts = [], [], []
    terminated = truncated = False
    try:
        _, _ = env.reset(seed=0, options={"start_index": 0})
        with args.out.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            step = 0
            while not (terminated or truncated):
                _, reward, terminated, truncated, info = env.step(np.zeros(26, dtype=np.float32))
                step += 1
                rewards.append(reward)
                errors.append(info["keypoint_error"])
                contacts.append(info["contacts"])
                writer.writerow({
                    "step": step,
                    "frame": info["frame"],
                    "reward": reward,
                    "keypoint_error": info["keypoint_error"],
                    "wrist_pos_error": info["wrist_pos_error"],
                    "wrist_rot_error": info["wrist_rot_error"],
                    "contacts": info["contacts"],
                    "terminated": terminated,
                    "success": info["success"],
                    "failure_reason": info["failure_reason"],
                })

        print(f"steps: {step}; success: {info['success']}; terminated: {terminated}; truncated: {truncated}; failure: {info['failure_reason'] or 'none'}")
        print(f"reward: total={sum(rewards):.3f}, mean={np.mean(rewards):.3f}")
        print(f"keypoint error (m): mean={np.mean(errors):.4f}, max={np.max(errors):.4f}")
        print(f"contacts: mean={np.mean(contacts):.1f}, max={np.max(contacts)}")
        print(f"metrics: {args.out}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
