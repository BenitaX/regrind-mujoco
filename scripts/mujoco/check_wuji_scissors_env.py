"""Headless smoke check for the WujiHand scissors Gymnasium environment."""

import numpy as np

from wuji_scissors_env import WujiScissorsEnv


def main():
    env = WujiScissorsEnv()
    try:
        observation, info = env.reset(seed=0, options={"start_index": 0})
        assert env.observation_space.contains(observation), observation.shape
        print(f"reset ok: obs={observation.shape}, frame={info['frame']}")

        for step in range(5):
            observation, reward, terminated, truncated, info = env.step(np.zeros(26, dtype=np.float32))
            assert env.observation_space.contains(observation), observation.shape
            assert np.isfinite(reward), reward
            print(
                f"step={step + 1} reward={reward:.3f} contacts={info['contacts']} "
                f"keypoint_error={info['keypoint_error']:.4f} terminated={terminated}"
            )
            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
