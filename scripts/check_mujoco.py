"""Validate every registered MuJoCo task and representative API edge cases."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/regrind"))

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env
import regrind


def main():
    for task in sorted(name for name in gym.registry if name.startswith("Regrind-")):
        env = gym.make(task, randomization=False, observation_noise=False, observation_delay=0)
        try:
            check_env(env.unwrapped, skip_render_check=True)
            obs, _ = env.reset(seed=123, options={"start_index": 0})
            assert env.observation_space.contains(obs)
            for _ in range(30):
                obs, reward, done, truncated, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
                assert np.all(np.isfinite(obs)) and np.isfinite(reward)
                if done or truncated:
                    env.reset(options={"start_index": 0})
            print(f"PASS {task}: action={env.action_space.shape}, observation={obs.shape}")
            try:
                env.reset(options={"start_index": -1})
            except ValueError:
                pass
            else:
                raise AssertionError("Negative trajectory index was accepted")
        finally:
            env.close()
        randomized = gym.make(task)
        try:
            first, _ = randomized.reset(seed=42, options={"start_index": 0})
            second, _ = randomized.reset(seed=42, options={"start_index": 0})
            np.testing.assert_array_equal(first, second)
            result = randomized.step(np.zeros(randomized.action_space.shape))
            assert np.all(np.isfinite(result[0])) and np.isfinite(result[1])
        finally:
            randomized.close()
    assert not any(name.startswith(("isaaclab", "omni", "isaacsim")) for name in sys.modules)
    print("All tasks passed without importing Isaac Sim or Isaac Lab")


if __name__ == "__main__":
    main()
