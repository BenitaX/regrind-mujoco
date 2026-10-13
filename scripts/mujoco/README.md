# MuJoCo WujiHand scissors environment

This is a standalone Gymnasium environment and does not import Isaac Lab. Install its optional dependencies in the Python environment used for MuJoCo:

```powershell
python -m pip install -e "source/regrind[mujoco]"
```

Train a PPO policy using parallel MuJoCo environments (no Isaac Sim or Isaac Lab):

```powershell
python scripts/mujoco/train_wuji_scissors.py --num-envs 8 --total-timesteps 10000000
```

The policy is saved under `models/mujoco/`. Adjust `--num-envs` to fit available CPU resources.

It implements Gymnasium's `reset()` and `step()` API with a 26-value action (6 wrist deltas and 20 finger deltas), a 131-value observation, and headless MuJoCo physics by default. The default trajectory is the repo's retargeted WujiHand scissors HDF5.

```python
from scripts.mujoco.wuji_scissors_env import WujiScissorsEnv

env = WujiScissorsEnv()
obs, info = env.reset(seed=1)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

Use `render_mode="human"` to open the MuJoCo viewer. The environment runs physics at 120 Hz and advances the 120 fps trajectory at 30 Hz, matching the Isaac task's four physics steps per policy action.

This MuJoCo path currently covers one robot/object pair. Observation noise and delay, trajectory augmentation, domain randomization, the other robot/object pairs, and parity with the Isaac task's observations remain future migration work.
