# Isaac Sim to MuJoCo migration

The default package import, task registration and launchers now use MuJoCo and
Gymnasium. All four robot/object pairs have training and play task IDs.
The original launchers are retained in `scripts/legacy_isaac/`; the manager-based
Isaac implementation remains as a reference under `regrind/tasks/manager_based`.
It is not loaded by `import regrind` or `import regrind.tasks`.

## Implemented

- WujiHand MJCF and LeapHand URDF scenes with free wrists, position-controlled
  finger joints, gravity-compensated hands, convex object collision pieces,
  scissors hinge and task-specific scales/table heights.
- Recorded 120 Hz HDF5 trajectories sampled at the policy interval, world-frame
  wrist force/torque control, relative finger targets, keypoint and velocity
  rewards, bounds/deviation failures and a 9-second episode timeout.
- Seeded random starting frames, trajectory yaw/translation augmentation,
  friction and object mass/inertia randomization, object position observation
  noise and discrete state observation delay.
- Gravity curriculum from zero to 9.81 m/s² using the original stage schedule,
  counted independently per worker; play uses full gravity.
- CPU PPO training with Stable Baselines3, parallel workers, checkpoints,
  evaluation, zero/random rollouts and kinematic trajectory replay.

## Differences requiring further parity work

- The policy uses a flat observation containing current state/velocities, target
  poses/joints, previous action and trajectory phase. It does not implement the
  original separate privileged critic, two-frame hand histories, or custom
  asymmetric actor/critic. Isaac checkpoints cannot be loaded into this PPO path.
- Noise/delay and randomization are a smaller subset of the original settings:
  rotational/joint noise, fractional per-channel delays, COM/mesh scale changes,
  actuator randomization, random pushes and adaptive failure sampling remain.
- Trajectory augmentation is applied to the whole trajectory at reset, rather
  than using the original augmentation start/blend interval.
- MuJoCo contact parameters and URDF collision/inertia assets are used. USD-only
  printed-object physics properties and PhysX friction/restitution behavior have
  not been replicated or quantitatively compared. Table force termination and
  contact observations/debug drawing remain to be ported.
- Resuming PPO restores policy/optimizer state and derives worker gravity clocks
  from the checkpoint timestep count divided by the current worker count. Keep
  the worker count unchanged to preserve the curriculum schedule.
- Human viewing is supported; RGB video export and performance equivalence to
  Isaac's 4096 GPU environments are not implemented.

## Validation

Run `python scripts/check_mujoco.py` for all eight task registrations, Gymnasium
API checks, finite headless rollouts and confirmation that no Isaac modules load.
This checks runtime behavior, not learned manipulation quality or physics parity.

Validated on this workspace with MuJoCo 3.14.0 and Gymnasium 1.3.0:

- All eight task variants passed `check_env`, seeded randomized resets and
  finite zero-action rollouts without importing Isaac modules.
- A two-worker PPO run completed 16 timesteps and saved a checkpoint.
- Checkpoint evaluation completed 20 headless steps with finite rewards.
- LeapHand screwdriver replay completed 271 policy frames.
- `git diff --check` passed. Human viewing and long-run learning were not tested.

The scene conversion follows the frame/inertia conventions in the
[MuJoCo XML reference](https://mujoco.readthedocs.io/en/latest/XMLreference.html).
