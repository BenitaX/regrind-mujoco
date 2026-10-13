# REGRIND

Implementation of [A Minimalist Retargeting-Guided Reinforcement Learning Recipe for Dexterous Manipulation](https://www.yunhaifeng.com/REGRIND/), with a MuJoCo simulation backend.

![Method overview](data/figures/method_overview.png)

## Installation

### Prerequisites

The full setup supports Linux, macOS, and Windows through WSL2. Install
[Miniforge](https://github.com/conda-forge/miniforge) or
[Miniconda](https://docs.conda.io/projects/conda/en/stable/), plus
[Git](https://git-scm.com/downloads) and [Git LFS](https://git-lfs.com/).
Use an OS and Python version supported by
[Drake](https://drake.mit.edu/installation.html) for retargeting; the commands
below use Python 3.12. Ubuntu 24.04 is a supported Linux option. macOS users
should check Drake's current OS and architecture requirements.

Initialize your shell with `conda init bash` or `conda init zsh` if needed,
then reopen the terminal. For native Windows simulation/training or an install
without retargeting, see [Optional installation paths](#optional-installation-paths).

#### Windows: set up WSL2 first

From an Administrator PowerShell:

```powershell
wsl --install -d Ubuntu-24.04
```

Restart if prompted, open Ubuntu, and finish its user setup. Install the Linux
version of Miniforge/Miniconda inside Ubuntu and run the following setup commands
in its terminal. See
[Microsoft's WSL installation guide](https://learn.microsoft.com/en-us/windows/wsl/install).
Linux and macOS users can proceed directly to the repository setup.

### Get the repository

On Ubuntu/WSL, install Git and Git LFS with:

```bash
sudo apt update
sudo apt install -y git git-lfs
```

Clone the repository:

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/yunhaif/regrind.git
cd regrind
git lfs install
git lfs pull
```

Use your existing checkout if you have one. Preserve local edits when moving
between systems; a fresh clone does not include them. In WSL, prefer a checkout
under `~/projects` for Linux filesystem performance; see
[Microsoft's filesystem guidance](https://learn.microsoft.com/en-us/windows/wsl/filesystems).

Git LFS downloads actual meshes and data if the checkout contains LFS pointers.
The Python `git-lfs` dependency does not replace the Git LFS executable.

### Create a Conda environment

From the repository root:

```bash
conda create -n regrind -c conda-forge python=3.12 pip -y
conda activate regrind
python -c "import sys; print(sys.executable)"
python -m pip install --upgrade pip
```

The interpreter should be inside `.../envs/regrind/bin/python`.
If it points to a global Python, fix activation before
installing packages. If an environment named `regrind` already exists,
reuse it after checking its Python version, or choose a new name and substitute
it in the commands. See
[Conda environment management](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html).

### Install dependencies

```bash
python -m pip install -e "source/regrind[train,retargeting]"
python -m pip install tqdm
python -m pip check
```

This installs MuJoCo, Gymnasium, NumPy, SciPy, h5py, the other core dependencies,
Stable Baselines3 and PyTorch for training, and Drake for retargeting. `tqdm` is
installed explicitly because the current retargeting extra does not declare it.
Python 3.12 on Ubuntu 24.04 is listed in
[Drake's supported configurations](https://drake.mit.edu/installation.html).
Drake notes that Conda is not regularly tested; if a Conda-specific issue occurs,
a Python virtual environment is an alternative.

The current PPO launcher uses CPU workers and does not require CUDA. Isaac Sim
and Isaac Lab are not required. Mosek licensing is covered under
[Retargeting](#retargeting); it is only needed for generating trajectories with
the script's default solver. Installing packages alone does not configure it.

### Set data paths

From the repository root:

```bash
export REGRIND_DATA_DIR="$PWD/data"
conda env config vars set "REGRIND_DATA_DIR=$REGRIND_DATA_DIR"
conda deactivate
conda activate regrind
```

This persists the absolute data path in the environment. Update it if you move
the checkout. Retargeting requires this variable; simulation also resolves
included data automatically when run from the checkout. Robot and object models
are packaged with `regrind`.

### Verify installation

```bash
python -c "import mujoco, gymnasium, numpy, scipy, h5py, torch, stable_baselines3, regrind; print('Simulation and training imports OK')"
python -c "from pydrake.all import MosekSolver, ClarabelSolver; from regrind.retargeting.retargeter import HandInteractionMeshOneStageRetargeter; print('Retargeting imports OK')"
python scripts/list_envs.py
python scripts/check_mujoco.py
python scripts/zero_agent.py --task Regrind-WujiHand-Scissors-v0 --headless --steps 30
```

For a short training check:

```bash
python scripts/rsl_rl/train.py --task Regrind-WujiHand-Scissors-v0 --num-envs 1 --total-timesteps 64 --n-steps 32 --batch-size 32 --out models/mujoco/smoke
```

A successful retargeting import does not verify the solver license or an actual
optimization. Follow the solver and short retargeting checks below as well.

Each new terminal needs `conda activate regrind`. Run commands from the repository root.
For a MuJoCo viewer, omit `--headless` from rollout/replay/play commands; this
requires a working graphical session and OpenGL support. See
[WSL GUI application support](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps)
if graphics need configuration. Headless commands do not need a viewer.

### Optional installation paths

#### Simulation and training without retargeting

On any supported MuJoCo platform, use the included trajectories and install only
the dependencies you need. After creating and activating your Conda environment:

```bash
python -m pip install -e "source/regrind[train]"
python -m pip check
```

For simulation only, use `python -m pip install -e source/regrind` instead.
Skip the Drake import and solver checks. You can add the retargeting extra later
on Linux, macOS, or WSL.

#### Native Windows simulation/training with retargeting in WSL

If you prefer native Windows for simulation/training, open your checkout in a
Windows terminal with Windows Miniforge/Miniconda initialized and run:

```powershell
conda create -n regrind-mujoco -c conda-forge python=3.12 pip -y
conda activate regrind-mujoco
python -c "import sys; print(sys.executable)"
python -m pip install --upgrade pip
python -m pip install -e "source/regrind[train]"
python -m pip check
$env:REGRIND_DATA_DIR = (Resolve-Path ./data).Path
python scripts/check_mujoco.py
```

Drake does not support native Windows. If you also need to generate trajectories,
use the full setup above inside WSL for retargeting, then load the resulting
`.h5` files in Windows using their Windows paths. Windows and WSL need separate
Python environments because they run different operating systems; separate
repository copies are optional. This split is a platform preference, not a
requirement caused by known dependency conflicts or a runtime optimization.

## Simulation and training

All four robot/object pairs are available: `WujiHand` or `LeapHand`, with
`Scissors` or `Screwdriver`. Evaluation IDs include `-Play-v0`.

```bash
python scripts/list_envs.py
python scripts/zero_agent.py --task Regrind-WujiHand-Scissors-v0 --headless --steps 300
python scripts/random_agent.py --task Regrind-LeapHand-Screwdriver-v0 --headless
python scripts/replay_retargeted_traj.py --task Regrind-LeapHand-Scissors-Play-v0 --headless
python scripts/rsl_rl/train.py --task Regrind-WujiHand-Scissors-v0 --num-envs 8 --total-timesteps 10000000
python scripts/rsl_rl/play.py --task Regrind-WujiHand-Scissors-Play-v0 --checkpoint models/mujoco/Regrind-WujiHand-Scissors-v0.zip
python scripts/check_mujoco.py
```

The historical `scripts/rsl_rl` paths now launch Stable Baselines3 PPO. The
installed `regrind` command exposes the same `list`, `zero`, `random`, `replay`,
`train`, and `play` commands. Physics runs at 120 Hz and the default policy at
30 Hz. Training is headless and uses CPU workers; omit `--headless` for a viewer
in rollout/evaluation commands.

```python
import gymnasium as gym
import regrind

env = gym.make("Regrind-WujiHand-Scissors-v0")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

See [MIGRATION.md](MIGRATION.md) for implemented features and remaining parity
work. Existing Isaac policies require retraining. Original launchers remain in
`scripts/legacy_isaac/` for reference.

## Retargeting

Note: The retargeted trajectories are also provided in this repository. You can directly
[start training](#simulation-and-training) without running retargeting.

### Environment

The [full installation](#installation) already installs Drake and
`tqdm` alongside simulation and training dependencies. Activate the environment before running retargeting:

```bash
conda activate regrind
```

If you initially installed only simulation/training dependencies on Linux or supported macOS,
add retargeting to that existing environment:

```bash
python -m pip install -e "source/regrind[train,retargeting]"
python -m pip install tqdm
python -m pip check
```

The retargeter uses [Drake](https://drake.mit.edu/) (`pydrake`), which does not
support native Windows. It remains optional for simulation/training with the
included trajectories. Check
[Drake's platform requirements](https://drake.mit.edu/installation.html) for
native Linux/macOS installations.

### License (Mosek)

The script currently defaults to Drake's [Mosek solver](https://drake.mit.edu/pydrake/pydrake.solvers.html),
which requires a license. Drake includes its solver integration; installing the
separate Python `mosek` package is not required for this code path.

Mosek offers free [personal academic licenses](https://www.mosek.com/products/academic-licenses/)
and trial licenses. Download the license file and point Drake at it explicitly:

```bash
export MOSEKLM_LICENSE_FILE=/path/to/mosek.lic
```

Verify that Drake can acquire the license before a long run:

```bash
python -c "from pydrake.all import MosekSolver; license_handle = MosekSolver.AcquireLicense(); print('Mosek license acquired')"
```

Mosek is optional at the retargeter API level: pass `solver=ClarabelSolver()` to
`HandInteractionMeshOneStageRetargeter(...)` for the license-free alternative.
The current `scripts/retarget_hand_object.py` does **not** expose a `--solver`
argument; using Clarabel with that script requires adding the solver argument
at its constructor call and importing `ClarabelSolver` from `pydrake.all`.
Drake is still required with either solver.

### Running retargeting

Set `REGRIND_DATA_DIR` to the absolute repository `data` directory so demos, keypoints,
and assets resolve, then run:

```bash
python scripts/retarget_hand_object.py --robot wujihand --object scissors --no-visualize
```

Choose `--robot leaphand` or `wujihand`, and `--object scissors` or `screwdriver`.
For an initial test, add `--num-timesteps 5 --out /tmp/regrind-test.h5`.
The default solver needs the Mosek license configured above. `--no-visualize`
disables Meshcat visualization. Ensure the demo and keypoint files are present
under `REGRIND_DATA_DIR`; these are separate from the robot/object assets.

The output `.h5` follows the schema expected by `load_retargeted_traj` (keys `robot_pos`,
`robot_quat`, `robot_joints`, `object_pos`, `object_quat`, `object_joint`, `robot_keypoints`,
`mano_joint_coords`). Visualize / sanity-check it against an environment by overriding the
trajectory path:

```bash
python scripts/replay_retargeted_traj.py --task Regrind-LeapHand-Scissors-Play-v0 \
    --headless --retargeted_traj_path /path/to/out.h5
```

Match the replay task to the robot and object used for retargeting. In the
Linux/macOS/WSL setup, use the generated file directly in the same environment:
pass its Linux path with `--retargeted_traj_path` for replay or `--trajectory`
for training. No copying between systems is needed. Solver licensing is only
needed when generating trajectories, not when loading an existing `.h5` file.
