"""REGRIND MuJoCo random launcher."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/regrind"))
from regrind.mujoco.cli import main

if __name__ == "__main__":
    main(["random", *sys.argv[1:]])
