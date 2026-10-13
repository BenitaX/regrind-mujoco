"""MuJoCo simulation backend for REGRIND."""

from gymnasium import register, registry


def register_tasks():
    for robot in ("WujiHand", "LeapHand"):
        for object_name in ("Scissors", "Screwdriver"):
            for play in (False, True):
                task = f"Regrind-{robot}-{object_name}{'-Play' if play else ''}-v0"
                if task not in registry:
                    register(task, entry_point="regrind.mujoco.env:RegrindEnv", kwargs={
                        "robot": robot.lower(), "object_name": object_name.lower(), "play": play,
                    })
