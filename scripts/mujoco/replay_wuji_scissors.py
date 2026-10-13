"""Replay a retargeted WujiHand/scissors trajectory in MuJoCo."""

import argparse
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import mujoco
import mujoco.viewer


ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = ROOT / "source/regrind/regrind/assets"
HAND_XML = ASSET_ROOT / "mjcf/free_wuji_right_hand_full_simplified_v1.xml"
SCISSORS_URDF = ASSET_ROOT / "scissors/scissors.urdf"
DEFAULT_TRAJECTORY = ROOT / "data/retargeted_traj/wujihand/scissors/retargeted_120fps.h5"
JOINT_NAMES = [f"right_finger{i}_joint{j}" for i in range(1, 6) for j in range(1, 5)]


def _add_inertial(body, link):
    inertial = link.find("inertial")
    origin = inertial.find("origin")
    mass = inertial.find("mass").get("value")
    tensor = inertial.find("inertia")
    ET.SubElement(
        body,
        "inertial",
        pos=origin.get("xyz", "0 0 0"),
        mass=mass,
        fullinertia=" ".join(
            tensor.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
        ),
    )


def _add_link_meshes(body, link, model_root, prefix):
    asset = model_root.find("asset")
    for index, collision in enumerate(link.findall("collision")):
        mesh = collision.find("geometry/mesh")
        if mesh is None:
            continue
        name = f"{prefix}_{index}"
        mesh_path = (SCISSORS_URDF.parent / mesh.get("filename")).resolve()
        ET.SubElement(asset, "mesh", name=name, file=mesh_path.as_posix(), scale=mesh.get("scale", "1 1 1"))
        origin = collision.find("origin")
        geom = {"type": "mesh", "mesh": name, "rgba": "0.9 0.45 0.12 1"}
        if origin is not None:
            if origin.get("xyz"):
                geom["pos"] = origin.get("xyz")
            if origin.get("rpy") and origin.get("rpy") != "0 0 0":
                # The repository scissors collision mesh origins are all identity.
                raise ValueError("Non-identity scissors mesh rotations need explicit conversion to MuJoCo quaternions.")
        ET.SubElement(body, "geom", geom)

def build_scene_xml():
    tree = ET.parse(HAND_XML)
    model = tree.getroot()
    for asset in model.findall("./asset/*"):
        if asset.get("file"):
            asset.set("file", (HAND_XML.parent / asset.get("file")).resolve().as_posix())
    world = model.find("worldbody")
    hand_root = world.find("body[@name='root']")
    hand_root.insert(0, ET.Element("freejoint", name="wrist_free"))

    ET.SubElement(world, "geom", name="table", type="plane", pos="0 0 0.952", size="1 1 0.1", rgba="0.35 0.35 0.38 1", friction="1 0.005 0.0001")

    robot = ET.parse(SCISSORS_URDF).getroot()
    links = {link.get("name"): link for link in robot.findall("link")}
    scissors = ET.SubElement(world, "body", name="scissors", pos="0 0 1")
    ET.SubElement(scissors, "freejoint", name="scissors_free")
    _add_inertial(scissors, links["bottom"])
    _add_link_meshes(scissors, links["bottom"], model, "scissors_bottom")

    top = ET.SubElement(scissors, "body", name="scissors_top")
    ET.SubElement(top, "joint", name="scissors_joint", type="hinge", axis="0 0 -1", range="0 0.5", damping="0")
    _add_inertial(top, links["top"])
    _add_link_meshes(top, links["top"], model, "scissors_top")

    return ET.tostring(model, encoding="unicode")


def _set_free_pose(model, data, joint_name, pos, quat):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos = model.jnt_qposadr[joint_id]
    data.qpos[qpos : qpos + 3] = pos
    data.qpos[qpos + 3 : qpos + 7] = quat  # HDF5 quaternions are wxyz.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--headless", action="store_true", help="Replay without opening the MuJoCo viewer.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed must be positive")

    with h5py.File(args.trajectory, "r") as h5:
        trajectory = {key: h5[key][:] for key in (
            "robot_pos", "robot_quat", "robot_joints", "object_pos", "object_quat", "object_joint"
        )}
    count = len(trajectory["robot_pos"])
    if any(len(value) != count for value in trajectory.values()):
        raise ValueError("Trajectory datasets have inconsistent lengths")
    if trajectory["robot_joints"].shape[1] != len(JOINT_NAMES):
        raise ValueError("Expected 20 WujiHand joint values in robot_joints")

    # Close the XML before MuJoCo opens it (required on Windows).
    with tempfile.TemporaryDirectory(dir=HAND_XML.parent) as temp_dir:
        scene_path = Path(temp_dir) / "scene.xml"
        scene_path.write_text(build_scene_xml(), encoding="utf-8")
        model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    finger_qpos = [
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in JOINT_NAMES
    ]
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "scissors_joint")
    scissors_joint_qpos = model.jnt_qposadr[joint_id]

    def replay(viewer=None):
        for frame in range(count):
            _set_free_pose(model, data, "wrist_free", trajectory["robot_pos"][frame], trajectory["robot_quat"][frame])
            _set_free_pose(model, data, "scissors_free", trajectory["object_pos"][frame], trajectory["object_quat"][frame])
            data.qpos[scissors_joint_qpos] = float(trajectory["object_joint"][frame].reshape(-1)[0])
            data.qpos[finger_qpos] = trajectory["robot_joints"][frame]
            mujoco.mj_forward(model, data)
            if viewer is not None:
                viewer.sync()
                time.sleep((1.0 / 120.0) / args.speed)
        print(f"Replayed {count} frames from {args.trajectory}")

    if args.headless:
        replay()
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            start = trajectory["robot_pos"][0]
            _set_free_pose(model, data, "wrist_free", start, trajectory["robot_quat"][0])
            _set_free_pose(model, data, "scissors_free", trajectory["object_pos"][0], trajectory["object_quat"][0])
            mujoco.mj_forward(model, data)
            viewer.cam.lookat[:] = start + (0.0, 0.0, 0.12)
            viewer.cam.distance = 0.65
            viewer.cam.azimuth = 135
            viewer.cam.elevation = 20
            viewer.sync()
            replay(viewer)
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.03)


if __name__ == "__main__":
    main()
