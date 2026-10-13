"""Build portable MJCF scenes from the repository's robot and object assets."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from regrind.assets import REGRIND_ASSETS_DIR as ASSETS

WUJI_JOINTS = [f"right_finger{i}_joint{j}" for i in range(1, 6) for j in range(1, 5)]
LEAP_JOINTS = [f"{finger}_{joint}" for finger in ("if", "mf", "rf")
               for joint in ("mcp", "rot", "pip", "dip")] + ["th_cmc", "th_axl", "th_mcp", "th_ipl"]


def origin(element):
    if element is None:
        return {}
    result = {"pos": element.get("xyz", "0 0 0")}
    quat = Rotation.from_euler("xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")).as_quat()
    result["quat"] = " ".join(map(str, quat[[3, 0, 1, 2]]))
    return result


def add_urdf(model, path, root_name, free_name, gravcomp=False):
    """Preserve URDF link frames, inertia, primitive collisions and convex mesh pieces."""
    urdf = ET.parse(path).getroot()
    links = {link.get("name"): link for link in urdf.findall("link")}
    joints = urdf.findall("joint")
    if root_name == "root":
        for joint in joints:
            if joint.get("name", "").isdigit():
                joint.set("name", LEAP_JOINTS[int(joint.get("name"))])
    children = {joint.find("child").get("link") for joint in joints}
    root_link = next(name for name in links if name not in children)
    asset = model.find("asset")

    def add(parent, link_name, joint=None):
        name = root_name if joint is None else f"{root_name}_{link_name}"
        body = ET.SubElement(parent, "body", name=name, **(origin(joint.find("origin")) if joint is not None else {}))
        if gravcomp:
            body.set("gravcomp", "1")
        if joint is None:
            ET.SubElement(body, "freejoint", name=free_name)
        elif joint.get("type") != "fixed":
            limit = joint.find("limit")
            attrs = dict(name=joint.get("name"), type="slide" if joint.get("type") == "prismatic" else "hinge",
                         axis=joint.find("axis").get("xyz", "1 0 0") if joint.find("axis") is not None else "1 0 0",
                         damping="0.1" if gravcomp else "0", armature="0.0002")
            if limit is not None and joint.get("type") != "continuous":
                attrs.update(limited="true", range=f"{limit.get('lower')} {limit.get('upper')}")
            ET.SubElement(body, "joint", **attrs)
        link = links[link_name]
        inertial = link.find("inertial")
        if inertial is not None:
            tensor = inertial.find("inertia")
            # URDF inertia is expressed in the inertial frame; rotate it into the link frame.
            xx, yy, zz, xy, xz, yz = [float(tensor.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]
            matrix = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
            pose = origin(inertial.find("origin"))
            q = np.fromstring(pose.get("quat", "1 0 0 0"), sep=" ")
            rotation = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
            matrix = rotation @ matrix @ rotation.T
            ET.SubElement(body, "inertial", pos=pose.get("pos", "0 0 0"), mass=inertial.find("mass").get("value"),
                          fullinertia=" ".join(map(str, [matrix[0, 0], matrix[1, 1], matrix[2, 2], matrix[0, 1], matrix[0, 2], matrix[1, 2]])))
        elif joint is None:
            ET.SubElement(body, "inertial", pos="0 0 0", mass="0.001", diaginertia="1e-6 1e-6 1e-6")
        for index, collision in enumerate(link.findall("collision")):
            geometry = collision.find("geometry")
            geom = dict(origin(collision.find("origin")), rgba="0.8 0.5 0.2 1")
            mesh = geometry.find("mesh")
            if mesh is not None:
                mesh_name = f"{name}_{index}"
                ET.SubElement(asset, "mesh", name=mesh_name, file=(path.parent / mesh.get("filename")).resolve().as_posix(), scale=mesh.get("scale", "1 1 1"))
                geom.update(type="mesh", mesh=mesh_name)
            elif geometry.find("box") is not None:
                geom.update(type="box", size=" ".join(map(str, np.fromstring(geometry.find("box").get("size"), sep=" ") / 2)))
            elif geometry.find("sphere") is not None:
                geom.update(type="sphere", size=geometry.find("sphere").get("radius"))
            elif geometry.find("cylinder") is not None:
                cylinder = geometry.find("cylinder")
                geom.update(type="cylinder", size=f"{cylinder.get('radius')} {float(cylinder.get('length')) / 2}")
            else:
                raise ValueError(f"Unsupported geometry in {path}: {link_name}")
            ET.SubElement(body, "geom", **geom)
        for child_joint in joints:
            if child_joint.find("parent").get("link") == link_name:
                add(body, child_joint.find("child").get("link"), child_joint)
        return body

    return add(model.find("worldbody"), root_link)


def build_scene_xml(robot="wujihand", object_name="scissors"):
    if robot not in ("wujihand", "leaphand") or object_name not in ("scissors", "screwdriver"):
        raise ValueError("Expected wujihand/leaphand and scissors/screwdriver")
    if robot == "wujihand":
        path = ASSETS / "mjcf/free_wuji_right_hand_full_simplified_v1.xml"
        model = ET.parse(path).getroot()
        for asset in model.findall("./asset/*"):
            if asset.get("file"):
                asset.set("file", (path.parent / asset.get("file")).resolve().as_posix())
        root = model.find("worldbody/body[@name='root']")
        root.insert(0, ET.Element("freejoint", name="wrist_free"))
        for body in root.iter("body"):
            body.set("gravcomp", "1")
    else:
        model = ET.Element("mujoco", model="regrind")
        ET.SubElement(model, "compiler", angle="radian", balanceinertia="true")
        ET.SubElement(model, "option")
        ET.SubElement(model, "asset")
        ET.SubElement(model, "worldbody")
        add_urdf(model, ASSETS / "leaphand_urdf/leap_hand_right.urdf", "root", "wrist_free", True)
        actuators = ET.SubElement(model, "actuator")
        for name in LEAP_JOINTS:
            joint = next(j for j in model.iter("joint") if j.get("name") == name)
            ET.SubElement(actuators, "position", name=f"{name}_actuator", joint=name, kp="3", kv="0.1", forcelimited="true", forcerange="-0.5 0.5", ctrllimited="true", ctrlrange=joint.get("range"))
    model.find("option").set("timestep", str(1 / 120))
    model.find("option").set("integrator", "implicitfast")
    suffix = "_2x" if object_name == "scissors" else "_1.5x"
    filename = object_name + (suffix if robot == "leaphand" else "") + ".urdf"
    add_urdf(model, ASSETS / object_name / filename, object_name, f"{object_name}_free")
    height = 0.952 if object_name == "scissors" else (0.507 if robot == "leaphand" else 0.012)
    ET.SubElement(model.find("worldbody"), "geom", name="table", type="plane", pos=f"0 0 {height}", size="1 1 0.1", friction="1 0.005 0.0001", rgba="0.35 0.35 0.38 1")
    return ET.tostring(model, encoding="unicode")
