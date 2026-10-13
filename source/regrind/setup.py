"""Installation script for the 'regrind' python package."""

import os
import toml

from setuptools import setup, find_packages

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mujoco>=3.2,<4",
    "gymnasium>=1.0,<2",
    "numpy>=1.26",
    "scipy>=1.11",
    "h5py>=3.10",
    "psutil",
    "wandb==0.21.1",  # Note: temporary fix for compatibility with protobuf
                      # (https://github.com/isaac-sim/IsaacLab/issues/3138)
    "git-lfs",
]

EXTRAS_REQUIRE = {
    "train": ["stable-baselines3>=2.4,<3"],
    "retargeting": ["drake"],
    "mujoco": ["mujoco", "gymnasium", "stable-baselines3"],
}

# Installation operation
setup(
    name="regrind",
    packages=find_packages(),
    package_data={"regrind": ["assets/**/*"]},
    entry_points={"console_scripts": ["regrind=regrind.mujoco.cli:main"]},
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    license="MIT",
    include_package_data=True,
    python_requires=">=3.11",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.11",
    ],
    zip_safe=False,
)
