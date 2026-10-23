"""Shared fixtures: real G1 model, generated CSV, and a four-world mjlab scene."""

from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="session")
def asset_path():
    from supermimic.robot import MJCF_PATH

    path = Path(MJCF_PATH)
    if not path.is_file():
        pytest.skip("G1 assets are missing; run tools/download_data.sh")
    return path


@pytest.fixture(scope="session")
def cpu_kin(asset_path):
    from supermimic.kinematics import G1Kinematics

    return G1Kinematics("cpu")


@pytest.fixture(scope="session")
def cpu_motion(cpu_kin):
    from tools._smoke import load_motion

    return load_motion(cpu_kin, "cpu")


@pytest.fixture(scope="module")
def gpu_components(asset_path):
    if not torch.cuda.is_available():
        pytest.skip("mjlab integration tests require CUDA")
    pytest.importorskip("mujoco_warp")
    from supermimic.kinematics import G1Kinematics
    from supermimic.mdp import MDP
    from supermimic.robot import default_joint_pos_list
    from tools._smoke import load_motion

    device = "cuda:0"
    kin = G1Kinematics(device)
    motion = load_motion(kin, device)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=device))
    return kin, motion, mdp


@pytest.fixture(scope="module")
def eval_env(gpu_components):
    from supermimic.env import TrackingSim

    _, motion, mdp = gpu_components
    with TrackingSim(4, motion, mdp, device="cuda:0", eval_mode=True) as env:
        yield env


@pytest.fixture(scope="module")
def train_env(gpu_components):
    from supermimic.env import TrackingSim

    _, motion, mdp = gpu_components
    with TrackingSim(4, motion, mdp, device="cuda:0", reset_noise=False) as env:
        yield env
