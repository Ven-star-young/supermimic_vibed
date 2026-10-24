"""Real G1 asset, generated motion CSVs and native mjlab integration fixtures."""

from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="session")
def asset_path():
    from supermimic.assets.g1.g1_constants import MJCF_PATH
    path = Path(MJCF_PATH)
    if not path.is_file():
        pytest.skip("G1 assets are missing; run tools/download_data.sh")
    return path


@pytest.fixture(scope="session")
def cpu_kin(asset_path):
    from supermimic.assets.g1.kinematics import G1Kinematics
    return G1Kinematics("cpu")


@pytest.fixture(scope="session")
def motion_dir(cpu_kin):
    from tools._smoke import motion_directory
    with motion_directory(cpu_kin) as directory:
        yield directory


@pytest.fixture(scope="session")
def cpu_motion(cpu_kin, motion_dir):
    from supermimic.motion import MotionLib
    return MotionLib.from_dir(motion_dir, cpu_kin, "cpu")


@pytest.fixture(scope="session")
def cuda_device(asset_path):
    if not torch.cuda.is_available():
        pytest.skip("mjlab integration tests require CUDA")
    pytest.importorskip("mujoco_warp")
    return "cuda:0"


@pytest.fixture(scope="module")
def native_eval_env(cuda_device, motion_dir):
    from tools._smoke import native_env
    with native_env(4, device=cuda_device, motion_dir=motion_dir, auto_reset=False) as env:
        yield env


@pytest.fixture(scope="module")
def eval_env(native_eval_env):
    from supermimic.rl.vecenv import ReplayEnvAdapter
    return ReplayEnvAdapter(native_eval_env)


@pytest.fixture(scope="module")
def train_env(cuda_device, motion_dir):
    from supermimic.rl.vecenv import ReplayEnvAdapter
    from tools._smoke import native_env
    with native_env(4, device=cuda_device, motion_dir=motion_dir, play=False, auto_reset=False) as env:
        yield ReplayEnvAdapter(env)
