"""Right-side opening scene using the shared G1 and passive-door physics."""

from pathlib import Path

from ..open_door.env_cfgs import DoorEnvCfg, make_env_cfg as make_door_env_cfg
from .commands import SideDoorCommandCfg


DEFAULT_MOTION_FILE = "video6_g1_door_video45_right_side_asset.npz"


def make_env_cfg(play=False):
    baseline = make_door_env_cfg(play)
    values = vars(baseline.commands["motion"]).copy()
    values.pop("door_reward_cfg")
    command = SideDoorCommandCfg(**values)
    path = Path(command.motion_dir)
    command.motion_dir = str(path if path.suffix == ".npz" else path / DEFAULT_MOTION_FILE)
    baseline.commands["motion"] = command
    baseline.episode_length_s = 15.0
    # Reconstruct so the door asset factory binds to the side command too.
    return DoorEnvCfg(**vars(baseline))
