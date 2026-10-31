"""Train right-side door tracking with its own task and editable JSON config."""

from pathlib import Path

from .train_open_door import main as train_main


TASK_ID = "Mjlab-SuperMimic-OpenDoorSide-Unitree-G1"
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/open_door_right_side.json"


def main(argv=None):
    train_main(argv, task_id=TASK_ID, default_config=DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
