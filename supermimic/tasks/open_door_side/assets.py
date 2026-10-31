"""Reject references with a different contact convention before side training."""

from motion_data import load_motion

from ..open_door.assets import select_motion_files as select_door_motion_files


def select_motion_files(cfg):
    files = select_door_motion_files(cfg)
    for path in files:
        region = load_motion(path).get("door_contact_region")
        if region is None or region.shape != () or region.item() != "right_side":
            raise ValueError(
                f"{path}: side-opening references require scalar "
                "door_contact_region='right_side'; back or unlabelled clips cannot be mixed in"
            )
    return files
