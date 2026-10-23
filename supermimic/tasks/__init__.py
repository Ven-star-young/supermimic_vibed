"""Explicit task selection. Add each implemented task here with its own schema."""


def load_task(name, motion_dir=None, *, device="cuda:0", play=False, **kwargs):
    if name != "g1_tracking":
        raise ValueError(f"Unknown task {name!r}; available tasks: g1_tracking")
    from .g1_tracking.task import build_task
    return build_task(motion_dir, device=device, play=play, **kwargs)


def get_train_cfg(name):
    if name != "g1_tracking":
        raise ValueError(f"Unknown task {name!r}; available tasks: g1_tracking")
    from .g1_tracking.rl_cfg import make_train_cfg
    return make_train_cfg()
