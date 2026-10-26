"""Robot-only tracking using the same dynamics schema as object interaction."""
from mjlab.tasks.registry import register_mjlab_task

from ..rl.runner import SuperMimicRunner
from .g1_tracking.rl_cfg import make_supermimic_runner_cfg
from .sitting.env_cfgs import make_env_cfg

register_mjlab_task(
    task_id="Mjlab-SuperMimic-SharedTracking-Unitree-G1",
    env_cfg=make_env_cfg(object_enabled=False),
    play_env_cfg=make_env_cfg(play=True, object_enabled=False),
    rl_cfg=make_supermimic_runner_cfg(),
    runner_cls=SuperMimicRunner,
)
