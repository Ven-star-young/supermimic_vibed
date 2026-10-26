"""Register SuperMimic with mjlab's stock train/play commands."""
from mjlab.tasks.registry import register_mjlab_task

from ...rl.runner import SuperMimicRunner
from .env_cfgs import make_env_cfg
from .rl_cfg import make_supermimic_runner_cfg

register_mjlab_task(
    task_id="Mjlab-SuperMimic-Sitting-Unitree-G1",
    env_cfg=make_env_cfg(),
    play_env_cfg=make_env_cfg(play=True),
    rl_cfg=make_supermimic_runner_cfg(),
    runner_cls=SuperMimicRunner,
)
