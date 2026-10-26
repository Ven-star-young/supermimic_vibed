"""Multi-entity task entry points; all files/assets are loaded at runtime."""

from mjlab.tasks.registry import register_mjlab_task

from ...rl.runner import SuperMimicRunner
from .env_cfgs import make_env_cfg
from .rl_cfg import make_runner_cfg


for _name, _enabled in (("EntityTracking", False), ("EntitySitting", True)):
    register_mjlab_task(
        task_id=f"Mjlab-SuperMimic-{_name}-Unitree-G1",
        env_cfg=make_env_cfg(object_enabled=_enabled),
        play_env_cfg=make_env_cfg(play=True, object_enabled=_enabled),
        rl_cfg=make_runner_cfg(), runner_cls=SuperMimicRunner,
    )
