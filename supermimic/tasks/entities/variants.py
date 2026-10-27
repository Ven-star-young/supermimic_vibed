"""Fixed per-environment chair geometry using mjlab's native mesh variants.

All worlds share one robot and chair topology. Chair collision meshes vary;
unused mesh slots are disabled by mjlab's per-world ``geom_dataid = -1``.
"""

from functools import partial

import mujoco
import numpy as np
from mjlab.entity import EntityCfg, VariantEntityCfg

from .assets import make_object_spec


def make_variant_chair_spec(command):
    """Keep robot-contact and floor-contact mesh slots structurally distinct.

    mjlab pads variable mesh counts by (body, visual/collision, ordinal). Its
    contact masks are shared by all worlds, so the two contact roles cannot
    occupy different ordinals on the same body when chair part counts differ.
    A massless welded child holds floor copies, preserving rigid dynamics.
    """
    if not command.object_enabled:
        raise ValueError("Chair variants require object_enabled=True")
    spec = make_object_spec(command, 0)
    root = spec.body("chair")
    floor = root.add_body(name="chair_ground")
    for geom in list(root.geoms):
        if geom.contype != 0 or geom.conaffinity != 1:
            continue
        fields = ("name", "type", "meshname", "pos", "quat", "group", "contype",
                  "conaffinity", "priority", "condim", "friction", "rgba", "margin",
                  "gap", "solref", "solimp", "solmix")
        values = {field: getattr(geom, field) for field in fields}
        # Copy array views before removing their owning MjsGeom.
        values = {key: value.copy() if isinstance(value, np.ndarray) else value
                  for key, value in values.items()}
        spec.delete(geom)
        floor.add_geom(**values, mass=0, density=0)
    if not list(floor.geoms):
        raise ValueError("Chair variant has no floor collision mesh")
    return spec


def make_chair_variant_cfg(commands, assignment=None):
    """Create ordered variants; assignment indices refer to ``commands`` order.

    Assignment is fixed at simulation construction. The default allocates
    environments uniformly (counts differ by at most one), without changing
    geometry at episode reset.
    """
    if not commands:
        raise ValueError("At least one chair variant is required")
    return VariantEntityCfg(
        variants={f"scene_{index:04d}": partial(make_variant_chair_spec, command)
                  for index, command in enumerate(commands)},
        assignment=assignment,
        init_state=EntityCfg.InitialStateCfg(pos=(0., 0., 2.)),
    )


def normalize_variant_collision_slots(spec, *, entity_name="chair", robot_friction=1.,
                                      ground_friction=1.):
    """Restore source contact roles on mjlab's synthesized padding slots.

    Call from the assembled scene callback *before* compiling/sizing buffers.
    mjlab 1.6 initializes padded collision slots with generic 1/1 contact bits;
    actual slot semantics here are shared across every source variant.
    """
    prefix = f"{entity_name}/mjlab/pad/"
    for geom in spec.geoms:
        if not geom.name.startswith(prefix):
            continue
        if "/visual/" in geom.name:
            geom.contype, geom.conaffinity, geom.group = 0, 0, 1
            continue
        is_floor = "/chair_ground/" in geom.name
        geom.contype, geom.conaffinity = (0, 1) if is_floor else (2, 2)
        geom.priority, geom.group, geom.condim = (0 if is_floor else 1), 3, 3
        geom.friction = [ground_friction if is_floor else robot_friction, .005, .0001]
        geom.mass, geom.density = 0., 0.
