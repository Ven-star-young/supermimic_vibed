"""Portable entity-physics replay with per-slot geometry remapping."""
from pathlib import Path

import torch

from .entity_world_model import EntityGeometryBank
from .shared_replay import FIELDS, SharedReplay


def save_entity_replay(path, buffer, geometry_bank, dynamics):
    if not buffer.size:
        raise ValueError("Cannot export empty replay")
    rows = buffer._row(torch.arange(buffer.size, device=buffer.device))
    data = {key: buffer.data[key][rows].detach().cpu() for key in FIELDS}
    dynamics.validate_context(data["xi"], len(geometry_bank.keys))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(schema=dynamics.schema(), transitions=data,
                    geometry=geometry_bank.state_dict()), temporary)
    temporary.replace(path)


class EntityReplay(SharedReplay):
    """Reuse archive/window balancing; never reinterpret one-body archives."""

    def __init__(self, paths, geometry_bank, dynamics):
        self.archives = []
        for path in paths:
            archive = torch.load(path, map_location="cpu", weights_only=True)
            if archive.get("schema") != dynamics.schema():
                raise ValueError(f"Incompatible entity replay schema: {path}")
            data = archive["transitions"]
            if set(data) != set(FIELDS) or data["s"].ndim != 3:
                raise ValueError(f"Invalid entity replay fields: {path}")
            time, envs = data["s"].shape[:2]
            if min(time, envs) < 1:
                raise ValueError("Empty archive")
            for key, width in (("s", dynamics.state_dim), ("s1", dynamics.state_dim),
                               ("a", 29), ("xi", dynamics.context_dim)):
                value = data[key]
                if value.shape != (time, envs, width) or not torch.isfinite(value).all():
                    raise ValueError(f"Invalid {key} in entity replay: {path}")
            if data["done"].shape != (time, envs) or data["done"].dtype != torch.bool:
                raise ValueError("Invalid replay episode boundaries")
            incoming = EntityGeometryBank.from_dict(archive["geometry"])
            dynamics.validate_context(data["xi"], len(incoming.keys))
            objects = dynamics.object_context(data["xi"])
            # A slot is an identity across a rollout; resets may select a new scene.
            changed = (objects[1:, ..., 10:13] != objects[:-1, ..., 10:13]).any(-1).any(-1)
            if (changed & ~data["done"][:-1]).any():
                raise ValueError("Entity identities changed inside an episode")
            ids = objects[..., 12].long()
            mapping = geometry_bank.merge(archive["geometry"])
            objects[..., 12] = mapping[ids].float()
            self.archives.append(data)
        if not self.archives:
            raise ValueError("Entity replay requires at least one archive")
