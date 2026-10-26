"""Portable real transitions for world-model training across tasks and scenes."""
from pathlib import Path

import torch

from .shared_world_model import SharedDynamics

FIELDS = ("s", "s1", "a", "xi", "done")


def save_replay(path, buffer, geometry_bank):
    if buffer.size == 0:
        raise ValueError("Cannot export empty replay")
    rows = buffer._row(torch.arange(buffer.size, device=buffer.device))
    data = {key: buffer.data[key][rows].detach().cpu() for key in FIELDS}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(schema=SharedDynamics.schema(), transitions=data,
                    geometry=geometry_bank.state_dict()), temporary)
    temporary.replace(path)


class SharedReplay:
    """Keep archive episode boundaries and remap scene IDs to the current bank."""

    def __init__(self, paths, geometry_bank):
        self.archives = []
        for path in paths:
            archive = torch.load(path, map_location="cpu", weights_only=True)
            if archive.get("schema") != SharedDynamics.schema():
                raise ValueError(f"Incompatible world-model replay schema: {path}")
            data = archive["transitions"]
            if set(data) != set(FIELDS) or data["s"].ndim != 3:
                raise ValueError(f"Invalid replay fields: {path}")
            time, envs = data["s"].shape[:2]
            for key, width in (("s", 84), ("s1", 84), ("a", 29), ("xi", 47)):
                value = data[key]
                if value.shape != (time, envs, width) or not torch.isfinite(value).all():
                    raise ValueError(f"Invalid {key} in replay: {path}")
            if data["done"].shape != (time, envs) or data["done"].dtype != torch.bool:
                raise ValueError(f"Invalid episode boundaries in replay: {path}")
            presence = data["xi"][..., 45]
            if not ((presence == 0) | (presence == 1)).all():
                raise ValueError(f"Invalid object presence mask in replay: {path}")
            old_to_new = geometry_bank.merge(archive["geometry"])
            ids = data["xi"][..., 46]
            if not torch.equal(ids, ids.round()) or (ids < 0).any() or (ids >= len(old_to_new)).any():
                raise ValueError(f"Invalid geometry index in replay: {path}")
            if not torch.equal(ids > 0, presence.bool()):
                raise ValueError(f"Geometry/presence mismatch in replay: {path}")
            data["xi"][..., 46] = old_to_new[ids.long()].float()
            self.archives.append(data)
        if not self.archives:
            raise ValueError("Shared replay requires at least one archive")

    def sample_windows(self, batch, length, device):
        eligible = [d for d in self.archives if len(d["s"]) >= length]
        if not eligible:
            raise ValueError(f"No replay archive contains {length} steps")
        # Uniform archive sampling prevents a large source from erasing a small
        # task's dynamics. No sequence crosses an archive or environment boundary.
        choices = torch.randint(len(eligible), (batch,))
        chunks = []
        for index in choices.unique().tolist():
            data = eligible[index]
            count = int((choices == index).sum())
            start = torch.randint(len(data["s"]) - length + 1, (count,))
            env = torch.randint(data["s"].shape[1], (count,))
            rows = start[:, None] + torch.arange(length)
            chunks.append({key: value[rows, env[:, None]] for key, value in data.items()})
        result = {key: torch.cat([chunk[key] for chunk in chunks]).to(device) for key in FIELDS}
        done = result["done"].float()
        result["mask"] = torch.cumprod(torch.cat((torch.ones(batch, 1, device=device), 1 - done[:, :-1]), -1), -1)
        return result
