"""Audit a real multi-chair training run without changing its update rules.

Use the same arguments as supermimic.scripts.chair_experiment, under torchrun:
  python -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m tools.check_distributed_training train --split ... --run-dir ...

Every saved checkpoint gets a .audit.json with per-rank model/optimizer hashes,
finite-value checks, memory peaks and visited scenes. Hashing synchronizes CUDA
and copies tensors to CPU; this diagnostic is not a throughput benchmark.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from supermimic.rl import distributed
from supermimic.rl.chair_experiment import ChairExperimentRunner


def tensor_digest(values):
    digest = hashlib.sha256()
    finite = True
    for name, tensor in values:
        tensor = tensor.detach().cpu().contiguous()
        finite = finite and bool(torch.isfinite(tensor).all())
        digest.update(str((name, tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return dict(sha256=digest.hexdigest(), finite=finite)


def optimizer_digest(optimizer):
    state = optimizer.state_dict()
    tensors = [(f"{index}.{name}", value)
               for index, fields in sorted(state["state"].items())
               for name, value in sorted(fields.items()) if isinstance(value, torch.Tensor)]
    return dict(**tensor_digest(tensors), param_groups=state["param_groups"],
                initialized_parameters=len(state["state"]))


class AuditedChairRunner(ChairExperimentRunner):
    def save(self, path, infos=None):
        signatures = dict(
            models={name: tensor_digest(module.state_dict().items()) for name, module in
                    (("actor", self.pi), ("wm", self.wm), ("critic", self.critic))},
            optimizers={name: optimizer_digest(optimizer) for name, optimizer in
                        (("actor", self.opt_pi), ("wm", self.opt_wm), ("critic", self.opt_v))},
            iteration=self.current_learning_iteration, sim_steps=self.sim_steps)
        reports = distributed.gather_objects(signatures)
        if any(report != reports[0] for report in reports[1:]):
            raise AssertionError("Model, normalizer or optimizer state diverged between ranks")
        if not all(value["finite"] for section in ("models", "optimizers")
                   for value in signatures[section].values()):
            raise AssertionError("Nonfinite model/normalizer/optimizer state")
        record = dict(rank=distributed.rank(), device=str(self.device),
                      backend=torch.distributed.get_backend() if distributed.active() else None,
                      scene_visits=self.visit_log, visited_shapes=sorted(self.visited_shapes),
                      state=signatures)
        if self.device.type == "cuda":
            record["peak_torch_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device)
            record["peak_torch_reserved_bytes"] = torch.cuda.max_memory_reserved(self.device)
        if self.buffer is not None:
            rows = self.buffer._row(torch.arange(self.buffer.size, device=self.device))
            record["replay"] = tensor_digest((key, value[rows]) for key, value in self.buffer.data.items())
        records = distributed.gather_objects(record)
        super().save(path, infos)
        if distributed.primary():
            Path(path).with_suffix(".audit.json").write_text(json.dumps(dict(
                synchronized=True, finite=True, ranks=records), indent=2) + "\n")
            print(f"[distributed-audit] {path}: all {len(records)} ranks finite and synchronized", flush=True)


def main():
    from supermimic.scripts import chair_experiment
    chair_experiment.ChairExperimentRunner = AuditedChairRunner
    chair_experiment.main()


if __name__ == "__main__":
    main()
