"""Synchronous data parallel updates for simulator-backed BPTT learning.

Explicit gradient reduction lets the world model/critic freeze during policy
imagination without attaching DDP hooks to shared geometry parameters twice.
"""
from __future__ import annotations

import atexit
from datetime import timedelta
import hashlib
import os

import torch
import torch.distributed as dist


def active():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def world_size():
    return dist.get_world_size() if dist.is_initialized() else int(os.environ.get("WORLD_SIZE", "1"))


def rank():
    return dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", "0"))


def primary():
    return rank() == 0


def local_device(device):
    device = torch.device(device)
    if world_size() > 1 and device.type == "cuda":
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        torch.cuda.set_device(device)
    return device


def shutdown():
    if dist.is_initialized():
        dist.destroy_process_group()


def initialize(device):
    device = local_device(device)
    if world_size() > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo",
                                init_method="env://", timeout=timedelta(minutes=10))
        atexit.register(shutdown)
    return device


def gather_objects(value):
    if not active():
        return [value]
    result = [None] * world_size()
    dist.all_gather_object(result, value)
    return result


def assert_same(value, description):
    values = gather_objects(value)
    if any(other != values[0] for other in values[1:]):
        raise ValueError(f"Distributed ranks disagree on {description}")


def bank_digest(bank):
    digest = hashlib.sha256()
    for key, points, extents in zip(bank.keys, bank.points, bank.extents):
        digest.update(key.encode())
        digest.update(points.detach().cpu().contiguous().numpy().tobytes())
        digest.update(extents.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def broadcast_modules(*modules):
    if not active():
        return
    seen = set()
    for module in modules:
        for tensor in (*module.parameters(), *module.buffers()):
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            dist.broadcast(tensor, src=0)


@torch.no_grad()
def global_moments(x, mask=None):
    """Return global per-feature counts, means and population variances.

    Empty ranks must participate. Two passes avoid cancellation in variance;
    reductions use float64 even when input/normalizer buffers use float32.
    """
    shape = x.shape
    values = x.reshape(-1, shape[-1]).double()
    valid = torch.ones_like(values, dtype=torch.bool) if mask is None else torch.broadcast_to(
        mask, shape).reshape_as(values).bool()
    values = torch.where(valid, values, 0.)
    count = valid.sum(0).double()
    totals = torch.stack((count, values.sum(0)))
    if active():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    count, sums = totals.unbind(0)
    mean = sums / count.clamp_min(1)
    deviations = torch.where(valid, (values - mean).square(), 0.).sum(0)
    if active():
        dist.all_reduce(deviations, op=dist.ReduceOp.SUM)
    return count.float(), mean.float(), (deviations / count.clamp_min(1)).float()


@torch.no_grad()
def average_gradients(optimizer, bucket_elements=4_000_000):
    """Mean gradients across equal-sized rank batches, before clipping.

    A parameter unused locally gets the other ranks' gradient. Globally unused
    parameters retain grad=None, so optimizer momentum cannot update them.
    All ranks reject nonfinite updates together.
    """
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    finite = torch.ones((), device=parameters[0].device, dtype=torch.int32)
    for parameter in parameters:
        if parameter.grad is not None:
            finite.mul_(torch.isfinite(parameter.grad).all().to(torch.int32))
    if active():
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not bool(finite):
        optimizer.zero_grad(set_to_none=True)
        return False
    if not active():
        return True
    used = torch.tensor([p.grad is not None for p in parameters], device=parameters[0].device,
                        dtype=torch.int32)
    dist.all_reduce(used, op=dist.ReduceOp.MAX)
    bucket, size = [], 0

    def flush():
        if not bucket:
            return
        flat = torch.cat([p.grad.reshape(-1) for p in bucket])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world_size())
        offset = 0
        for parameter in bucket:
            parameter.grad.copy_(flat[offset:offset + parameter.numel()].view_as(parameter))
            offset += parameter.numel()

    for parameter, present in zip(parameters, used.tolist()):
        if not present:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        if bucket and (size + parameter.numel() > bucket_elements or parameter.dtype != bucket[0].dtype):
            flush()
            bucket, size = [], 0
        bucket.append(parameter)
        size += parameter.numel()
    flush()
    return True


def require_all(condition, message, device):
    value = torch.tensor(int(condition), device=device)
    if active():
        dist.all_reduce(value, op=dist.ReduceOp.MIN)
    if not bool(value):
        raise RuntimeError(message)


def replay_path(path):
    from pathlib import Path
    path = Path(path)
    if world_size() > 1:
        return path.with_name(f"{path.stem}.rank{rank():04d}{path.suffix}")
    return path
