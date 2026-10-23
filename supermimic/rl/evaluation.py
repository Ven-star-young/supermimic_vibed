"""Checkpoint evaluation (paper Appendix F.1) through a registered task.

Each motion starts at its first frame without domain randomization or observation/action
noise. A motion fails under the training termination condition; return normalization is
provided by the task. Importing this module performs no evaluation or argument parsing.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from supermimic.tasks import load_task
from . import models as networks
from .config import REPO
from .models import Policy


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate a SuperMimic checkpoint.")
    parser.add_argument("--task", default="g1_tracking")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None, help="evaluate one checkpoint")
    parser.add_argument("--record", type=str, default=None, help="save G1 trajectories (npz) for rendering")
    parser.add_argument("--motion_dir", type=str, default=str(REPO / "data" / "lafan1_g1"))
    parser.add_argument("--every", type=int, default=1, help="evaluate every n-th checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser


@torch.no_grad()
def evaluate(env, ck: dict, task, record: str | None = None) -> dict:
    """Evaluate a checkpoint against a task; recording currently supports G1 tracking."""
    task.validate_checkpoint(ck)
    if record and task.name != "g1_tracking":
        raise ValueError("Trajectory recording requires a recorder for this task; only g1_tracking is supported.")
    dev = env.device
    motion, mdp, dims = task.motion, task.mdp, task.dimensions
    C = len(motion.names)
    a_ = ck["args"]
    K, Hh = a_["goal_frames"], a_["hist"]
    networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
    pi = Policy(dims.observation, task.goal_dim(K), hist=Hh, action_dim=dims.action).to(dev)
    pi.load_state_dict(ck["pi"])
    pi.eval()
    ids = torch.arange(C, device=dev)
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset(ids, motion.clip_start.clone())
    clip_end = motion.clip_end.clone()
    ret = torch.zeros(C, device=dev)
    steps = torch.zeros(C, device=dev)
    hist = None
    traj = {"root_p": [], "root_quat": [], "q": [], "f": [], "alive": []}
    while not bool(env.done_eval.all()):
        S = env.state()
        o = mdp.obs(S, env.a_prev, env.xi, env.f)
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        a = pi(hist, motion.goal(env.f, K))
        alive = ~env.done_eval.clone()
        if record:
            from supermimic.rotations import mat_to_quat

            traj["root_p"].append(S[0].cpu()), traj["root_quat"].append(mat_to_quat(S[1]).cpu())
            traj["q"].append(S[4].cpu()), traj["f"].append(env.f.cpu()), traj["alive"].append(alive.cpu())
        out = env.step(a)
        ret += torch.where(alive, out["r"], 0.0)
        steps += alive.float()
        env.f = torch.minimum(env.f, clip_end - 1)  # finished clips keep stepping; keep f+1 in range
    norm = ret / ((motion.clip_len - 1).float() * task.max_step_reward)
    res = {
        "norm_return": norm.mean().item(),
        "norm_return_framew": (ret.sum() / ((motion.clip_len - 1).sum() * task.max_step_reward)).item(),
        "n_failed": int(env.failed_eval.sum().item()),
        "failed": [motion.names[i] for i in env.failed_eval.nonzero().flatten().tolist()],
        "per_clip": {motion.names[i]: round(norm[i].item(), 4) for i in range(C)},
    }
    if record:
        np.savez_compressed(record, names=np.array(motion.names), **{k: torch.stack(v).numpy() for k, v in traj.items()},
                            ref_root_p=motion.root_p.cpu().numpy(), ref_root_R=motion.root_R.cpu().numpy(), ref_q=motion.q.cpu().numpy(),
                            clip_start=motion.clip_start.cpu().numpy(), clip_len=motion.clip_len.cpu().numpy())  # fmt: skip
    return res


def main(argv=None):
    return run(build_parser().parse_args(argv))


def run(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = args.device
    task = load_task(args.task, args.motion_dir, device=dev, play=True)
    C = len(task.motion.names)
    with task.make_env(C, device=dev) as env:
        if args.ckpt:
            ck = torch.load(args.ckpt, map_location=dev)
            res = evaluate(env, ck, task, args.record)
            print(json.dumps({k: v for k, v in res.items() if k != "per_clip"}, indent=1))
            return res
        out_csv = os.path.join(args.run_dir, "eval.csv")
        done = set()
        if os.path.exists(out_csv):
            done = {int(l.split(",")[0]) for l in open(out_csv).read().strip().splitlines()[1:]}
        else:
            open(out_csv, "w").write("iteration,train_wall_h,saved_at,eval_at,norm_return,norm_return_framew,n_failed,failed\n")
        while True:
            ckpts = sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.pt")))
            todo = [c for c in ckpts if int(os.path.basename(c)[5:11]) not in done]
            todo = [c for c in todo if (int(os.path.basename(c)[5:11]) // 2000) % args.every == 0]
            for c in todo:
                it = int(os.path.basename(c)[5:11])
                ck = torch.load(c, map_location=dev)
                t0 = time.time()
                res = evaluate(env, ck, task)
                line = f"{it},{ck['wall_time'] / 3600:.4f},{ck['saved_at']},{time.strftime('%Y-%m-%d %H:%M:%S')},{res['norm_return']:.4f},{res['norm_return_framew']:.4f},{res['n_failed']},{'|'.join(res['failed'])}\n"
                open(out_csv, "a").write(line)
                json.dump(res, open(c.replace(".pt", "_eval.json"), "w"), indent=1)
                print(f"[eval] it {it} ({ck['wall_time'] / 3600:.2f} h): norm return {res['norm_return']:.4f}, failed {res['n_failed']}/{C} ({time.time() - t0:.0f}s)", flush=True)
                done.add(it)
            if not args.watch:
                break
            time.sleep(60)


if __name__ == "__main__":
    main()
