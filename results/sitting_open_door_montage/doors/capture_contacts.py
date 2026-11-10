"""Capture a trained policy's solved MuJoCo contacts without changing task code."""
from pathlib import Path
import argparse
from dataclasses import asdict
import hashlib
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from supermimic.scripts.evaluate import _policy_config, _restore_open_door_config, evaluate
from supermimic.tasks.open_door.evaluation import DoorContactMetrics
from supermimic.tasks.open_door_side.evaluation import SideDoorContactMetrics


class CaptureMixin:
    def __init__(self, native, out):
        super().__init__(native)
        self.out = out
        self.qpos_frames, self.contact_frames = [], []
        self.frame_times = []
        self.body_names = [self.sim.mj_model.body(i).name for i in range(self.sim.mj_model.nbody)]
        self.geom_bodies = torch.as_tensor(self.sim.mj_model.geom_bodyid.copy(), device=native.device)
        self.dt = float(self.sim.mj_model.opt.timestep)
        mujoco.mj_saveModel(self.sim.mj_model, str(out / "scene.mjb"))

    def begin_step(self, alive):
        super().begin_step(alive)
        if not self.qpos_frames:
            self.qpos_frames.append(self.sim.data.qpos[0].cpu().numpy().copy())
            self.frame_times.append(0.0)
            self.contact_frames.append([])
            self.mocap_pos = self.sim.data.mocap_pos[0].cpu().numpy().copy()
            self.mocap_quat = self.sim.data.mocap_quat[0].cpu().numpy().copy()

    @torch.no_grad()
    def _capture_physics(self):
        super()._capture_physics()
        import mujoco_warp as mjwarp
        import warp as wp
        data = self.sim.data
        count = int(data.nacon.item())
        contact = data.contact
        geom = contact.geom[:count].long()
        worlds = contact.worldid[:count].long()
        robot = torch.where(geom[:, 0] == self.door_geom, geom[:, 1], geom[:, 0]).clamp_min(0)
        valid = ((geom >= 0).all(-1) & (contact.efc_address[:count, 0] >= 0)
                 & (geom == self.door_geom).any(-1) & self.robot_geom[robot] & self.alive[worlds])
        ids = valid.nonzero(as_tuple=False).flatten()
        records = []
        if ids.numel():
            forces = torch.zeros(len(ids), 6, device=self.alive.device)
            with wp.ScopedDevice(self.sim.wp_device):
                mjwarp.contact_force(self.sim.wp_model, self.sim.wp_data,
                    wp.from_torch(ids.to(torch.int32)), False,
                    wp.from_torch(forces, dtype=wp.spatial_vector))
            rotation = data.xmat[worlds[ids], self.torso].reshape(-1, 3, 3)
            local = (rotation.transpose(-1, -2) @
                (contact.pos[ids] - data.xpos[worlds[ids], self.torso]).unsqueeze(-1)).squeeze(-1)
            allowed = self.classify_contacts(local, robot[ids])
            pos = contact.pos[ids].cpu().numpy()
            normals = contact.frame[ids].reshape(-1, 3, 3)[:, 0].cpu().numpy()
            bodies = self.geom_bodies[robot[ids]].cpu().numpy()
            force_np = forces.cpu().numpy()
            allowed_np = allowed.cpu().numpy()
            for j in range(len(ids)):
                normal_force = max(0., float(force_np[j, 0]))
                if normal_force > .01:
                    records.append(dict(position=pos[j].tolist(), normal=normals[j].tolist(),
                        normal_force_n=normal_force, allowed=bool(allowed_np[j]),
                        body_id=int(bodies[j]), body=self.body_names[int(bodies[j])]))
        self.qpos_frames.append(data.qpos[0].cpu().numpy().copy())
        self.frame_times.append((len(self.qpos_frames)-1) * self.dt)
        self.contact_frames.append(records)

    def save(self):
        np.savez_compressed(self.out / "physics_states.npz", qpos=np.asarray(self.qpos_frames),
            time=np.asarray(self.frame_times), mocap_pos=self.mocap_pos, mocap_quat=self.mocap_quat)
        (self.out / "measured_contacts.json").write_text(json.dumps(dict(
            source="mujoco_warp.contact_force immediately after each physical simulation step",
            geometry_source="saved native simulation MJB; no contact recomputation used for markers",
            dt=self.dt, body_names=self.body_names, frames=self.contact_frames), separators=(",", ":")))


class BackCapture(CaptureMixin, DoorContactMetrics):
    pass


class SideCapture(CaptureMixin, SideDoorContactMetrics):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("back", "side"))
    args = parser.parse_args()
    out = Path(__file__).resolve().parent / args.kind
    out.mkdir(parents=True, exist_ok=True)
    if args.kind == "back":
        checkpoint_file = Path("logs/wm_transfer/back_model_10000.pt")
        motion = Path("data/motions/open_door_prepared/video1_g1_door_video45_centered_asset.npz")
        task = "Mjlab-SuperMimic-OpenDoor-Unitree-G1"
    else:
        checkpoint_file = Path("logs/side_policy_playback/2026-11-01_01-14-42_fresh_ac_model10000/model_10000_actor.pt")
        motion = Path("data/motions/open_door_prepared/video6_g1_door_video45_right_side_asset.npz")
        task = "Mjlab-SuperMimic-OpenDoorSide-Unitree-G1"
    torch.manual_seed(0)
    np.random.seed(0)
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    policy_cfg = _policy_config(checkpoint)
    cfg = load_env_cfg(task, play=True)
    cfg.auto_reset = False
    _restore_open_door_config(cfg, checkpoint)
    cfg.commands["motion"].motion_dir = str(motion)
    cfg.commands["motion"].history_length = policy_cfg["history_length"]
    cfg.commands["motion"].goal_frames = policy_cfg["goal_frames"]
    cfg.scene.num_envs = 1
    native = ManagerBasedRlEnv(cfg, device="cuda:0")
    metrics = None
    try:
        metrics = (BackCapture if args.kind == "back" else SideCapture)(native, out)
        env = RslRlVecEnvWrapper(native)
        runner = load_runner_cls(task)(env, asdict(load_rl_cfg(task)), device="cuda:0")
        runner.load(str(checkpoint_file), load_cfg={"actor": True}, map_location="cuda:0")
        result = evaluate(env, runner, out / "rollout.npz", metrics)
        metrics.save()
        result.update(checkpoint=str(checkpoint_file.resolve()), checkpoint_iteration=checkpoint["iteration"],
            motion=str(motion.resolve()), motion_sha256=hashlib.sha256(motion.read_bytes()).hexdigest(),
            source_motion_py_sha256=hashlib.sha256(Path("supermimic/tasks/open_door/motion.py").read_bytes()).hexdigest(),
            physics_samples=len(metrics.qpos_frames), physical_duration_s=metrics.frame_times[-1])
        (out / "evaluation.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        if metrics is not None:
            metrics.close()
        native.close()


if __name__ == "__main__":
    main()
