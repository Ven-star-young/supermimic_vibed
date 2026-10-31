"""Measured door-contact diagnostics, independent of training task heads."""

import torch


def back_contact_region(local_points, back_geom):
    """Only torso/logo contacts inside the rear torso patch are legal."""
    x, y, z = local_points.unbind(-1)
    return back_geom & (x < -.025) & (y.abs() < .13) & (z > .05) & (z < .38)


class DoorContactMetrics:
    """Sample solved contact forces before observation-time forward replaces them.

    A category is active when its summed normal force exceeds 1 N in a world.
    This establishes measured contact, not which contact caused opening torque.
    """

    def __init__(self, native):
        self.sim = native.sim
        model, device = self.sim.mj_model, native.device
        self.torso = model.body("robot/torso_link").id
        self.door_geom = model.geom("door/door_panel").id
        bodies = torch.as_tensor(model.geom_bodyid.copy(), device=device)
        self.back_geom = (bodies == self.torso) | (bodies == model.body("robot/logo_link").id)
        self.robot_geom = torch.zeros(model.ngeom, dtype=torch.bool, device=device)
        self.robot_geom[native.scene["robot"].indexing.geom_ids] = True
        self.alive = torch.zeros(native.num_envs, dtype=torch.bool, device=device)
        self.samples = torch.zeros(native.num_envs, dtype=torch.long, device=device)
        self.contact_steps = torch.zeros(native.num_envs, 2, dtype=torch.long, device=device)
        self.ever_forbidden = torch.zeros_like(self.alive)
        self._original_step, self._closed = self.sim.step, False

        def measured_step():
            self._original_step()
            if bool(self.alive.any()):
                self._capture_physics()

        self._measured_step = measured_step
        self.sim.step = measured_step

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.sim.step is self._measured_step:
            self.sim.step = self._original_step
        self.alive.zero_()
        self._closed = True

    def begin_step(self, alive):
        if self._closed:
            raise RuntimeError("Door contact metrics are closed")
        self.alive.copy_(alive)

    @torch.no_grad()
    def _capture_physics(self):
        import mujoco_warp as mjwarp
        import warp as wp

        data = self.sim.data
        if bool(data.overflow.any()):
            raise RuntimeError("Physics buffer overflow invalidates door contact evaluation")
        totals = torch.zeros_like(self.contact_steps, dtype=torch.float32)
        count = int(data.nacon.item())
        contact = data.contact
        geom = contact.geom[:count].long()
        worlds = contact.worldid[:count].long()
        robot = torch.where(geom[:, 0] == self.door_geom, geom[:, 1], geom[:, 0]).clamp_min(0)
        valid = ((geom >= 0).all(-1) & (contact.efc_address[:count, 0] >= 0)
                 & (geom == self.door_geom).any(-1) & self.robot_geom[robot] & self.alive[worlds])
        ids = valid.nonzero(as_tuple=False).flatten()
        if ids.numel():
            forces = torch.zeros(len(ids), 6, device=self.alive.device)
            with wp.ScopedDevice(self.sim.wp_device):
                mjwarp.contact_force(self.sim.wp_model, self.sim.wp_data,
                                     wp.from_torch(ids.to(torch.int32)), False,
                                     wp.from_torch(forces, dtype=wp.spatial_vector))
            worlds = worlds[ids]
            rotation = data.xmat[worlds, self.torso].reshape(-1, 3, 3)
            local = (rotation.transpose(-1, -2) @
                     (contact.pos[ids] - data.xpos[worlds, self.torso]).unsqueeze(-1)).squeeze(-1)
            if not bool(torch.isfinite(forces).all() & torch.isfinite(local).all()):
                raise FloatingPointError("Nonfinite door contact force or position")
            allowed = back_contact_region(local, self.back_geom[robot[ids]])
            normal = forces[:, 0].clamp_min(0)
            totals[:, 0].scatter_add_(0, worlds, normal * allowed)
            totals[:, 1].scatter_add_(0, worlds, normal * ~allowed)
        active = (totals > 1.) & self.alive[:, None]
        self.samples += self.alive
        self.contact_steps += active
        self.ever_forbidden |= active[:, 1]

    def report(self):
        rates = self.contact_steps / self.samples.clamp_min(1)[:, None]
        totals = self.contact_steps.sum(0) / self.samples.sum().clamp_min(1)
        return dict(
            source="mujoco_warp.contact_force immediately after each physics step",
            scope="executed_prefix_only; normal force summed per category and world",
            normal_force_threshold_n=1.,
            back_region_torso_m=dict(x_max=-.025, abs_y_max=.13, z_min=.05, z_max=.38),
            allowed_back_substep_rate=float(totals[0]), forbidden_door_substep_rate=float(totals[1]),
            n_episodes_with_forbidden_contact=int(self.ever_forbidden.sum()),
            per_env=[dict(observed_substeps=int(self.samples[i]),
                          allowed_back_substep_rate=float(rates[i, 0]),
                          forbidden_door_substep_rate=float(rates[i, 1]),
                          ever_forbidden_contact=bool(self.ever_forbidden[i]))
                     for i in range(len(self.samples))])
