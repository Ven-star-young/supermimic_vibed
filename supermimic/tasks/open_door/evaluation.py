"""Measured door-contact diagnostics, independent of training task heads."""

import torch


def _rotation_error_sq(actual, reference):
    """SO(3) geodesic error without the training loss's nonzero clamp floor."""
    relative = actual.transpose(-1, -2) @ reference
    sine = torch.linalg.vector_norm(torch.stack((
        relative[..., 2, 1] - relative[..., 1, 2],
        relative[..., 0, 2] - relative[..., 2, 0],
        relative[..., 1, 0] - relative[..., 0, 1]), -1), dim=-1) / 2
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.atan2(sine, cosine).square()


def summarize_door_trials(trials):
    """Equal weight per physical scenario; report survivor-only errors separately."""
    errors = ("door_angle_mae_rad", "door_angle_rmse_rad", "door_angle_max_error_rad",
              "door_peak_rad", "door_peak_error_rad", "door_peak_abs_error_rad",
              "root_position_rmse_m", "root_rotation_rmse_rad", "joint_position_rmse_rad",
              "body_mpjpe_m", "body_rotation_rmse_rad")

    def summary(rows):
        def means(values):
            return {key: sum(row[key] for row in values) / len(values) if values else None
                    for key in errors}
        completed = [row for row in rows if row["reference_completed"]]
        return dict(n_trials=len(rows), n_reference_completed=len(completed),
                    **{key + "_rate": sum(row[key] for row in rows) / len(rows) if rows else None
                       for key in ("reference_completed", "terminated", "truncated_early",
                                   "root_crossed", "passed_door", "passed_door_at_end",
                                   "completed_and_passed")},
                    mean_reference_fraction=sum(row["reference_fraction"] for row in rows) / len(rows)
                    if rows else None,
                    mean_norm_return=sum(row["norm_return"] for row in rows) / len(rows) if rows else None,
                    executed_prefix=means(rows), completed_only=means(completed))

    return {"all": summary(trials), **{
        group: summary([row for row in trials if row["group"] == group])
        for group in dict.fromkeys(row["group"] for row in trials)}}


@torch.no_grad()
def evaluate_door_sweep(native, runner, scenarios, record=None, clearance_margin_m=0.02):
    """Score one full reference under one fixed physical scenario per world.

    The caller installs scenario physics in reset events. Finished worlds are
    reset only to satisfy mjlab's manual-reset contract; their terminal values
    remain frozen and subsequent states never contribute to scores or records.
    Passage requires a directional pelvis crossing inside the actual wall
    opening, then every robot interaction geom's bounding sphere clearing the
    back of both walls. This is conservative, unlike sampled surface points.
    """
    import json
    import math
    from pathlib import Path

    import numpy as np

    command = native.command_manager.get_term("motion")
    motion, model, data = command.motion, native.sim.mj_model, native.sim.data
    n, device = native.num_envs, native.device
    if native.cfg.auto_reset or not command.cfg.play:
        raise ValueError("Door sweep requires auto_reset=False and play=True (fixed robot, no noise)")
    if len(motion.names) != 1 or len(scenarios) != n:
        raise ValueError("Door sweep requires a single reference and one scenario per world")
    if not math.isfinite(clearance_margin_m) or clearance_margin_m < 0:
        raise ValueError("Passage clearance margin must be finite and nonnegative")
    ids = torch.arange(n, device=device)
    start, end = int(motion.clip_start[0]), int(motion.clip_end[0])
    command.set_reset_frames(ids, torch.full_like(ids, start))
    obs, _ = native.reset()
    state = command.state()
    physics_names = ("mass", "damping", "hinge_friction", "stiffness", "robot_friction")
    expected_physics = torch.tensor([[item["physics"][name] for name in physics_names]
                                    for item in scenarios], device=device)
    if not torch.allclose(command.xi[:, [39, 40, 41, 44, 43]], expected_physics, atol=1e-6, rtol=1e-6):
        raise ValueError("Reset physics context does not match the requested scenarios")
    metadata = motion.door_metadata
    if abs(float(metadata["base_yaw"])) > 1e-6:
        raise ValueError("Passage evaluation currently requires the centered door asset")
    plane_x = float(metadata["hinge_pos"][0])
    side = 1.0 if float(motion.root_p[start, 0]) > plane_x else -1.0
    if abs(float(motion.root_p[start, 0]) - plane_x) < 1e-6:
        raise ValueError("Reference starts on the door plane; passage direction is ambiguous")
    try:
        walls = [model.geom("door/door_wall_" + name).id for name in ("latch", "hinge")]
    except KeyError as exc:
        raise ValueError("Door sweep passage requires paired door_wall_latch/hinge geometry") from exc
    wall_positions = data.geom_xpos[0, walls] - native.scene.env_origins[0]
    wall_rotations = data.geom_xmat[0, walls].reshape(2, 3, 3)
    wall_sizes = torch.tensor(model.geom_size[walls].copy(), device=device, dtype=wall_positions.dtype)
    wall_extent = (wall_rotations.abs() @ wall_sizes.unsqueeze(-1)).squeeze(-1)
    low, high = wall_positions - wall_extent, wall_positions + wall_extent
    aperture_low, aperture_high = float(high[0, 1]), float(low[1, 1])
    aperture_z_low, aperture_z_high = float(low[:, 2].max()), float(high[:, 2].min())
    wall_depth = plane_x - float(low[:, 0].min()) if side > 0 else float(high[:, 0].max()) - plane_x
    if aperture_high <= aperture_low or wall_depth < 0:
        raise ValueError("Invalid wall aperture or door plane")
    robot_geoms = native.scene["robot"].indexing.geom_ids
    interaction = torch.as_tensor(model.geom_conaffinity.copy(), device=device)[robot_geoms] & 2 != 0
    robot_geoms = robot_geoms[interaction]
    if not len(robot_geoms):
        raise ValueError("No robot interaction geometry for whole-body passage")
    radii = torch.as_tensor(model.geom_rbound.copy(), device=device, dtype=state[0].dtype)[robot_geoms]
    kin = command.mdp.kin
    body_ids = [model.body("robot/" + kin.body_names[i]).id for i in kin.tracked_idx]
    policy = runner.get_inference_policy(device=device)
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    completed, failed, early, root_crossed, entered, passed, passed_at_end = [finished.clone() for _ in range(7)]
    steps = torch.zeros(n, dtype=torch.long, device=device)
    returns = torch.zeros(n, device=device)
    sums = torch.zeros(n, 7, device=device)
    max_error = torch.zeros(n, device=device)
    peak = state[6].squeeze(-1).clone()
    first_pass = torch.full((n,), -1.0, device=device)
    term_names = native.termination_manager.active_terms
    term_flags = torch.zeros(n, len(term_names), dtype=torch.bool, device=device)
    reason_names = ("height", "tilt", "hinge_lower_limit", "hinge_upper_limit", "door_penetration")
    reasons = torch.zeros(n, len(reason_names), dtype=torch.bool, device=device)
    previous_root = state[0].clone()
    frozen_state = [value.clone() for value in state]
    frozen_frames = command.time_steps.clone()
    trajectory = {}

    def capture(alive):
        values = dict(theta=frozen_state[6].squeeze(-1), ref_theta=motion.door_angle[frozen_frames],
                      root_p=frozen_state[0], ref_root_p=motion.root_p[frozen_frames],
                      root_R=frozen_state[1], ref_root_R=motion.root_R[frozen_frames],
                      q=frozen_state[4], ref_q=motion.q[frozen_frames],
                      frame=frozen_frames, alive=alive, passed_door=passed_at_end)
        for key, value in values.items():
            trajectory.setdefault(key, []).append(value.cpu().clone())

    if record is not None:
        capture(~finished)
    while not bool(finished.all()):
        alive = ~finished
        action = policy(obs)
        if not bool(torch.isfinite(action).all()):
            raise FloatingPointError("Nonfinite policy action invalidates door evaluation")
        obs, reward, terminated, truncated, _ = native.step(action)
        state = command.state()
        packed = torch.cat([value.reshape(n, -1) for value in state], -1)
        if not bool(torch.isfinite(packed).all() & torch.isfinite(reward).all()):
            raise FloatingPointError("Nonfinite simulator state or reward invalidates door evaluation")
        if bool(data.overflow.any()):
            raise RuntimeError("Physics buffer overflow invalidates door evaluation")
        frames = command.time_steps
        reached_end = frames >= end
        done = alive & (terminated | truncated | reached_end)
        completed |= alive & reached_end & ~terminated
        failed |= alive & terminated
        early |= alive & truncated & ~reached_end
        returns += torch.where(alive, reward, 0)
        steps += alive
        theta = state[6].squeeze(-1)
        error = theta - motion.door_angle[frames]
        body_p = data.xpos[:, body_ids] - native.scene.env_origins[:, None]
        body_R = data.xmat[:, body_ids].reshape(n, len(body_ids), 3, 3)
        values = torch.stack((error.abs(), error.square(),
                              (state[0] - motion.root_p[frames]).square().sum(-1),
                              _rotation_error_sq(state[1], motion.root_R[frames]),
                              (state[4] - motion.q[frames]).square().mean(-1),
                              torch.linalg.vector_norm(body_p - motion.body_p[frames], dim=-1).mean(-1),
                              _rotation_error_sq(body_R, motion.body_R[frames]).mean(-1)), -1)
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError("Nonfinite tracking metric invalidates door evaluation")
        sums += torch.where(alive[:, None], values, 0)
        max_error = torch.where(alive, torch.maximum(max_error, error.abs()), max_error)
        peak = torch.where(alive, torch.maximum(peak, theta), peak)
        distance, previous_distance = side * (state[0][:, 0] - plane_x), side * (previous_root[:, 0] - plane_x)
        crossing = alive & (previous_distance > 0) & (distance <= 0)
        fraction = previous_distance / (previous_distance - distance).clamp_min(1e-12)
        at_plane = previous_root + fraction[:, None] * (state[0] - previous_root)
        inside = ((at_plane[:, 1] >= aperture_low) & (at_plane[:, 1] <= aperture_high)
                  & (at_plane[:, 2] >= aperture_z_low) & (at_plane[:, 2] <= aperture_z_high))
        entered = torch.where(crossing, inside, entered)
        entered &= ~(alive & (previous_distance <= 0) & (distance > 0))
        root_crossed |= crossing & inside
        centers = data.geom_xpos[:, robot_geoms] - native.scene.env_origins[:, None]
        clearance = (-side * (centers[..., 0] - plane_x) - radii).amin(-1)
        current_pass = entered & (clearance >= wall_depth + clearance_margin_m)
        first_pass = torch.where(alive & current_pass & ~passed, steps * native.step_dt, first_pass)
        passed |= alive & current_pass
        passed_at_end = torch.where(alive, current_pass, passed_at_end)
        previous_root = torch.where(alive[:, None], state[0], previous_root)
        for j, name in enumerate(term_names):
            term_flags[:, j] |= done & native.termination_manager.get_term(name)
        cfg = command.cfg.termination_cfg
        margins, _ = command.mdp._door_margins(state)
        detail = torch.stack(((state[0][:, 2] - motion.root_p[frames, 2]).abs() > cfg.height,
                              (state[1][:, 2, 2] - motion.root_R[frames, 2, 2]).abs() > cfg.tilt,
                              margins[:, 0] <= 0, margins[:, 1] <= 0,
                              margins[:, 2] <= 0 if margins.shape[1] > 2 else torch.zeros_like(alive)), -1)
        reasons |= (alive & terminated)[:, None] & detail
        for j, value in enumerate(state):
            frozen_state[j] = torch.where(alive.reshape(n, *([1] * (value.ndim - 1))), value, frozen_state[j])
        frozen_frames = torch.where(alive, frames, frozen_frames)
        if record is not None:
            capture(alive)
        finished |= done
        if not bool(finished.all()):
            reset_ids = (terminated | truncated).nonzero(as_tuple=False).flatten()
            if len(reset_ids):
                obs, _ = native.reset(env_ids=reset_ids)

    averages = sums / steps.clamp_min(1)[:, None]
    ref_peak = float(motion.door_angle[start:end + 1].max())
    metrics = dict(door_angle_mae_rad=averages[:, 0], door_angle_rmse_rad=averages[:, 1].sqrt(),
                   door_angle_max_error_rad=max_error, door_peak_rad=peak,
                   door_peak_error_rad=peak - ref_peak, door_peak_abs_error_rad=(peak - ref_peak).abs(),
                   root_position_rmse_m=averages[:, 2].sqrt(), root_rotation_rmse_rad=averages[:, 3].sqrt(),
                   joint_position_rmse_rad=averages[:, 4].sqrt(), body_mpjpe_m=averages[:, 5],
                   body_rotation_rmse_rad=averages[:, 6].sqrt(), reference_fraction=steps / (end - start),
                   norm_return=returns / ((end - start) * command.task.max_step_reward))
    columns = {key: value.cpu().tolist() for key, value in dict(
        **metrics, reference_completed=completed, terminated=failed, truncated_early=early,
        root_crossed=root_crossed, passed_door=passed, passed_door_at_end=passed_at_end,
        completed_and_passed=completed & passed_at_end, observed_transitions=steps,
        first_pass_time_s=first_pass, termination_frame=frozen_frames).items()}
    term_values, reason_values = term_flags.cpu().tolist(), reasons.cpu().tolist()
    trials = [dict(scenario, **{key: values[i] for key, values in columns.items()},
                   ref_peak_rad=ref_peak, termination_terms=[name for name, value in zip(term_names, term_values[i]) if value],
                   failure_reasons=[name for name, value in zip(reason_names, reason_values[i]) if value])
              for i, scenario in enumerate(scenarios)]
    for row in trials:
        if row["first_pass_time_s"] < 0:
            row["first_pass_time_s"] = None
    protocol = dict(reference=motion.names[0], reference_transitions=end - start,
                    reference_duration_s=(end - start) * native.step_dt,
                    error_scope="executed transitions including terminal state; fixed reference phase; world coordinates",
                    peak_scope="observed prefix including initial state versus full-reference maximum",
                    pose_bodies=[kin.body_names[i] for i in kin.tracked_idx],
                    passage="pelvis crossed inside wall aperture, then all interaction-geom bounding spheres cleared walls",
                    passage_clearance="conservative collision bounding spheres",
                    clearance_margin_m=clearance_margin_m, door_plane_x_m=plane_x,
                    aperture_y_m=[aperture_low, aperture_high], wall_depth_m=wall_depth,
                    completion="reached final reference frame without termination; final motion_complete truncation allowed",
                    aggregation="equal weight per scenario; completed_only errors accompany all executed-prefix errors")
    if record is not None:
        Path(record).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(record, **{key: torch.stack(values).numpy() for key, values in trajectory.items()},
                            scenarios_json=json.dumps(scenarios), protocol_json=json.dumps(protocol),
                            control_dt=native.step_dt, ref_peak_rad=ref_peak)
    return dict(protocol=protocol, per_trial=trials, summaries=summarize_door_trials(trials))


def back_contact_region(local_points, back_geom):
    """Only torso/logo contacts inside the rear torso patch are legal."""
    x, y, z = local_points.unbind(-1)
    return back_geom & (x < -.025) & (y.abs() < .13) & (z > .05) & (z < .38)


class DoorContactMetrics:
    """Sample solved contact forces before observation-time forward replaces them.

    A category is active when its summed normal force exceeds 1 N in a world.
    This establishes measured contact, not which contact caused opening torque.
    """

    allowed_rate_key = "allowed_back_substep_rate"

    def region_metadata(self):
        return {"back_region_torso_m": dict(x_max=-.025, abs_y_max=.13, z_min=.05, z_max=.38)}

    def classify_contacts(self, local, robot_geom):
        return back_contact_region(local, self.back_geom[robot_geom])

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
            allowed = self.classify_contacts(local, robot[ids])
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
            **self.region_metadata(),
            **{self.allowed_rate_key: float(totals[0])}, forbidden_door_substep_rate=float(totals[1]),
            n_episodes_with_forbidden_contact=int(self.ever_forbidden.sum()),
            per_env=[dict(observed_substeps=int(self.samples[i]),
                          **{self.allowed_rate_key: float(rates[i, 0])},
                          forbidden_door_substep_rate=float(rates[i, 1]),
                          ever_forbidden_contact=bool(self.ever_forbidden[i]))
                     for i in range(len(self.samples))])
