"""Deployment-only sitting metrics; no reward, training or dynamics changes.

The default contact region is a reference-conditioned geometric predicate, NOT
a semantic seat annotation. Forces are captured immediately after each real
Warp step, before mjlab's observation-time forward recomputes the constraints.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

from ...rotations import mat_to_quat, quat_slerp, quat_to_mat, so3_log


@dataclass(frozen=True)
class SittingEvaluationCfg:
    stable_window_s: float = 1.0
    window_end_s: float | None = None
    min_stable_fraction: float = .95
    max_gap_s: float = .04
    min_support_bodyweight: float = .20
    max_contact_slip_mps: float = .10
    max_chair_linear_mps: float = .05
    max_chair_angular_radps: float = .15
    max_pelvis_relative_error_m: float = .10
    max_root_orientation_error_deg: float = 20.
    min_normal_up_cosine: float = math.sqrt(.5)
    support_radius_m: float = .22
    support_below_pelvis_min_m: float = .025
    support_below_pelvis_max_m: float = .25
    # Optional reviewed bounds in the scaled chair-local frame, in metres.
    seat_bounds_min: tuple[float, float, float] | None = None
    seat_bounds_max: tuple[float, float, float] | None = None

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key.startswith("seat_bounds") or value is None:
                continue
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid evaluation threshold {key}: {value}")
        if self.stable_window_s <= 0 or not 0 < self.min_stable_fraction <= 1:
            raise ValueError("Positive stable window and stable fraction in (0, 1] required")
        if not 0 < self.min_normal_up_cosine <= 1 or self.min_support_bodyweight <= 0:
            raise ValueError("Support requires a positive load and upward normal threshold")
        if self.support_below_pelvis_min_m >= self.support_below_pelvis_max_m:
            raise ValueError("Invalid reference support height interval")
        if (self.seat_bounds_min is None) != (self.seat_bounds_max is None):
            raise ValueError("Both seat_bounds_min and seat_bounds_max are required")
        if self.seat_bounds_min is not None:
            lo, hi = np.asarray(self.seat_bounds_min), np.asarray(self.seat_bounds_max)
            if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or np.any(lo >= hi):
                raise ValueError("Seat bounds must be finite ordered 3-vectors")


def _summary(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {"mean": None, "max": None, "final": None}
    return {"mean": float(values.mean()), "max": float(values.max()), "final": float(values[-1])}


def _longest_gap(passed):
    longest, gap = 0, 0
    for ok in passed:
        gap = 0 if ok else gap + 1
        longest = max(longest, gap)
    return longest


def summarize_episode(cfg, physics, tracking, *, frame0, clip_end, control_dt,
                      physics_dt, reference_completed):
    """Pure aggregation, including missing-window failures in the denominator.

    Physics samples describe intervals ending at ``time_s``. The declared target
    window is (end - duration, end]; it never searches for a favourable interval.
    Early termination remains eligible and fails. Only an impossible protocol
    (short reference, out-of-range window, or a late initial phase) is ineligible.
    """
    clip_duration = clip_end * control_dt
    end = clip_duration if cfg.window_end_s is None else cfg.window_end_s
    start = end - cfg.stable_window_s
    reasons = []
    eligible = True
    if start < -1e-8 or end > clip_duration + 1e-8:
        reasons.append("reference_does_not_cover_declared_window")
        eligible = False
    if frame0 * control_dt > start + 1e-8:
        reasons.append("initial_phase_is_after_window_start")
        eligible = False
    expected = round(cfg.stable_window_s / physics_dt)
    if expected < 1 or not math.isclose(expected * physics_dt, cfg.stable_window_s, abs_tol=1e-7):
        raise ValueError("Stable window must be a multiple of the physics timestep")
    if not math.isclose(end / physics_dt, round(end / physics_dt), abs_tol=1e-6):
        raise ValueError("Window end must align with the physics timestep")
    window = [sample for sample in physics if start + 1e-8 < sample["time_s"] <= end + 1e-8]
    limits = {
        "support_load": ("load_ratio", cfg.min_support_bodyweight, True),
        "contact_slip": ("slip_mps", cfg.max_contact_slip_mps, False),
        "chair_linear_velocity": ("chair_linear_mps", cfg.max_chair_linear_mps, False),
        "chair_angular_velocity": ("chair_angular_radps", cfg.max_chair_angular_radps, False),
        "pelvis_relative_tracking": ("pelvis_relative_error_m", cfg.max_pelvis_relative_error_m, False),
        "root_orientation_tracking": ("root_orientation_error_rad", math.radians(cfg.max_root_orientation_error_deg), False),
    }
    predicates = {}
    for name, (field, threshold, minimum) in limits.items():
        values = np.asarray([sample[field] for sample in window])
        if not np.isfinite(values).all():
            raise FloatingPointError(f"Nonfinite evaluation metric: {field}")
        predicates[name] = (values >= threshold) if minimum else (values <= threshold)
    satisfied = np.logical_and.reduce(list(predicates.values()))
    # Count unsampled target intervals as failures, including failure before the
    # target window. This prevents survival-conditioned success inflation.
    complete_window = len(window) == expected
    stable_fraction = float(satisfied.sum() / expected) if expected else 0.
    max_gap = _longest_gap(satisfied.tolist() + [False] * max(0, expected - len(window)))
    if not reference_completed:
        reasons.append("full_reference_not_completed")
    if not complete_window:
        reasons.append("declared_window_not_fully_observed")
    if stable_fraction + 1e-8 < cfg.min_stable_fraction:
        reasons.append("insufficient_jointly_stable_fraction")
    if max_gap * physics_dt > cfg.max_gap_s + 1e-8:
        reasons.append("stability_gap_too_long")
    # Report observed predicate failures separately from absent data. A missing
    # window is not evidence of e.g. excessive sliding or poor pose tracking.
    for name, passed in predicates.items():
        if len(passed) and float(passed.mean()) + 1e-8 < cfg.min_stable_fraction:
            reasons.append(name + "_below_required_fraction")
        if _longest_gap(passed) * physics_dt > cfg.max_gap_s + 1e-8:
            reasons.append(name + "_gap_too_long")
    tracking_summary = {key: _summary([sample[key] for sample in tracking])
                        for key in tracking[0] if key != "time_s"} if tracking else {}
    contact_summary = {key: _summary([sample[key] for sample in window]) for key in
                       ("load_ratio", "slip_mps", "chair_linear_mps", "chair_angular_radps")}
    return {
        "stable_contact_success": bool(eligible and reference_completed and complete_window and not reasons),
        "stable_contact_eligible": eligible,
        "failure_reasons": reasons,
        "metrics": {
            "definition": "sitting_stable_contact_v1",
            "thresholds": asdict(cfg),
            "force_source": "mujoco_warp.contact_force immediately after each real physics step",
            "reference_support_region_provenance": (
                "explicit_chair_local_bounds" if cfg.seat_bounds_min is not None else
                "reference_pelvis_cylinder_not_semantically_annotated_seat"),
            "allowed_robot_bodies": list(SittingEpisodeMetrics.SUPPORT_BODIES),
            "tracking_scope": "executed_prefix_only; fixed reference phase; no spatial or time alignment",
            "tracking": tracking_summary,
            "stable_window": {
                "start_s": start, "end_s": end, "physics_dt": physics_dt,
                "expected_samples": expected, "observed_samples": len(window),
                "jointly_stable_fraction": stable_fraction,
                "max_unstable_gap_s": max_gap * physics_dt,
                "stable_contact_duration_s": int(satisfied.sum()) * physics_dt,
                "load_bearing_contact_duration_s": int(predicates["support_load"].sum()) * physics_dt,
                "criterion_fractions": {key: float(value.sum() / expected) for key, value in predicates.items()},
                "contact": contact_summary,
            },
        },
    }


class SittingEpisodeMetrics:
    """One-environment collector. Call reset, update after control steps, close.

    MuJoCo contact normal points from geom[0] to geom[1]; contact_force gives
    force on geom[1]. The sign is reversed when the robot is geom[0]. Contact
    slip uses both bodies' velocities at that contact point, including rotation.
    """
    SUPPORT_BODIES = ("pelvis", "pelvis_contour_link", "left_hip_pitch_link", "left_hip_roll_link",
                      "left_hip_yaw_link", "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link")

    def __init__(self, native, cfg=None):
        import warp as wp

        if native.num_envs != 1:
            raise ValueError("Deployment evaluation requires exactly one environment")
        self.native, self.cfg = native, cfg or SittingEvaluationCfg()
        self.command = native.command_manager.get_term("motion")
        self.sim = native.sim
        model, device = self.sim.mj_model, native.device
        robot_ids = native.scene["robot"].indexing.body_ids
        chair_ids = native.scene["chair"].indexing.body_ids
        self.robot_root = int(native.scene["robot"].indexing.root_body_id)
        self.chair_root = int(native.scene["chair"].indexing.root_body_id)
        self.support_body = torch.zeros(model.nbody, dtype=torch.bool, device=device)
        for body in robot_ids.tolist():
            if model.body(body).name.rsplit("/", 1)[-1] in self.SUPPORT_BODIES:
                self.support_body[body] = True
        if not bool(self.support_body.any()):
            raise ValueError("No pelvis or proximal-thigh bodies found in this robot")
        self.chair_body = torch.zeros_like(self.support_body)
        self.chair_body[chair_ids] = True
        self.geom_body = torch.as_tensor(model.geom_bodyid.copy(), dtype=torch.long, device=device)
        self.body_root = torch.as_tensor(model.body_rootid.copy(), dtype=torch.long, device=device)
        gravity = torch.as_tensor(model.opt.gravity.copy(), device=device, dtype=torch.float32)
        gravity_norm = float(torch.linalg.vector_norm(gravity))
        if gravity_norm <= 0:
            raise ValueError("Stable support requires nonzero gravity")
        self.up = -gravity / gravity_norm
        masses = self.sim.model.body_mass
        self.bodyweight = float(masses[0, robot_ids].sum()) * gravity_norm
        if self.bodyweight <= 0:
            raise ValueError("Invalid robot bodyweight")
        self._wp = wp
        self._original_step = self.sim.step
        self._active = False
        self._closed = False
        self.physics, self.tracking = [], []

        def measured_step():
            self._original_step()
            if self._active:
                self._capture_physics()

        self._measured_step = measured_step
        self.sim.step = measured_step

    def close(self):
        if not self._closed:
            if self.sim.step is self._measured_step:
                self.sim.step = self._original_step
            self._active, self._closed = False, True

    def reset(self, frame0):
        self.frame0, self.physics_steps = int(frame0), 0
        self.physics, self.tracking = [], []
        self._active = True

    def _reference(self, time_s):
        motion = self.command.motion
        fraction = min(max(time_s / self.native.step_dt, 0.), int(motion.clip_end[0]))
        lo = int(math.floor(fraction + 1e-8))
        hi = min(lo + 1, int(motion.clip_end[0]))
        indices = torch.tensor([lo, hi], device=self.native.device)
        states = motion.full_ref_state(indices)
        alpha = fraction - lo
        result = []
        for value in states:
            if value.ndim == 3:
                result.append(quat_to_mat(quat_slerp(mat_to_quat(value[:1]), mat_to_quat(value[1:]),
                                                   value.new_tensor([alpha])))[0])
            else:
                result.append(torch.lerp(value[0], value[1], alpha))
        return tuple(result)

    def _point_velocity(self, body, point):
        data = self.sim.data
        velocity = data.cvel[0, body]
        center = data.subtree_com[0, self.body_root[body]]
        return velocity[..., 3:] + torch.cross(velocity[..., :3], point - center, dim=-1)

    def _capture_physics(self):
        import mujoco_warp as mjwarp

        data, cfg = self.sim.data, self.cfg
        if bool(data.overflow.any()):
            raise RuntimeError("Physics buffer overflow invalidates contact evaluation")
        # mj_step's contact/kinematic buffers describe the force solution before
        # its integration. Label that interval by its end, compare against its
        # beginning reference pose; never mix refreshed kinematics with old force.
        time0 = self.frame0 * self.native.step_dt + self.physics_steps * self.native.physics_dt
        ref = self._reference(time0)
        self.physics_steps += 1
        time1 = self.frame0 * self.native.step_dt + self.physics_steps * self.native.physics_dt
        origin = self.native.scene.env_origins[0]
        p = data.xpos[0, self.robot_root] - origin
        R = quat_to_mat(data.xquat[0, self.robot_root])
        cp = data.xpos[0, self.chair_root] - origin
        cR = quat_to_mat(data.xquat[0, self.chair_root])
        relative_error = torch.linalg.vector_norm(cR.T @ (p - cp) - ref[7].T @ (ref[0] - ref[6]))
        root_error = torch.linalg.vector_norm(so3_log(ref[1].T @ R))
        chair_v = self._point_velocity(self.chair_root, data.xpos[0, self.chair_root])
        chair_w = data.cvel[0, self.chair_root, :3]
        count = int(data.nacon[0])
        pairs = self.geom_body[data.contact.geom[:count].long()]
        robot_second = self.chair_body[pairs[:, 0]] & self.support_body[pairs[:, 1]]
        robot_first = self.support_body[pairs[:, 0]] & self.chair_body[pairs[:, 1]]
        valid = (robot_first | robot_second) & (data.contact.worldid[:count] == 0)
        ids = valid.nonzero(as_tuple=False).flatten()
        load, slip, valid_count = 0., 0., 0
        if ids.numel():
            wp = self._wp
            ids32 = ids.to(torch.int32).contiguous()
            with wp.ScopedDevice(self.sim.wp_device):
                forces_wp = wp.empty(len(ids), dtype=wp.spatial_vector)
                mjwarp.contact_force(self.sim.wp_model, self.sim.wp_data, wp.from_torch(ids32), True, forces_wp)
                forces = wp.to_torch(forces_wp)
                if not bool(torch.isfinite(forces).all()):
                    raise FloatingPointError("Nonfinite physics contact force")
                forces = forces[:, :3]
                sign = torch.where(robot_second[ids], 1., -1.)
                forces = forces * sign[:, None]
                normal = data.contact.frame[ids, 0] * sign[:, None]
                point = data.contact.pos[ids]
                if not bool(torch.isfinite(normal).all() & torch.isfinite(point).all()):
                    raise FloatingPointError("Nonfinite physics contact normal or position")
                pair = pairs[ids]
                relative_v = self._point_velocity(pair[:, 1], point) - self._point_velocity(pair[:, 0], point)
                if not bool(torch.isfinite(relative_v).all()):
                    raise FloatingPointError("Nonfinite physics contact relative velocity")
                point_local = (point - origin - cp) @ cR
                if cfg.seat_bounds_min is not None:
                    lo = point_local.new_tensor(cfg.seat_bounds_min)
                    hi = point_local.new_tensor(cfg.seat_bounds_max)
                    region = ((point_local >= lo) & (point_local <= hi)).all(-1)
                else:
                    # Express the actual contact in the REFERENCE chair pose,
                    # making this predicate independent of policy displacement.
                    delta = point_local @ ref[7].T + ref[6] - ref[0]
                    height = delta @ self.up
                    lateral = torch.linalg.vector_norm(delta - height[:, None] * self.up, dim=-1)
                    region = ((lateral <= cfg.support_radius_m) &
                              (height <= -cfg.support_below_pelvis_min_m) &
                              (height >= -cfg.support_below_pelvis_max_m))
                upward = forces @ self.up
                support = region & (normal @ self.up >= cfg.min_normal_up_cosine) & (upward > 0)
                # Only genuinely load-bearing contacts enter maximum slip;
                # contacts carrying <0.1% bodyweight are numerical grazing.
                support &= upward >= .001 * self.bodyweight
                valid_count = int(support.sum())
                if valid_count:
                    load = float(upward[support].sum()) / self.bodyweight
                    normal, relative_v = normal[support], relative_v[support]
                    tangent = relative_v - (relative_v * normal).sum(-1, keepdim=True) * normal
                    slip = float(torch.linalg.vector_norm(tangent, dim=-1).max())
        sample = dict(time_s=time1, load_ratio=load, slip_mps=slip, valid_contact_count=valid_count,
                      chair_linear_mps=float(torch.linalg.vector_norm(chair_v)),
                      chair_angular_radps=float(torch.linalg.vector_norm(chair_w)),
                      pelvis_relative_error_m=float(relative_error), root_orientation_error_rad=float(root_error))
        if not all(math.isfinite(value) for value in sample.values()):
            raise FloatingPointError("Nonfinite physics contact evaluation sample")
        self.physics.append(sample)

    def update(self):
        expected = (len(self.tracking) + 1) * self.native.cfg.decimation
        if self.physics_steps != expected:
            raise RuntimeError("Missing or duplicate real physics-step contact samples")
        S = tuple(value[0] for value in self.command.state())
        frame = int(self.command.time_steps[0])
        ref = self._reference(frame * self.native.step_dt)
        norm = lambda value: float(torch.linalg.vector_norm(value))
        self.tracking.append(dict(
            time_s=frame * self.native.step_dt,
            root_position_error_m=norm(S[0] - ref[0]),
            root_orientation_error_rad=norm(so3_log(ref[1].T @ S[1])),
            joint_rmse_rad=float((S[4] - ref[4]).square().mean().sqrt()),
            joint_max_abs_error_rad=float((S[4] - ref[4]).abs().max()),
            chair_position_error_m=norm(S[6] - ref[6]),
            chair_orientation_error_rad=norm(so3_log(ref[7].T @ S[7])),
            pelvis_chair_position_error_m=norm(S[7].T @ (S[0] - S[6]) - ref[7].T @ (ref[0] - ref[6])),
            root_linear_speed_mps=norm(S[2]), root_angular_speed_radps=norm(S[3]),
            chair_linear_speed_mps=norm(S[8]), chair_angular_speed_radps=norm(S[9]),
        ))

    def finalize(self, reference_completed: bool):
        return summarize_episode(self.cfg, self.physics, self.tracking, frame0=self.frame0,
            clip_end=int(self.command.motion.clip_end[0]), control_dt=self.native.step_dt,
            physics_dt=self.native.physics_dt, reference_completed=reference_completed)
