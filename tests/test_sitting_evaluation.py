"""Success denominators, physical support semantics and force-sampling timing."""
from types import SimpleNamespace

import pytest
import torch

from supermimic.tasks.sitting.evaluation import (
    SittingEpisodeMetrics, SittingEvaluationCfg, summarize_episode,
)


def series(count=400):
    return [dict(time_s=(i + 1) * .005, load_ratio=.4, slip_mps=.01,
                 chair_linear_mps=.005, chair_angular_radps=.01,
                 pelvis_relative_error_m=.01, root_orientation_error_rad=.02)
            for i in range(count)]


def evaluate(samples, *, cfg=None, frame0=0, clip_end=100, completed=True):
    return summarize_episode(cfg or SittingEvaluationCfg(), samples,
        [dict(time_s=.02, root_position_error_m=.03)], frame0=frame0,
        clip_end=clip_end, control_dt=.02, physics_dt=.005,
        reference_completed=completed)


def test_reference_completion_and_load_bearing_stability_are_distinct():
    success = evaluate(series())
    assert success["stable_contact_success"] and success["stable_contact_eligible"]
    assert success["metrics"]["stable_window"]["stable_contact_duration_s"] == 1.
    unsupported = series()
    for sample in unsupported:
        sample["load_ratio"] = 0
    result = evaluate(unsupported)
    assert not result["stable_contact_success"] and result["stable_contact_eligible"]
    assert result["metrics"]["stable_window"]["criterion_fractions"]["support_load"] == 0
    assert result["metrics"]["tracking"]["root_position_error_m"]["mean"] == .03


def test_early_failure_is_not_removed_from_success_denominator():
    result = evaluate(series(20), completed=False)
    assert result["stable_contact_eligible"] and not result["stable_contact_success"]
    assert "full_reference_not_completed" in result["failure_reasons"]
    assert result["metrics"]["stable_window"]["jointly_stable_fraction"] == 0
    assert result["metrics"]["stable_window"]["max_unstable_gap_s"] == 1.
    empty = summarize_episode(SittingEvaluationCfg(), [], [], frame0=0, clip_end=100,
        control_dt=.02, physics_dt=.005, reference_completed=False)
    assert empty["stable_contact_eligible"] and not empty["stable_contact_success"]
    assert empty["metrics"]["tracking"] == {}


def test_fixed_final_window_does_not_pick_earlier_favourable_contact():
    samples = series()
    for sample in samples[200:]:
        sample["slip_mps"] = .3
    assert not evaluate(samples)["stable_contact_success"]
    # An independently declared task phase can select the earlier interval.
    assert evaluate(samples, cfg=SittingEvaluationCfg(window_end_s=1.))["stable_contact_success"]


def test_joint_fraction_and_longest_gap_both_matter():
    samples = series()
    for sample in samples[250:259]:
        sample["load_ratio"] = 0
    result = evaluate(samples)
    assert result["metrics"]["stable_window"]["jointly_stable_fraction"] == pytest.approx(.955)
    assert "stability_gap_too_long" in result["failure_reasons"]
    assert "support_load_gap_too_long" in result["failure_reasons"]
    samples = series()
    for index in range(210, 300, 10):
        samples[index]["load_ratio"] = 0
    assert evaluate(samples)["stable_contact_success"]
    for index in (310, 320):
        samples[index]["load_ratio"] = 0
    assert "insufficient_jointly_stable_fraction" in evaluate(samples)["failure_reasons"]


def test_short_references_and_late_initial_phases_are_explicitly_ineligible():
    assert not evaluate(series(100), clip_end=25)["stable_contact_eligible"]
    result = evaluate(series(400)[300:], frame0=75)
    assert not result["stable_contact_eligible"]
    assert "initial_phase_is_after_window_start" in result["failure_reasons"]


@pytest.mark.parametrize("field,value", [("load_ratio", .19), ("slip_mps", .101),
    ("chair_linear_mps", .051), ("chair_angular_radps", .151),
    ("pelvis_relative_error_m", .101), ("root_orientation_error_rad", .36)])
def test_each_strict_predicate_can_reject_an_episode(field, value):
    samples = series()
    for sample in samples:
        sample[field] = value
    assert not evaluate(samples)["stable_contact_success"]


def test_bad_configuration_and_nonfinite_measurements_fail_loudly():
    with pytest.raises(ValueError, match="Both seat"):
        SittingEvaluationCfg(seat_bounds_min=(0, 0, 0))
    with pytest.raises(ValueError, match="3-vectors"):
        SittingEvaluationCfg(seat_bounds_min=(1, 0, 0), seat_bounds_max=(0, 1, 1))
    with pytest.raises(ValueError):
        SittingEvaluationCfg(max_gap_s=float("nan"))
    samples = series()
    samples[-1]["load_ratio"] = float("nan")
    with pytest.raises(FloatingPointError):
        evaluate(samples)


def test_real_warp_force_direction_timing_and_nonseat_body_rejection(cuda_device, monkeypatch):
    """Small box standing for pelvis: inspect actual step force, not a replayed forward."""
    import mujoco
    import mujoco_warp
    from mjlab.entity import EntityCfg
    from mjlab.scene import Scene, SceneCfg
    from mjlab.sim import Simulation, SimulationCfg

    def robot_spec():
        spec = mujoco.MjSpec()
        body = spec.worldbody.add_body(name="pelvis", pos=[0, 0, .549])
        body.add_freejoint()
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[.05] * 3, mass=1., condim=3)
        # A second unallowed foot contact must never count toward seated support.
        foot = body.add_body(name="left_ankle_roll_link", pos=[.4, 0, 0])
        foot.add_joint(name="foot_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1])
        foot.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[.05] * 3, mass=1., condim=3)
        return spec

    def chair_spec():
        spec = mujoco.MjSpec()
        body = spec.worldbody.add_body(name="chair", pos=[0, 0, .25])
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[1., 1., .25], condim=3)
        return spec

    scene = Scene(SceneCfg(num_envs=1, entities={
        "robot": EntityCfg(spec_fn=robot_spec), "chair": EntityCfg(spec_fn=chair_spec)}), device=cuda_device)
    scene.compile()
    sim = Simulation(cfg=SimulationCfg(nconmax=32, njmax=128), num_envs=1,
                     spec=scene.spec, device=cuda_device)
    scene.initialize(sim.mj_model, sim.model, sim.data)
    device = cuda_device
    p = torch.tensor([0., 0., .549], device=device)
    # A fixed entity has a zero-pose attachment root; its box is offset by .25 m.
    cp = torch.zeros(3, device=device)
    R = torch.eye(3, device=device)
    zero = torch.zeros(3, device=device)
    q = torch.zeros(29, device=device)
    def ref_state(frames):
        return tuple(value.expand(len(frames), *value.shape) for value in (p, R, zero, zero, q, q, cp, R, zero, zero))
    command = SimpleNamespace(motion=SimpleNamespace(clip_end=torch.tensor([100], device=device),
                                                     full_ref_state=ref_state))
    native = SimpleNamespace(num_envs=1, device=device, scene=scene, sim=sim,
        step_dt=4 * sim.mj_model.opt.timestep, physics_dt=sim.mj_model.opt.timestep,
        command_manager=SimpleNamespace(get_term=lambda _: command), cfg=SimpleNamespace(decimation=4))
    original = sim.step
    collector = SittingEpisodeMetrics(native, SittingEvaluationCfg(support_radius_m=1.))
    try:
        collector.reset(0)
        assert collector.finalize(False)["metrics"]["tracking"] == {}
        sim.step()
        sample = collector.physics[-1]
        assert sample["load_ratio"] > .2 and sample["valid_contact_count"] > 0
        assert sample["slip_mps"] < .01
        preserved = sample.copy()
        # A NaN force must invalidate the rollout before the support predicate
        # can turn its failed comparison into an apparently valid zero load.
        with monkeypatch.context() as patch:
            def invalid_contact_force(*args):
                collector._wp.to_torch(args[-1]).fill_(float("nan"))
            patch.setattr(mujoco_warp, "contact_force", invalid_contact_force)
            with pytest.raises(FloatingPointError, match="contact force"):
                sim.step()
        assert len(collector.physics) == 1
        # Changing qpos then forward must not rewrite the previously captured
        # physical force; a post-control forward-only implementation fails this.
        sim.data.qpos[:, 2] += 1.
        sim.data.qpos[:, 7] -= 1.
        sim.forward()
        assert collector.physics[-1] == preserved
        collector.reset(0)
        sim.step()
        assert collector.physics[-1]["load_ratio"] == 0.
        assert collector.physics[-1]["valid_contact_count"] == 0
    finally:
        collector.close()
    assert sim.step == original
