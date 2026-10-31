"""Measured-contact classification and hook lifecycle without a GPU backend."""

from contextlib import nullcontext
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from supermimic.tasks.open_door.evaluation import DoorContactMetrics, back_contact_region


class Scene(dict):
    @property
    def entities(self):
        return self


def test_back_patch_has_strict_boundaries_and_excludes_other_bodies():
    points = torch.tensor([[-.026, 0., .2], [-.025, 0., .2], [-.1, .13, .2],
                           [-.1, -.13, .2], [-.1, 0., .05], [-.1, 0., .38], [-.1, 0., .2]])
    back_body = torch.tensor([True] * 6 + [False])
    assert back_contact_region(points, back_body).tolist() == [True, False, False, False, False, False, False]


@pytest.fixture
def contact_native(monkeypatch):
    events = []
    model = SimpleNamespace(ngeom=5, geom_bodyid=np.array([1, 2, 3, 4, 0]),
        body=lambda name: SimpleNamespace(id={"robot/torso_link": 1, "robot/logo_link": 2}[name]),
        geom=lambda name: SimpleNamespace(id=3))
    xpos = torch.zeros(2, 5, 3)
    xmat = torch.eye(3).expand(2, 5, 3, 3).clone()
    xpos[1, 1] = torch.tensor([2., 3., 1.])
    xmat[1, 1] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    local = torch.tensor([[-.05, 0., .2]] * 7)
    local[3, 0] = -.02  # outside the rear patch, away from float rounding at its boundary
    world = torch.tensor([0, 0, 0, 1, 1, 0, 0])
    points = (xmat[world, 1] @ local[..., None]).squeeze(-1) + xpos[world, 1]
    contacts = SimpleNamespace(
        geom=torch.tensor([[3, 0], [1, 3], [2, 3], [0, 3], [3, 2], [4, 0], [3, 0]]),
        worldid=world, pos=points, efc_address=torch.tensor([[0]] * 6 + [[-1]]),
    )
    data = SimpleNamespace(overflow=torch.zeros(2, dtype=torch.int32), nacon=torch.tensor([7]),
                           contact=contacts, xpos=xpos, xmat=xmat)
    normal_forces = torch.tensor([.6, .6, 1.1, 2., 0., 100., 100.])

    def contact_force(model, data, ids, to_world, output):
        events.append("contact_force")
        assert not to_world
        output[:, 0] = normal_forces[ids.long()]

    monkeypatch.setitem(sys.modules, "mujoco_warp", SimpleNamespace(contact_force=contact_force))
    monkeypatch.setitem(sys.modules, "warp", SimpleNamespace(
        ScopedDevice=lambda device: nullcontext(), from_torch=lambda value, **kwargs: value,
        spatial_vector=object(),
    ))
    sim = SimpleNamespace(mj_model=model, data=data, step=lambda: events.append("physics"),
                          wp_device="cpu", wp_model=None, wp_data=None)
    native = SimpleNamespace(sim=sim, device="cpu", num_envs=2, close=lambda: events.append("close"),
        scene=Scene(robot=SimpleNamespace(indexing=SimpleNamespace(geom_ids=torch.tensor([0, 1, 2]))),
                    door=object()))
    return native, events, normal_forces


def test_substep_force_sums_active_worlds_and_episode_flags(contact_native):
    native, events, normal_forces = contact_native
    original = native.sim.step
    with DoorContactMetrics(native) as metrics:
        native.sim.step()  # no evaluation episode has started
        assert metrics.samples.tolist() == [0, 0]
        metrics.begin_step(torch.tensor([True, False]))
        native.sim.step()
        assert events[-2:] == ["physics", "contact_force"]
        assert metrics.samples.tolist() == [1, 0]
        assert metrics.contact_steps.tolist() == [[1, 1], [0, 0]]
        # World 0 now has no force. World 1 has only forbidden torso contact;
        # local classification must account for its translated/rotated torso.
        normal_forces[:3] = 0.
        metrics.begin_step(torch.tensor([True, True]))
        native.sim.step()
        report = metrics.report()
        assert report["n_episodes_with_forbidden_contact"] == 2
        assert report["per_env"][0]["allowed_back_substep_rate"] == .5
        assert report["per_env"][0]["forbidden_door_substep_rate"] == .5
        assert report["per_env"][1]["forbidden_door_substep_rate"] == 1.
        assert report["allowed_back_substep_rate"] == pytest.approx(1 / 3)
    assert native.sim.step is original
    metrics.close()  # restoration is idempotent
    with pytest.raises(RuntimeError, match="closed"):
        metrics.begin_step(torch.tensor([True, True]))


def test_force_threshold_is_strict_and_invalid_constraints_are_ignored(contact_native):
    native, _, normal_forces = contact_native
    normal_forces[:5] = torch.tensor([.5, .5, 1., 0., 0.])
    with DoorContactMetrics(native) as metrics:
        metrics.begin_step(torch.tensor([True, True]))
        native.sim.step()
        assert not metrics.contact_steps.any()
        assert not metrics.ever_forbidden.any()


@pytest.mark.parametrize("failure", ["overflow", "nonfinite_force"])
def test_capture_failure_restores_simulation_hook(contact_native, failure):
    native, _, normal_forces = contact_native
    original = native.sim.step
    if failure == "overflow":
        native.sim.data.overflow[0] = 1
        error = RuntimeError
    else:
        normal_forces[0] = torch.nan
        error = FloatingPointError
    with pytest.raises(error):
        with DoorContactMetrics(native) as metrics:
            metrics.begin_step(torch.tensor([True, False]))
            native.sim.step()
    assert native.sim.step is original


def test_score_checkpoint_restores_contact_hook_when_loading_fails(contact_native, tmp_path, monkeypatch):
    from supermimic.scripts import evaluate as evaluation

    native, events, _ = contact_native
    original = native.sim.step
    (tmp_path / "clip.npz").touch()
    cfg = SimpleNamespace(auto_reset=True, scene=SimpleNamespace(num_envs=1), commands={
        "motion": SimpleNamespace(motion_dir=str(tmp_path), motion_files=(), door_xml=None),
    })
    monkeypatch.setattr(evaluation.torch, "load", lambda *args, **kwargs: {})
    monkeypatch.setattr(evaluation, "load_env_cfg", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(evaluation, "load_rl_cfg", lambda *args: None)
    monkeypatch.setattr(evaluation, "asdict", lambda cfg: {})
    monkeypatch.setattr(evaluation, "ManagerBasedRlEnv", lambda *args, **kwargs: native)
    monkeypatch.setattr(evaluation, "RslRlVecEnvWrapper", lambda env: env)

    def fail_load(*args, **kwargs):
        assert native.sim.step is not original
        raise RuntimeError("checkpoint rejected")

    monkeypatch.setattr(evaluation, "load_runner_cls", lambda task: lambda *args, **kwargs:
                        SimpleNamespace(load=fail_load))
    with pytest.raises(RuntimeError, match="checkpoint rejected"):
        evaluation.score_checkpoint(tmp_path / "missing.pt", evaluation.build_parser().parse_args([]))
    assert native.sim.step is original and events[-1] == "close"
