"""Entity archives retain chronology, physical identities and cross-task data."""

import pytest
import torch

from supermimic.rl.entity_replay import EntityReplay, save_entity_replay
from supermimic.rl.entity_state import EntityDynamics
from supermimic.rl.entity_world_model import EntityGeometryBank, EntityWorldModel
from supermimic.rl.shared_world_model import SharedDynamics
from supermimic.rl.storage import ReplayBuffer


def add_shape(bank, name, size=1.):
    cloud = torch.zeros(1024, 6)
    cloud[:, :3] = torch.linspace(-size / 2, size / 2, 3072).reshape(1024, 3)
    cloud[:, 5] = 1
    return bank.add(name, cloud, torch.full((3,), float(size)))


def make_archive(path, offset=0, *, task_key="chair", size=1.):
    dynamics = EntityDynamics(max_objects=3)
    bank = EntityGeometryBank()
    add_shape(bank, task_key, size)
    add_shape(bank, "fixed-obstacle", 2.)
    buffer = ReplayBuffer(4, 2, "cpu", state_dim=dynamics.state_dim, context_dim=dynamics.context_dim)
    for step in range(6):
        state = torch.zeros(2, dynamics.state_dim)
        state[:, 0] = step + offset
        state[:, 1] = torch.arange(2)
        state[:, 3] = 1
        for slot in range(dynamics.max_objects):
            state[:, 74 + slot * 13] = 1
        context = dynamics.empty_context(2)
        context[:, 0] = .8
        objects = dynamics.object_context(context)
        objects[:, :2, 0] = 10
        objects[:, :2, 4:7] = torch.tensor([.2, .3, .4])
        objects[:, :2, 10] = 1
        objects[:, 0, 11] = 1
        objects[:, :2, 12] = torch.tensor([1., 2.])
        dynamics.pair_friction(context)[..., 0] = .7
        next_state = state.clone()
        next_state[:, 0] += 1
        buffer.add(dict(s=state, s1=next_state, a=torch.zeros(2, 29), a_prev=torch.zeros(2, 29),
                        xi=context, r=torch.full((2,), float(offset)), f=torch.full((2,), step),
                        ep_step=torch.full((2,), step), done=torch.full((2,), step == 3)))
    save_entity_replay(path, buffer, bank, dynamics)
    return dynamics, bank, buffer


def test_entity_archive_remaps_each_slot_keeps_padding_and_ring_chronology(tmp_path):
    path = tmp_path / "sitting.pt"
    dynamics, source_bank, _ = make_archive(path, offset=10)
    destination = EntityGeometryBank()
    add_shape(destination, "already-loaded", .5)
    replay = EntityReplay([path], destination, dynamics)
    assert destination.keys == ["__none__", "already-loaded", "chair", "fixed-obstacle"]
    loaded = replay.archives[0]
    torch.testing.assert_close(loaded["s"][:, 0, 0], torch.tensor([12., 13., 14., 15.]))
    torch.testing.assert_close(dynamics.object_context(loaded["xi"])[..., 12],
                               torch.tensor([2., 3., 0.]).expand(4, 2, 3))
    for source_id, destination_id in ((1, 2), (2, 3)):
        torch.testing.assert_close(source_bank.points[source_id], destination.points[destination_id])
    assert set(loaded) == {"s", "s1", "a", "xi", "done"}
    assert not (tmp_path / "sitting.pt.tmp").exists()


def test_mixed_task_windows_never_mix_archives_environments_or_unmask_post_reset_rows(tmp_path):
    first, second = tmp_path / "sitting.pt", tmp_path / "pushing.pt"
    dynamics, _, _ = make_archive(first, offset=10, task_key="chair")
    make_archive(second, offset=100, task_key="box")
    bank = EntityGeometryBank()
    replay = EntityReplay([first, second], bank, dynamics)
    assert bank.keys == ["__none__", "chair", "fixed-obstacle", "box"]
    torch.manual_seed(19)
    windows = replay.sample_windows(32, 4, "cpu")
    source_ids = dynamics.object_context(windows["xi"])[..., 0, 12]
    assert set(source_ids[:, 0].tolist()) == {1., 3.}
    for row in range(32):
        assert (source_ids[row] == source_ids[row, 0]).all()
        assert (windows["s"][row, :, 1] == windows["s"][row, 0, 1]).all()
        torch.testing.assert_close(windows["s"][row, 1:, 0] - windows["s"][row, :-1, 0], torch.ones(3))
    torch.testing.assert_close(windows["mask"], torch.tensor([1., 1., 0., 0.]).expand(32, 4))
    with pytest.raises(ValueError, match="No replay archive"):
        replay.sample_windows(1, 5, "cpu")


@pytest.mark.parametrize("kind", ["geometry", "presence", "dynamic"])
def test_nonterminal_entity_identity_changes_are_rejected_before_geometry_merge(tmp_path, kind):
    path = tmp_path / "bad.pt"
    dynamics, _, _ = make_archive(path)
    data = torch.load(path, weights_only=True)
    objects = dynamics.object_context(data["transitions"]["xi"])
    # Archive row zero has not terminated. Changing row one changes the live slot.
    if kind == "geometry":
        objects[1:, :, 0, 12] = 2
    elif kind == "presence":
        objects[1:, :, 0, 10:13] = 0
    else:
        objects[1:, :, 0, 11] = 0
    torch.save(data, path)
    bank = EntityGeometryBank()
    with pytest.raises(ValueError, match="identities changed"):
        EntityReplay([path], bank, dynamics)
    assert bank.keys == ["__none__"]


def test_entity_identity_can_change_after_done_and_new_episode_stays_masked(tmp_path):
    path = tmp_path / "reset.pt"
    dynamics, _, _ = make_archive(path)
    data = torch.load(path, weights_only=True)
    objects = dynamics.object_context(data["transitions"]["xi"])
    # Row one is terminal; row two can select a different chair geometry.
    objects[2:, :, 0, 12] = 2
    torch.save(data, path)
    replay = EntityReplay([path], EntityGeometryBank(), dynamics)
    samples = replay.sample_windows(4, 4, "cpu")
    torch.testing.assert_close(samples["mask"], torch.tensor([1., 1., 0., 0.]).expand(4, 4))
    assert (dynamics.object_context(samples["xi"])[:, 2:, 0, 12] == 2).all()


def test_single_body_schema_is_not_silently_reinterpreted(tmp_path):
    path = tmp_path / "old.pt"
    dynamics, _, _ = make_archive(path)
    data = torch.load(path, weights_only=True)
    data["schema"] = SharedDynamics.schema()
    torch.save(data, path)
    with pytest.raises(ValueError, match="Incompatible entity replay schema"):
        EntityReplay([path], EntityGeometryBank(), dynamics)


def test_same_geometry_is_deduplicated_across_tasks_and_conflicts_fail_atomically(tmp_path):
    sitting, pushing, conflicting = (tmp_path / name for name in ("sitting.pt", "pushing.pt", "conflict.pt"))
    dynamics, _, _ = make_archive(sitting, offset=10)
    make_archive(pushing, offset=20)
    make_archive(conflicting, offset=30, size=1.5)
    bank = EntityGeometryBank()
    replay = EntityReplay([sitting, pushing], bank, dynamics)
    assert bank.keys == ["__none__", "chair", "fixed-obstacle"]
    assert len(replay.archives) == 2
    saved_points = bank.points.clone()
    with pytest.raises(ValueError, match="different content"):
        EntityReplay([conflicting], bank, dynamics)
    torch.testing.assert_close(bank.points, saved_points)


def test_checkpoint_network_contract_ignores_capacity_but_records_architecture():
    first = EntityWorldModel(EntityDynamics(1), width=32, layers=2, heads=4)
    larger = EntityWorldModel(EntityDynamics(8), width=32, layers=2, heads=4)
    changed = EntityWorldModel(EntityDynamics(8), width=64, layers=3, heads=8)
    assert first.network_schema() == larger.network_schema()
    assert first.dynamics.schema() != larger.dynamics.schema()
    assert changed.network_schema() != first.network_schema()
    larger.load_state_dict(first.state_dict())
    assert first.network_schema()["geometry"]["channels"] == 6
