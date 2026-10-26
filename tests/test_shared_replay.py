"""Cross-scene physics archives preserve chronology, IDs and episode boundaries."""
import pytest
import torch

from supermimic.rl.shared_replay import SharedReplay, save_replay
from supermimic.rl.shared_world_model import GeometryBank
from supermimic.rl.storage import ReplayBuffer


def make_archive(path, key, offset):
    bank = GeometryBank()
    bank.add(key, torch.full((1024, 3), float(offset)), torch.ones(3))
    buffer = ReplayBuffer(4, 2, "cpu", state_dim=84, context_dim=47)
    for step in range(6):
        s = torch.zeros(2, 84)
        s[:, 0] = step + offset
        s[:, 1] = torch.arange(2)
        s[:, 3] = s[:, 74] = 1
        xi = torch.zeros(2, 47)
        xi[:, 33], xi[:, 45:] = 10, 1
        next_state = s.clone()
        next_state[:, 0] += 1
        buffer.add(dict(s=s, s1=next_state, a=torch.zeros(2, 29), a_prev=torch.zeros(2, 29),
                        xi=xi, r=torch.zeros(2), f=torch.full((2,), step),
                        ep_step=torch.full((2,), step), done=torch.full((2,), step == 3)))
    save_replay(path, buffer, bank)


def test_archive_remapping_ring_order_and_episode_masks(tmp_path):
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    make_archive(a, "chair_a", 10)
    make_archive(b, "chair_b", 20)
    bank = GeometryBank()
    bank.add("third_shape", torch.zeros(1024, 3), torch.ones(3))
    mixed = SharedReplay((a, b), bank)
    assert bank.keys == ["__none__", "third_shape", "chair_a", "chair_b"]
    for archive, first, index in zip(mixed.archives, (12, 22), (2, 3)):
        torch.testing.assert_close(archive["s"][:, 0, 0], torch.arange(first, first + 4).float())
        assert (archive["xi"][..., 46] == index).all()
    samples = mixed.sample_windows(16, 4, "cpu")
    for row in range(16):
        assert (samples["xi"][row, :, 46] == samples["xi"][row, 0, 46]).all()
        assert (samples["s"][row, :, 1] == samples["s"][row, 0, 1]).all()
        torch.testing.assert_close(samples["s"][row, 1:, 0] - samples["s"][row, :-1, 0], torch.ones(3))
        torch.testing.assert_close(samples["mask"][row], torch.tensor([1., 1., 0., 0.]))


@pytest.mark.parametrize("field", ["schema", "geometry_id", "mask"])
def test_invalid_shared_replay_fails_before_training(tmp_path, field):
    path = tmp_path / "bad.pt"
    make_archive(path, "chair", 10)
    archive = torch.load(path, weights_only=True)
    if field == "schema":
        archive["schema"]["version"] = -1
    elif field == "geometry_id":
        archive["transitions"]["xi"][..., 46] = -1
    else:
        archive["transitions"]["xi"][..., 45] = .5
    torch.save(archive, path)
    with pytest.raises(ValueError):
        SharedReplay((path,), GeometryBank())
