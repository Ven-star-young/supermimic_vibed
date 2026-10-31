"""Failed physics must terminate replay sequences without becoming training targets."""

from types import SimpleNamespace

import pytest
import torch

from supermimic.rl.models import RunningNorm
from supermimic.rl.runner import SuperMimicRunner
from supermimic.rl.storage import ReplayBuffer


def _add(replay, step, *, valid=True, done=False, episode_step=None):
    transition = {
        key: torch.zeros_like(value[0]) for key, value in replay.data.items()
    }
    transition.update(
        s=torch.full((1, 1), float(step)),
        s1=torch.full((1, 1), float(step + 1 if valid else step)),
        f=torch.tensor([step]),
        ep_step=torch.tensor([step if episode_step is None else episode_step], dtype=torch.int32),
        done=torch.tensor([done]),
        valid=torch.tensor([valid]),
    )
    replay.add(transition)


def _buffer(length=4):
    return ReplayBuffer(length, 1, "cpu", state_dim=1, action_dim=1, context_dim=1)


@pytest.mark.parametrize("invalid_step", [0, 1, 3])
def test_failed_transition_and_following_reset_episode_are_excluded(invalid_step):
    replay = _buffer()
    for step in range(4):
        _add(replay, step, valid=step != invalid_step, done=step == invalid_step)

    window = replay.sample_windows(2, 4)
    expected = (torch.arange(4) < invalid_step).float().expand(2, -1)
    torch.testing.assert_close(window["mask"], expected)


def test_valid_terminal_transition_remains_a_world_model_target():
    replay = _buffer()
    for step in range(4):
        _add(replay, step, done=step == 1)

    window = replay.sample_windows(2, 4)
    torch.testing.assert_close(window["mask"], torch.tensor([[1., 1., 0., 0.]]).expand(2, -1))


def test_invalid_transition_breaks_model_rollout_even_without_done_flag():
    replay = _buffer()
    for step in range(4):
        _add(replay, step, valid=step != 1)

    window = replay.sample_windows(2, 4)
    # There is no usable dynamics transition linking the prediction before the
    # failure to later targets, even if an external producer omits done=True.
    torch.testing.assert_close(window["mask"], torch.tensor([[1., 0., 0., 0.]]).expand(2, -1))


def test_failure_mask_tracks_chronological_order_after_ring_wrap():
    replay = _buffer(3)
    for step in range(5):
        _add(replay, step, valid=step != 3, done=step == 3)

    window = replay.sample_windows(2, 3)
    torch.testing.assert_close(window["s"][..., 0], torch.tensor([[2., 3., 4.]]).expand(2, -1))
    torch.testing.assert_close(window["mask"], torch.tensor([[1., 0., 0.]]).expand(2, -1))


@pytest.mark.parametrize("invalid_step", [0, 1])
def test_failed_transition_has_no_loss_or_gradient(invalid_step):
    replay = _buffer()
    for step in range(4):
        _add(replay, step, valid=step != invalid_step, done=step == invalid_step)
    window = replay.sample_windows(1, 4)
    # A large finite prediction error on the safe placeholder must not train the
    # model to hold still when the physical transition was actually unusable.
    error = torch.tensor([[[2.], [1000.], [1000.], [1000.]]], requires_grad=True)
    runner = SimpleNamespace(
        wm=SimpleNamespace(out_norm=SimpleNamespace(std=lambda: torch.ones(1)),
                           state_error=lambda predicted, actual: predicted - actual),
        dynamics=SimpleNamespace(unpack=lambda values: values),
        error_floor=torch.ones(1),
    )
    loss, _ = SuperMimicRunner._wm_error(
        runner, window["s1"] + error, window["s1"], window["mask"],
    )
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(4. if invalid_step else 0.))
    expected_gradient = torch.zeros_like(error)
    if invalid_step:
        expected_gradient[:, 0] = 4.
    torch.testing.assert_close(error.grad, expected_gradient)


def _choose_ages(monkeypatch, ages):
    """Control the two random draws while retaining real replay/history logic."""
    draws = iter((torch.tensor(ages), torch.zeros(len(ages), dtype=torch.long)))
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: next(draws))


def test_actor_samples_expose_validity_and_history_stops_at_failure_reset(monkeypatch):
    replay = _buffer()
    _add(replay, 0)
    _add(replay, 1, valid=False, done=True)
    _add(replay, 2, episode_step=0)
    _add(replay, 3, episode_step=1)
    _choose_ages(monkeypatch, [1, 2, 3])

    start, history = replay.sample_starts(3, 3)
    torch.testing.assert_close(start["valid"], torch.tensor([False, True, True]))
    torch.testing.assert_close(history["s"][1, :, 0], torch.tensor([2., 2., 2.]))
    torch.testing.assert_close(history["s"][2, :, 0], torch.tensor([2., 2., 3.]))


def test_actor_imagination_filters_failed_starts_with_sufficient_reference(monkeypatch):
    replay = _buffer()
    for step in range(4):
        _add(replay, step, valid=step != 1, done=step == 1,
             episode_step=step if step < 2 else step - 2)
    _choose_ages(monkeypatch, [0, 1, 2, 3])
    runner = SimpleNamespace(
        cfg=SimpleNamespace(pi_batch=2, horizon=2), buffer=replay, history_length=3,
        motion=SimpleNamespace(clip_id=torch.zeros(100, dtype=torch.long), clip_end=torch.tensor([100])),
    )

    starts, _ = SuperMimicRunner._sample_valid_starts(runner)
    torch.testing.assert_close(starts["f"], torch.tensor([0, 2]))
    assert starts["valid"].all()


def test_no_valid_actor_start_reports_failure_instead_of_imagining_from_placeholder():
    replay = _buffer(1)
    _add(replay, 0, valid=False, done=True)
    runner = SimpleNamespace(
        cfg=SimpleNamespace(pi_batch=1, horizon=2), buffer=replay, history_length=3,
        motion=SimpleNamespace(clip_id=torch.zeros(100, dtype=torch.long), clip_end=torch.tensor([100])),
    )
    with pytest.raises(RuntimeError, match="Replay has too few"):
        SuperMimicRunner._sample_valid_starts(runner)


def test_empty_valid_batch_leaves_single_rank_normalizer_unchanged():
    norm = RunningNorm(2)
    norm.update(torch.tensor([[1., 2.], [3., 4.]]))
    before = {key: value.clone() for key, value in norm.state_dict().items()}
    norm.update(torch.empty(0, 2))
    for key, value in norm.state_dict().items():
        torch.testing.assert_close(value, before[key])
