"""Mixed-chair RSI must preserve the per-chair history EMA's configured rate."""

from types import SimpleNamespace

import torch

from supermimic.tasks.entities.mixed import ChairReferenceSampler, MixedChairCommandCfg


def make_sampler(num_scenes=83):
    # Two worlds and two reference bins per chair; all computation stays on CPU.
    cfg = MixedChairCommandCfg(
        scene_motion_files=tuple((f"scene_{scene}.npz",) for scene in range(num_scenes)),
        history_update=0.001,
    )
    motion = SimpleNamespace(
        num_bins=2 * num_scenes,
        bin_clip=torch.arange(num_scenes).repeat_interleave(2),
        bin_start=torch.arange(2 * num_scenes) * 5,
        bin_frames=5,
        clip_start=torch.arange(num_scenes) * 10,
        clip_end=torch.arange(num_scenes) * 10 + 9,
        clip_id=torch.arange(10 * num_scenes) // 10,
        frame_bin=torch.arange(10 * num_scenes) // 5,
    )
    env_scene = torch.arange(num_scenes).repeat_interleave(2)
    return ChairReferenceSampler(cfg, motion, len(env_scene), "cpu", 1.0,
                                 env_scene, torch.arange(num_scenes))


def test_history_initializes_as_a_probability_distribution_per_chair():
    sampler = make_sampler()
    torch.testing.assert_close(sampler.bin_history.reshape(83, 2), torch.full((83, 2), 0.5))


def test_history_ema_rate_does_not_scale_with_number_of_chairs():
    sampler = make_sampler()
    sampler.bin_returns[1] = 1.0
    expected = torch.lerp(torch.full((2,), 0.5),
                          torch.softmax(-sampler.cfg.temperature * torch.tensor([0.0, 1.0]), dim=0),
                          sampler.cfg.history_update)
    # Both worlds of the same chair reset together: this is one EMA update.
    frames, bins = sampler.sample_for_envs(torch.tensor([0, 1]))
    actual = sampler.bin_history[:2]
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=0)
    torch.testing.assert_close(actual / actual.sum(), expected, atol=1e-7, rtol=0)
    assert (sampler.bin_scene[bins] == 0).all()
    assert (sampler.clip_scene[sampler.motion.clip_id[frames]] == 0).all()


def test_resetting_one_chair_does_not_advance_other_chairs_history():
    sampler = make_sampler(num_scenes=2)
    # Set explicit distributions so this test isolates reset membership from
    # initialization. Both chairs would change if their histories were updated.
    sampler.bin_history[:] = torch.tensor([0.5, 0.5, 0.2, 0.8])
    sampler.bin_returns[:] = torch.tensor([0.0, 1.0, 0.0, 1.0])
    untouched = sampler.bin_history[2:].clone()
    sampler.sample_for_envs(torch.tensor([0]))
    assert not torch.equal(sampler.bin_history[:2], torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(sampler.bin_history[2:], untouched, atol=0, rtol=0)
