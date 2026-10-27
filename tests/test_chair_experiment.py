"""Chair-level leakage prevention and one learner across changing physics scenes."""
from dataclasses import asdict
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from motion_data import load_motion, save_motion
from motion_data.chair_split import (cache_report, create_split, fingerprint, load_split,
                                    scenes, shape_key, verify_files)


@pytest.fixture
def chair_dataset(tmp_path):
    import trimesh
    from supermimic.assets.g1.g1_constants import G1_JOINT_NAMES, default_joint_pos_list
    root = tmp_path / "motions"
    frames = 40
    for i in range(5):
        mesh = trimesh.creation.box(extents=[.4 + .04 * i, .4, .5])
        for j in range(2):
            save_motion(root / f"shape{i}_clip{j}.npz", dict(
                fps=25., joint_names=G1_JOINT_NAMES,
                root_pos=np.tile([.1 * j, 0., 1.5], (frames, 1)),
                root_quat=np.tile([1., 0., 0., 0.], (frames, 1)),
                joint_pos=np.tile(default_joint_pos_list(), (frames, 1)),
                object_pos=np.tile([2., 0., .25], (frames, 1)),
                object_quat=np.tile([1., 0., 0., 0.], (frames, 1)), object_scale=np.ones(3),
                object_vertices=np.asarray(mesh.vertices), object_faces=np.asarray(mesh.faces),
            ))
    return root


def test_split_is_shape_disjoint_deterministic_relocatable_and_scale_grouped(chair_dataset, tmp_path):
    clip = load_motion(chair_dataset / "shape0_clip0.npz")
    save_motion(chair_dataset / "renamed_scaled_copy.npz", {**clip, "object_scale": np.full(3, 1.2)})
    path = tmp_path / "splits" / "split.json"
    first = create_split(chair_dataset, path, seed=0)
    second = create_split(chair_dataset, tmp_path / "another.json", seed=0)
    assert first["fingerprint"] == second["fingerprint"]
    assert (len(first["train"]), len(first["test"])) == (4, 1)
    train = {g["shape_key"] for g in first["train"]}
    test = {g["shape_key"] for g in first["test"]}
    assert not train & test
    group = next(g for g in first["train"] + first["test"] if g["shape_key"] == shape_key(clip))
    assert {c["path"] for c in group["clips"]} == {"shape0_clip0.npz", "shape0_clip1.npz", "renamed_scaled_copy.npz"}
    manifest, root = load_split(path)
    assert root == chair_dataset
    verify_files(manifest, root, "train")
    verify_files(manifest, root, "test")
    assert len(scenes(first, "train")) + len(scenes(first, "test")) == 6
    with pytest.raises(FileExistsError):
        create_split(chair_dataset, path)


def test_modified_references_and_leaking_manifests_are_rejected(chair_dataset, tmp_path):
    path = tmp_path / "split.json"
    split = create_split(chair_dataset, path)
    test_file = chair_dataset / split["test"][0]["clips"][0]["path"]
    clip = load_motion(test_file)
    save_motion(test_file, {**clip, "root_pos": clip["root_pos"] + .1})
    verify_files(split, chair_dataset, "train")  # Does not read held-out motions.
    with pytest.raises(ValueError, match="changed reference"):
        verify_files(split, chair_dataset, "test")
    split["test"].append(split["train"][0])
    split["fingerprint"] = fingerprint(split)
    path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match="overlap"):
        load_split(path)


def test_missing_assets_are_reported_without_changing_membership(chair_dataset, tmp_path):
    from supermimic.rl.chair_experiment import preflight
    split = create_split(chair_dataset, tmp_path / "split.json")
    report = cache_report(split, tmp_path / "cache")
    assert report["train"]["missing_collision_scenes"] == 4
    assert report["test"]["missing_collision_scenes"] == 1
    with pytest.raises(FileNotFoundError, match="4 collision assets missing"):
        preflight(split, chair_dataset, tmp_path / "cache", "train")


def test_balanced_scene_schedule_is_deterministic():
    from supermimic.rl.chair_experiment import ChairExperimentRunner
    runner = object.__new__(ChairExperimentRunner)
    runner.scenes, runner.schedule_seed = list(range(151)), 0
    first = [runner.scene_for_visit(i) for i in range(151)]
    second = [runner.scene_for_visit(i) for i in range(151, 302)]
    assert set(first) == set(second) == set(range(151))
    assert first != second
    assert first == [runner.scene_for_visit(i) for i in range(151)]


def _prepare_boxes(root, cache_dir, monkeypatch):
    import importlib.metadata
    import sys
    from motion_data import collision
    original = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version",
                        lambda n: collision.COACD_VERSION if n == "coacd" else original(n))
    # Fixture objects already are convex boxes; only replace the offline
    # decomposition stage. Runtime physics and contact computation remain real.
    monkeypatch.setitem(sys.modules, "coacd", SimpleNamespace(set_log_level=lambda _: None,
        Mesh=lambda v, f: (v, f), run_coacd=lambda mesh, **_: [mesh]))
    for path in root.glob("*.npz"):
        collision.prepare_collision(load_motion(path), cache_dir)


def _small_models(monkeypatch):
    from supermimic.rl.entity_policy import EntityPolicy, EntityCritic
    from supermimic.rl.entity_world_model import EntityWorldModel
    wm, actor, critic = EntityWorldModel.__init__, EntityPolicy.__init__, EntityCritic.__init__
    monkeypatch.setattr(EntityWorldModel, "__init__", lambda self, **kw: wm(self, width=32, layers=2, **kw))
    monkeypatch.setattr(EntityPolicy, "__init__", lambda self, *a, **kw: actor(self, *a, hidden=(32, 32), **kw))
    monkeypatch.setattr(EntityCritic, "__init__", lambda self, *a, **kw: critic(self, *a, hidden=(32, 32), **kw))


def test_native_multi_chair_learning_resume_and_frozen_unseen_evaluation(
    chair_dataset, tmp_path, cuda_device, monkeypatch,
):
    from supermimic.rl.chair_experiment import ChairExperimentRunner
    from supermimic.rl.config import SuperMimicRunnerCfg
    from supermimic.scripts.chair_experiment import evaluate_split
    split = create_split(chair_dataset, tmp_path / "split.json")
    cache = tmp_path / "cache"
    _prepare_boxes(chair_dataset, cache, monkeypatch)
    _small_models(monkeypatch)
    cfg = asdict(SuperMimicRunnerCfg(buffer_steps=12, utd=1., horizon=2, wm_window=2,
        wm_batches=1, wm_batch=2, pi_batch=2, warmup_steps=0, wm_only_iters=0,
        critic_epochs=1, critic_minibatches=1, no_compile=True, amp="none"))
    torch.manual_seed(0)
    run = tmp_path / "run"
    runner = ChairExperimentRunner(split, chair_dataset, cache, cfg, num_envs=2, max_objects=2,
                                    iterations_per_chair=1, log_dir=run, device=cuda_device)
    try:
        runner._init_training_models()
        networks = tuple(id(model) for model in (runner.pi, runner.wm, runner.critic))
        optimizers = tuple(id(opt) for opt in (runner.opt_pi, runner.opt_wm, runner.opt_v))
        runner.learn(4)
        assert len(runner.visited_shapes) == 4
        assert networks == tuple(id(model) for model in (runner.pi, runner.wm, runner.critic))
        assert optimizers == tuple(id(opt) for opt in (runner.opt_pi, runner.opt_wm, runner.opt_v))
        assert runner.pi.geometry_encoder is runner.wm.geometry_encoder is runner.critic.geometry_encoder
        bank_keys = set(runner.global_bank.keys[1:])
        assert bank_keys == {s["geometry_key"] for s in scenes(split, "train")}
        assert not bank_keys & {s["geometry_key"] for s in scenes(split, "test")}
        rows = runner.buffer._row(torch.arange(runner.buffer.size, device=cuda_device))
        xi = runner.buffer.data["xi"][rows]
        frames = runner.buffer.data["f"][rows]
        ids = runner.dynamics.object_context(xi)[..., 0, 12].long()
        assert ids.unique().numel() == 4
        for index, scene in enumerate(runner.scenes):
            selected = ids == runner.global_bank.keys.index(scene["geometry_key"])
            assert selected.any()
            assert (frames[selected] >= runner.offsets[index]).all()
            if index + 1 < len(runner.scenes):
                assert (frames[selected] < runner.offsets[index + 1]).all()
        changed = ids[1:] != ids[:-1]
        assert runner.buffer.data["done"][rows][:-1][changed].all()
        assert len(runner.opt_wm.state) > 0
        checkpoint = run / "model_4.pt"
        runner.save_world_model_replay(tmp_path / "all_train_replay.pt")
    finally:
        runner.close()
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    resumed = ChairExperimentRunner(split, chair_dataset, cache, cfg, num_envs=2, max_objects=2,
                                      iterations_per_chair=1, log_dir=run, device=cuda_device)
    try:
        resumed.load(checkpoint)
        assert resumed.current_learning_iteration == 4 and len(resumed.visited_shapes) == 4
        resumed.learn(1)
        assert resumed.active_visit == 4
        assert resumed.current_learning_iteration == 5
    finally:
        resumed.close()
    del resumed
    gc.collect()
    torch.cuda.empty_cache()
    output = tmp_path / "zero_shot.json"
    report = evaluate_split(split, chair_dataset, checkpoint, cache_dir=cache,
                             device=cuda_device, output=output)
    assert report["evaluated_chairs"] == 1 and not report["incomplete_training"]
    assert report["protocol"] == "held_out_geometry_frozen_policy"
    assert len(report["per_scene"][0]["per_clip"]) == 2
    assert report["visited_training_chairs"] == 4
    assert output.is_file()


def test_heldout_evaluation_rejects_wrong_split_or_partial_training_before_physics(chair_dataset, tmp_path):
    from supermimic.scripts.chair_experiment import evaluate_split
    split = create_split(chair_dataset, tmp_path / "split.json")
    path = tmp_path / "checkpoint.pt"
    torch.save({"task": {"chair_experiment": {"split_fingerprint": "wrong"}}}, path)
    with pytest.raises(ValueError, match="fingerprints differ"):
        evaluate_split(split, chair_dataset, path, cache_dir=tmp_path)
    torch.save({"task": {"chair_experiment": {"split_fingerprint": split["fingerprint"]}}}, path)
    with pytest.raises(ValueError, match="visited only 0/4"):
        evaluate_split(split, chair_dataset, path, cache_dir=tmp_path)
