"""Final references retain original chair partitions and reach runtime file selection."""
import json

import numpy as np
import pytest

from motion_data import load_motion, save_motion
from motion_data.chair_split import create_split, load_split, scenes, verify_files
from tools.prepare_reference_training import prepare_reference_split


@pytest.fixture
def references_and_split(tmp_path):
    root = tmp_path / "motions"
    refs = {}
    for chair in range(5):
        for index in range(2):
            name = f"sitting__chair{chair}__{index:03d}.npz"
            save_motion(root / name, dict(fps=25., joint_names=["joint"],
                joint_pos=np.zeros((3, 1)), root_pos=np.full((3, 3), index * .1),
                root_quat=np.tile([1., 0., 0., 0.], (3, 1)), object_pos=np.zeros((3, 3)),
                object_quat=np.tile([1., 0., 0., 0.], (3, 1)), object_scale=np.ones(3),
                object_vertices=np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]) * (1 + chair),
                object_faces=np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])))
            refs[f"chair{chair}"] = name  # All choices intentionally use 001.
    base = tmp_path / "base.json"
    create_split(root, base)
    source = root / "manifest.json"
    source.write_text(json.dumps(dict(version=1, references=refs)))
    return root, source, base


def test_preserves_partitions_and_runtime_uses_exact_reference(references_and_split, tmp_path):
    from supermimic.rl.chair_experiment import scene_cfg
    from supermimic.tasks.interaction.assets import select_motion_files
    root, source, base = references_and_split
    output = tmp_path / "selected.json"
    manifest, selected_root = prepare_reference_split(source, base, output)
    original, _ = load_split(base)
    assert selected_root == root
    assert manifest["parent_split_fingerprint"] == original["fingerprint"]
    for partition in ("train", "test"):
        assert [g["shape_key"] for g in manifest[partition]] == [g["shape_key"] for g in original[partition]]
        verify_files(manifest, root, partition)
        for scene in scenes(manifest, partition):
            assert len(scene["clips"]) == 1
            chosen = scene["clips"][0]["path"]
            assert chosen.endswith("__001.npz")
            cfg = scene_cfg(root, scene, cache_dir=tmp_path / "cache", num_envs=2)
            assert select_motion_files(cfg.commands["motion"]) == [root / chosen]
            assert cfg.commands["motion"].object_motion_file == str(root / chosen)
    assert load_split(output)[0] == manifest
    assert prepare_reference_split(source, base, output)[0] == manifest  # Safe repeat.


def test_changed_choice_cannot_overwrite_existing_experiment(references_and_split, tmp_path):
    _, source, base = references_and_split
    output = tmp_path / "selected.json"
    prepare_reference_split(source, base, output)
    before = output.read_bytes()
    values = json.loads(source.read_text())
    values["references"]["chair0"] = "sitting__chair0__000.npz"
    source.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="different reference choices"):
        prepare_reference_split(source, base, output)
    assert output.read_bytes() == before


def test_changed_file_or_unknown_chair_cannot_reuse_base_partition(references_and_split, tmp_path):
    root, source, base = references_and_split
    name = "sitting__chair0__001.npz"
    clip = load_motion(root / name)
    save_motion(root / name, {**clip, "root_pos": clip["root_pos"] + .01})
    output = tmp_path / "selected.json"
    with pytest.raises(ValueError, match="differs from the original split"):
        prepare_reference_split(source, base, output)
    assert not output.exists()
