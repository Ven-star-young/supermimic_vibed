"""Reviewed membership reaches training without opening rejected candidates."""
import json
import shutil

import numpy as np
import pytest

from motion_data import save_motion
from motion_data.chair_split import verify_files
from motion_data.reference_training import load_reference_source


def _write_json(path, document):
    path.write_text(json.dumps(document))
    return path


@pytest.fixture
def reviewed(tmp_path):
    root = tmp_path / "motions"
    root.mkdir()
    names = ["sitting__chair_a__000.npz", "sitting__chair_a__001.npz", "sitting__chair_b__000.npz"]
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    for index, name in enumerate(names):
        save_motion(root / name, dict(fps=25., joint_names=["joint"], joint_pos=np.zeros((3, 1)),
            root_pos=np.zeros((3, 3)), root_quat=np.tile([1., 0., 0., 0.], (3, 1)),
            object_pos=np.zeros((3, 3)), object_quat=np.tile([1., 0., 0., 0.], (3, 1)),
            object_scale=np.ones(3), object_vertices=vertices * (1 + (index == 2)),
            object_faces=np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])))
    selection = _write_json(tmp_path / "selection.json", dict(version=1, source_dir=str(root),
        decisions={name: {"status": "keep"} for name in names}))
    choices = _write_json(tmp_path / "choices.json", dict(version=1, source_dir=str(root),
        selection_json=str(selection), choices={"chair_a": names[1]}))
    return root, selection, choices, names


def test_live_resolves_manual_and_single_without_opening_alternatives(reviewed):
    root, selection, choices, names = reviewed
    (root / names[0]).write_bytes(b"unreadable unselected candidate")
    (root / "unlisted.npz").write_bytes(b"not a reference")
    source, groups, provenance = load_reference_source(choices)
    assert source == root
    assert {group["clips"][0]["path"] for group in groups} == set(names[1:])
    assert len(groups) == 2 and all(len(group["clips"]) == 1 for group in groups)
    assert provenance["references"] == {"chair_a": names[1], "chair_b": names[2]}
    assert provenance["explicit_references"] == provenance["implicit_single_references"] == 1
    assert provenance["source_kind"] == "live_choices"
    assert str(root) not in json.dumps(provenance)
    verify_files({"train": groups}, source, "train")


@pytest.mark.parametrize("problem", ["pending", "no_longer_kept", "wrong_identity", "unsafe", "wrong_chair"])
def test_live_rejects_incomplete_or_changed_decisions(reviewed, problem):
    _, selection, choices, names = reviewed
    selected, initial = json.loads(choices.read_text()), json.loads(selection.read_text())
    if problem == "pending":
        selected["choices"] = {}
    elif problem == "no_longer_kept":
        initial["decisions"][names[1]]["status"] = "reject"
    elif problem == "wrong_identity":
        initial["source_dir"] = "/different/source"
    elif problem == "unsafe":
        selected["choices"]["chair_a"] = "../" + names[1]
    else:
        selected["choices"]["chair_a"] = names[2]
    _write_json(choices, selected)
    _write_json(selection, initial)
    with pytest.raises(ValueError):
        load_reference_source(choices)


def test_export_whitelist_and_relocated_live_sources(reviewed, tmp_path):
    root, selection, choices, names = reviewed
    old_root, old_groups, old_provenance = load_reference_source(choices)
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    for name in names[1:]:
        shutil.copy2(root / name, relocated / name)
    moved_selection = relocated / "selection.json"
    shutil.copy2(selection, moved_selection)
    selection.unlink()
    source, groups, provenance = load_reference_source(choices, relocated, selection_path=moved_selection)
    assert source == relocated and groups == old_groups and provenance == old_provenance
    export = _write_json(relocated / "manifest.json", dict(version=1,
        references={"chair_a": names[1], "chair_b": names[2]}))
    (relocated / "unexpected.npz").write_bytes(b"must not read")
    exported_source, exported_groups, exported_provenance = load_reference_source(export)
    assert exported_source == relocated and exported_groups == groups
    assert exported_provenance["source_kind"] == "export"
    assert load_reference_source(export, old_root)[1] == groups


@pytest.mark.parametrize("problem", ["missing", "invalid_motion", "duplicate_geometry"])
def test_selected_files_are_required_valid_and_one_per_geometry(reviewed, problem):
    root, _, choices, names = reviewed
    if problem == "missing":
        (root / names[2]).unlink()
    elif problem == "invalid_motion":
        np.savez(root / names[2], object_vertices=np.zeros((4, 3)))
    else:
        shutil.copy2(root / names[1], root / names[2])
    with pytest.raises((ValueError, FileNotFoundError)):
        load_reference_source(choices)


def test_modified_reference_detected_by_existing_split_verifier(reviewed):
    root, _, choices, names = reviewed
    _, groups, _ = load_reference_source(choices)
    with np.load(root / names[1], allow_pickle=False) as source:
        clip = dict(source)
    clip["root_pos"] += .1
    save_motion(root / names[1], clip)
    with pytest.raises(ValueError, match="changed reference"):
        verify_files({"train": groups}, root, "train")


@pytest.mark.parametrize("references", [[], {}, {"chair_a": "sitting__chair_b__000.npz"},
    {"chair_a": "sitting__chair_a__000.npz/other"}])
def test_export_rejects_malformed_mapping(tmp_path, references):
    path = _write_json(tmp_path / "manifest.json", dict(version=1, references=references))
    with pytest.raises(ValueError):
        load_reference_source(path)


def test_duplicate_json_keys_are_not_silently_overwritten(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"version":1,"references":{"chair_a":"first.npz","chair_a":"second.npz"}}')
    with pytest.raises(ValueError, match="Duplicate"):
        load_reference_source(path)
