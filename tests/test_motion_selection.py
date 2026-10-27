"""Manual selection must preserve source data and exact exported membership."""
import json

import pytest

from motion_data.selection import SelectionStore


@pytest.fixture
def selection(tmp_path):
    source = tmp_path / "motions"
    source.mkdir()
    for filename in ("c.npz", "a.npz", "b.npz"):
        (source / filename).write_bytes((filename + " original").encode())
    (source / "nested").mkdir()
    (source / "nested" / "ignored.npz").write_bytes(b"nested")
    (source / "directory.npz").mkdir()
    (source / "readme.txt").write_text("ignore")
    return SelectionStore(source, tmp_path / "decisions.json")


def test_autosave_resume_and_defensive_records(selection):
    assert selection.files == ("a.npz", "b.npz", "c.npz")
    assert selection.get("a.npz") == {"status": "unreviewed", "tags": [], "notes": ""}
    selection.set("a.npz", "keep", tags=["back_contact", "back_contact"], notes="椅背接触")
    selection.set("b.npz", "later", tags=["unclear"])
    resumed = SelectionStore(selection.source_dir, selection.output_json)
    assert resumed.summary() == {"unreviewed": 1, "keep": 1, "reject": 0, "later": 1, "total": 3}
    assert resumed.get("a.npz")["notes"] == "椅背接触"
    resumed.get("a.npz")["tags"].append("should_not_persist")
    assert resumed.set("a.npz", "reject")["tags"] == ["back_contact"]
    assert resumed.get("a.npz")["notes"] == "椅背接触"
    assert resumed.set("a.npz", "unreviewed", tags=[], notes="") == {
        "status": "unreviewed", "tags": [], "notes": ""
    }


@pytest.mark.parametrize("filename", ["../a.npz", "/a.npz", "nested/ignored.npz", "a\\b.npz", "missing.npz", 3])
def test_reject_unknown_and_unsafe_filename(selection, filename):
    with pytest.raises(ValueError, match="filename"):
        selection.set(filename, "keep")
    assert not selection.output_json.exists()


@pytest.mark.parametrize("arguments", [
    {"status": "selected"}, {"status": "keep", "tags": "back_contact"},
    {"status": "keep", "tags": [""]}, {"status": "keep", "tags": [1]},
    {"status": "keep", "notes": []},
])
def test_reject_invalid_decisions(selection, arguments):
    with pytest.raises(ValueError):
        selection.set("a.npz", **arguments)
    assert selection.get("a.npz")["status"] == "unreviewed"


def test_reject_saved_source_mismatch_and_unknown_names(selection, tmp_path):
    selection.set("a.npz", "keep")
    other = tmp_path / "other"
    other.mkdir()
    (other / "a.npz").write_bytes(b"other motion")
    with pytest.raises(ValueError, match="source directory"):
        SelectionStore(other, selection.output_json)
    saved = json.loads(selection.output_json.read_text())
    saved["decisions"]["../outside.npz"] = saved["decisions"]["a.npz"]
    selection.output_json.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="filename"):
        SelectionStore(selection.source_dir, selection.output_json)


def test_failed_atomic_save_preserves_previous_decision(selection, monkeypatch):
    selection.set("a.npz", "keep")
    previous = selection.output_json.read_bytes()

    def fail_replace(*args):
        raise OSError("disk full")

    monkeypatch.setattr("motion_data.selection.os.replace", fail_replace)
    with pytest.raises(OSError, match="disk full"):
        selection.set("a.npz", "reject")
    assert selection.get("a.npz")["status"] == "keep"
    assert selection.output_json.read_bytes() == previous
    assert not list(selection.output_json.parent.glob(".selection-*"))


@pytest.mark.parametrize("mode", ["symlink", "copy"])
def test_export_exact_membership_and_new_version(selection, tmp_path, mode):
    originals = {name: (selection.source_dir / name).read_bytes() for name in selection.files}
    selection.set("a.npz", "keep", tags=["back_contact"])
    selection.set("b.npz", "keep")
    first = selection.export(tmp_path / "exports", mode)
    assert first.name == "selection-0001"
    assert sorted(p.name for p in first.glob("*.npz")) == ["a.npz", "b.npz"]
    for name in ("a.npz", "b.npz"):
        assert (first / name).is_symlink() == (mode == "symlink")
        assert (first / name).read_bytes() == originals[name]
        if mode == "symlink":
            assert (first / name).readlink() == selection.source_dir / name
    selection.set("b.npz", "reject")
    second = selection.export(tmp_path / "exports", mode)
    assert second.name == "selection-0002"
    assert sorted(p.name for p in second.glob("*.npz")) == ["a.npz"]
    assert sorted(p.name for p in first.glob("*.npz")) == ["a.npz", "b.npz"]
    assert (second / "kept.txt").read_text() == "a.npz\n"
    manifest = json.loads((second / "manifest.json").read_text())
    assert manifest["files"] == ["a.npz"]
    assert manifest["decisions"]["a.npz"]["tags"] == ["back_contact"]
    assert {name: (selection.source_dir / name).read_bytes() for name in selection.files} == originals


def test_failed_export_removes_only_partial_new_directory(selection, tmp_path, monkeypatch):
    selection.set("a.npz", "keep")
    previous = selection.export(tmp_path / "exports", "copy")
    selection.set("b.npz", "keep")
    original_copy = __import__("shutil").copy2

    def fail_second(source, destination):
        if source.name == "b.npz":
            raise OSError("copy interrupted")
        return original_copy(source, destination)

    monkeypatch.setattr("motion_data.selection.shutil.copy2", fail_second)
    with pytest.raises(OSError, match="copy interrupted"):
        selection.export(tmp_path / "exports", "copy")
    assert list((tmp_path / "exports").iterdir()) == [previous]
    assert (previous / "a.npz").read_bytes() == (selection.source_dir / "a.npz").read_bytes()


def test_empty_selection_and_invalid_mode_do_not_create_export(selection, tmp_path):
    output = tmp_path / "exports"
    with pytest.raises(ValueError, match="No kept"):
        selection.export(output)
    with pytest.raises(ValueError, match="mode"):
        selection.export(output, "move")
    assert not output.exists()
