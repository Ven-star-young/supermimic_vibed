"""One-reference-per-chair persistence and grouped playback integration."""
from argparse import Namespace
from pathlib import Path
import socket
import time

import pytest

from motion_data.references import ChairReferences
from motion_data.selection import SelectionStore


@pytest.fixture
def selected(tmp_path):
    source = tmp_path / "motions"
    source.mkdir()
    names = ["sitting__chair_a__000.npz", "sitting__chair_a__001.npz",
             "sitting__chair_b__003.npz", "sitting__chair_c__000.npz"]
    for name in names:
        (source / name).write_bytes(b"example motion")
    store = SelectionStore(source, tmp_path / "selection.json")
    for name in names[:3]:
        store.set(name, "keep")
    return store, names


def test_only_kept_candidates_one_choice_per_chair_and_resume(selected, tmp_path):
    selection, names = selected
    original = selection.output_json.read_bytes()
    path = tmp_path / "references.json"
    refs = ChairReferences(selection, path)
    assert refs.summary() == dict(chairs=2, candidates=3, single=1, resolved=1, pending=1)
    assert refs.resolved("chair_b") == names[2]
    with pytest.raises(ValueError, match="Only manually kept"):
        refs.choose(names[3])
    refs.choose(names[0])
    refs.choose(names[1])
    resumed = ChairReferences(selection, path)
    assert resumed.resolved("chair_a") == names[1]
    assert resumed.summary()["resolved"] == 2
    assert selection.output_json.read_bytes() == original
    resumed.clear("chair_a")
    assert ChairReferences(selection, path).resolved("chair_a") is None


def test_export_is_exactly_one_per_chair_and_old_choice_cannot_leak(selected, tmp_path):
    selection, names = selected
    path = tmp_path / "refs.json"
    refs = ChairReferences(selection, path)
    with pytest.raises(ValueError, match="待选"):
        refs.export(tmp_path / "exports")
    refs.choose(names[0])
    first = refs.export(tmp_path / "exports")
    refs.choose(names[1])
    second = refs.export(tmp_path / "exports", mode="copy")
    assert sorted(p.name for p in first.glob("*.npz")) == [names[0], names[2]]
    assert sorted(p.name for p in second.glob("*.npz")) == [names[1], names[2]]
    assert (first / names[0]).is_symlink()
    assert not (second / names[1]).is_symlink()
    selection.set(names[1], "reject")
    reloaded = ChairReferences(selection, path)
    assert reloaded.resolved("chair_a") == names[0]  # Remaining unique kept candidate.
    selection.set(names[0], "reject")
    assert "chair_a" not in ChairReferences(selection, path).groups


def test_reference_file_cannot_overwrite_original_selection(selected):
    selection, _ = selected
    with pytest.raises(ValueError, match="separate file"):
        ChairReferences(selection, selection.output_json)


def test_grouped_viewer_navigation_choice_and_loop(tmp_path):
    pytest.importorskip("viser")
    from tools.review_chairs import ChairReviewer, ROOT
    originals = [ROOT / "data/motions/sitting" / name for name in (
        "sitting__acadian_rustic_wooden_dining_chair_distressed_white__000.npz",
        "sitting__acadian_rustic_wooden_dining_chair_distressed_white__001.npz",
        "sitting__barnwood_reclaimed_plank_chair_iron_fasteners__008.npz")]
    model = ROOT / "data/g1_description/g1_29dof.xml"
    if not model.exists() or not all(p.exists() for p in originals):
        pytest.skip("Downloaded sitting assets are required")
    source = tmp_path / "motions"
    source.mkdir()
    for p in originals:
        (source / p.name).symlink_to(p)
    selection = SelectionStore(source, tmp_path / "selection.json")
    for p in originals:
        selection.set(p.name, "keep")
    original_bytes = selection.output_json.read_bytes()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    args = Namespace(motion=source, model=model, selection=selection.output_json,
        references=tmp_path / "refs.json", export_dir=tmp_path / "exports", host="127.0.0.1", port=port)
    app = ChairReviewer(args)
    try:
        assert len(app.visible_chairs) == 1
        assert app.filtered == [p.name for p in originals[:2]]
        app.process_event("frame", (app.current, 60))
        app.navigate("下一段")
        assert app.current == originals[1].name and app.frame == 60
        assert not app.playing.value
        # Restart must restart THIS clip, not trigger the sequential loop.
        app.events.put(("step", (app.current, "重播")))
        app.tick()
        assert app.current == originals[1].name and app.frame == 0
        # Natural end-of-clip wraps only within the selected chair.
        app.playhead = 124.5
        app.last_time = time.monotonic() - .1
        app.tick()
        assert app.current == originals[0].name and app.frame == 0
        app.auto_next_chair.value = False
        app.process_event("reference", (app.current, None))
        assert app.references.summary()["pending"] == 0
        app.navigate("下一段")
        app.process_event("reference", (app.current, None))
        assert app.references.resolved(app.current_chair) == originals[1].name
        app.process_event("export_references", (app.current, None))
        exported = next(args.export_dir.iterdir())
        assert sorted(p.name for p in exported.glob("*.npz")) == [p.name for p in originals[1:]]
        assert selection.output_json.read_bytes() == original_bytes
    finally:
        app.close()
