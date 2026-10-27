"""Real viewer/store integration; downloaded sitting assets are optional."""
from argparse import Namespace
import json
from pathlib import Path
import socket
import time

import numpy as np
import pytest

pytest.importorskip("viser")
pytest.importorskip("mujoco")
pytest.importorskip("scipy")

from motion_data import load_motion, save_motion
from motion_data.selection import SelectionStore
from tools.select_motions import MotionSelector


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sitting_assets():
    model = ROOT / "data/g1_description/g1_29dof.xml"
    source = ROOT / "data/motions/sitting"
    motions = sorted(source.glob("*.npz"))[:3]
    if not model.is_file() or len(motions) < 3:
        pytest.skip("Downloaded G1 model and three sitting motions are required")
    near = source / "sitting__barnwood_reclaimed_plank_chair_iron_fasteners__008.npz"
    if near.is_file() and near not in motions:
        motions[-1] = near
    return model, motions


@pytest.fixture
def selector_factory(tmp_path, sitting_assets):
    model, motions = sitting_assets
    instances = []

    def create(*, extra=None):
        run_dir = tmp_path / str(len(instances))
        source = run_dir / "motions"
        source.mkdir(parents=True)
        for motion in motions:
            (source / motion.name).symlink_to(motion)
        if extra is not None:
            extra(source, motions[0])
        # Viser accepts a concrete port; reserve an available OS-assigned one.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        args = Namespace(motion=source, model=model, selection=run_dir / "selection.json",
                         export_dir=run_dir / "exports", host="127.0.0.1", port=port)
        app = MotionSelector(args)
        instances.append(app)
        return app

    yield create
    for app in instances:
        app.close()


def edit(app, handle, action, value):
    """Mimic a client updating its handle before its queued event is processed."""
    handle.value = value
    app.process_event(action, (app.current, value))


def assert_picker_aligned(app):
    assert app.labels[app.picker.value] == app.current


def test_playback_and_paused_threshold_refresh_scene(selector_factory):
    app = selector_factory()
    assert_picker_aligned(app)
    app.process_event("step", (app.current, "后一帧"))
    assert app.frame == 1
    assert not app.playing.value
    np.testing.assert_allclose(app.data.qpos[app.root_adr:app.root_adr + 3],
                               app.clip["root_pos"][1])
    app.process_event("step", (app.current, "前一帧"))
    assert app.frame == 0
    app.process_event("step", (app.current, "最贴近帧"))
    assert app.frame == int(np.argmin(app.analysis["distances"]))

    # Real near-contact frames exercise a visible threshold transition.
    candidates = []
    for name in app.store.files:
        target = next(label for label, item in app.labels.items() if item == name)
        edit(app, app.picker, "choose", target)
        distances = app.analysis["distances"]
        candidates = np.flatnonzero((distances > .005) & (distances <= .1))
        if len(candidates):
            break
    assert len(candidates), "Real fixture clips must include a frame between 0.5 and 10 cm"
    edit(app, app.slider, "frame", int(candidates[0]))
    edit(app, app.threshold, "threshold", .5)
    np.testing.assert_array_equal(app.contact_line.colors, [250, 182, 70])
    edit(app, app.threshold, "threshold", 10.)
    np.testing.assert_array_equal(app.contact_line.colors, [40, 215, 150])
    np.testing.assert_array_equal(app.contact_points.colors, [40, 215, 150])
    assert not app.playing.value
    edit(app, app.show_back, "overlay", False)
    assert not app.back_patch.visible and not app.contact_line.visible
    edit(app, app.show_back, "overlay", True)
    assert app.back_patch.visible and app.contact_line.visible


def test_review_autonext_resume_and_export(selector_factory):
    app = selector_factory()
    first, second, third = app.store.files
    edit(app, app.status_filter, "filter", "未筛选")
    edit(app, app.tag, "tag", "背部接触")
    edit(app, app.notes, "notes", "背部靠住椅背，保留")
    app.process_event("decision", (first, "keep"))
    assert app.current == second
    assert_picker_aligned(app)
    assert first not in app.filtered
    assert app.notes.value == ""
    assert app.tag.value == "未标注"
    resumed = SelectionStore(app.args.motion, app.args.selection)
    assert resumed.get(first) == {"status": "keep", "tags": ["back_contact"],
                                   "notes": "背部靠住椅背，保留"}
    app.process_event("decision", (second, "reject"))
    assert app.current == third
    assert_picker_aligned(app)
    app.process_event("export", (app.current, None))
    exported, = app.args.export_dir.iterdir()
    assert [path.name for path in exported.glob("*.npz")] == [first]
    assert (exported / first).is_symlink()
    assert (exported / first).resolve() == (app.args.motion / first).resolve()
    manifest = json.loads((exported / "manifest.json").read_text())
    assert manifest["files"] == [first]
    assert manifest["decisions"][first]["tags"] == ["back_contact"]


def test_corrupt_choice_restores_picker_and_does_not_label_bad_motion(selector_factory):
    broken = "zz_corrupt.npz"

    def add_broken(source, _):
        (source / broken).write_bytes(b"not an NPZ archive")

    app = selector_factory(extra=add_broken)
    previous = app.current
    app.playing.value = False
    target = next(label for label, name in app.labels.items() if name == broken)
    app.picker.value = target
    app.events.put(("choose", (previous, target)))
    app.tick()  # The production dispatch loop catches load errors.
    assert app.current == previous
    assert_picker_aligned(app)
    assert app.store.get(broken)["status"] == "unreviewed"
    app.auto_next.value = False
    app.process_event("decision", (app.current, "keep"))
    assert app.store.get(previous)["status"] == "keep"
    assert app.store.get(broken)["status"] == "unreviewed"


def test_single_frame_load_refreshes_proximity_even_when_throttled(selector_factory):
    single = "zz_single_frame.npz"

    def add_single(source, original):
        clip = load_motion(original)
        for key in ("root_pos", "root_quat", "joint_pos", "object_pos", "object_quat"):
            clip[key] = clip[key][:1].copy()
        clip["root_pos"][:, 2] += 2.0  # Visibly distinct separation from the previous clip.
        save_motion(source / single, clip)

    app = selector_factory(extra=add_single)
    app.playing.value = True
    app.proximity_info.content = "stale previous motion"
    # Make the render throttle deterministic; explicit load refresh must bypass it.
    app.last_info_time = time.monotonic() + 60
    target = next(label for label, name in app.labels.items() if name == single)
    edit(app, app.picker, "choose", target)
    assert app.current == single
    assert_picker_aligned(app)
    assert app.slider.disabled and app.frame == 0
    assert "stale" not in app.proximity_info.content
    assert f"{app.analysis['distances'][0] * 100:.1f} cm" in app.proximity_info.content
