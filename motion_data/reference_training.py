"""Resolve reviewed chair references without choosing a training/test partition.

Only selected files are opened. Saved review metadata may live separately from
the motion files, so both exported symlinks and relocated source data work.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import load_motion
from .chair_split import file_digest, scaled_key, shape_key
from .collision import collision_key
from .references import chair_id
from .selection import STATUSES


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate reference metadata key: {key}")
        result[key] = value
    return result


def _read_metadata(path):
    raw = path.read_bytes()
    document = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(document, dict) or document.get("version") != 1:
        raise ValueError(f"Unsupported reference metadata format: {path}")
    return document, hashlib.sha256(raw).hexdigest()


def _motion_name(name, chair=None):
    if (not isinstance(name, str) or not name or Path(name).name != name
            or any(character in name for character in ("/", "\\"))
            or any(ord(character) < 32 for character in name)
            or not name.endswith(".npz")):
        raise ValueError(f"Unsafe motion filename: {name!r}")
    actual_chair = chair_id(name)
    if chair is not None and (not isinstance(chair, str) or actual_chair != chair):
        raise ValueError(f"Reference chair does not match filename: {chair!r}: {name}")
    return actual_chair


def _saved_path(value, parent, field):
    if not isinstance(value, str) or not value:
        raise ValueError(f"Reference metadata requires {field}")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else parent / path).resolve()


def _live_references(document, selection):
    if not isinstance(document.get("source_dir"), str) or not document["source_dir"]:
        raise ValueError("Reference metadata requires source_dir")
    if document["source_dir"] != selection.get("source_dir"):
        raise ValueError("Reference and selection source identities differ")
    choices, decisions = document.get("choices"), selection.get("decisions")
    if not isinstance(choices, dict) or not isinstance(decisions, dict):
        raise ValueError("Reference choices and selection decisions must be objects")
    candidates = {}
    for name, decision in decisions.items():
        chair = _motion_name(name)
        if not isinstance(decision, dict) or decision.get("status") not in STATUSES:
            raise ValueError(f"Invalid selection decision: {name}")
        if decision["status"] == "keep":
            candidates.setdefault(chair, []).append(name)
    for chair, name in choices.items():
        _motion_name(name, chair)
        if name not in candidates.get(chair, []):
            raise ValueError(f"Explicit reference is no longer kept: {name}")
    references, pending = {}, []
    for chair, names in sorted(candidates.items()):
        if chair in choices:
            references[chair] = choices[chair]
        elif len(names) == 1:
            references[chair] = names[0]
        else:
            pending.append(chair)
    if pending:
        raise ValueError(f"Incomplete reference selection: {len(pending)} chairs pending; first: {pending[0]}")
    return references, len(choices), len(references) - len(choices)


def _export_references(document):
    references = document.get("references")
    if not isinstance(references, dict):
        raise ValueError("Export references must be an object")
    for chair, name in references.items():
        _motion_name(name, chair)
    candidates = document.get("candidates")
    if candidates is not None:
        if not isinstance(candidates, dict) or candidates.keys() != references.keys():
            raise ValueError("Export candidates and references must contain the same chairs")
        for chair, names in candidates.items():
            if not isinstance(names, list) or not names:
                raise ValueError(f"Invalid exported candidates for {chair}")
            for name in names:
                _motion_name(name, chair)
            if len(set(names)) != len(names) or references[chair] not in names:
                raise ValueError(f"Export reference is not a unique kept candidate: {chair}")
    implicit = document.get("implicit_single_candidates", [])
    if (not isinstance(implicit, list) or any(not isinstance(chair, str) for chair in implicit)
            or len(set(implicit)) != len(implicit) or not set(implicit) <= references.keys()):
        raise ValueError("Invalid implicit single-candidate chairs")
    if candidates is not None and any(len(candidates[chair]) != 1 for chair in implicit):
        raise ValueError("An implicit reference must have exactly one candidate")
    return dict(sorted(references.items())), len(references) - len(implicit), len(implicit)


def load_reference_source(path, data_root=None, *, selection_path=None):
    """Return ``(motion_root, chair_groups, provenance)`` for completed reviews.

    ``path`` is an exported ``manifest.json`` or the live reference choices JSON.
    ``data_root`` relocates only the selected NPZ files; ``selection_path`` may
    relocate the initial decisions JSON for live choices. Neither override
    changes the saved source identities or the partition-independent provenance.
    """
    path = Path(path).expanduser().resolve()
    document, source_digest = _read_metadata(path)
    provenance = {"source_sha256": source_digest}
    if "references" in document and "choices" not in document:
        if selection_path is not None:
            raise ValueError("selection_path applies only to live reference choices")
        references, explicit, implicit = _export_references(document)
        root = path.parent
        provenance["source_kind"] = "export"
    elif "choices" in document and "references" not in document:
        saved_selection = _saved_path(document.get("selection_json"), path.parent, "selection_json")
        selected_metadata = Path(selection_path).expanduser().resolve() if selection_path is not None else saved_selection
        selection, selection_digest = _read_metadata(selected_metadata)
        references, explicit, implicit = _live_references(document, selection)
        root = _saved_path(document["source_dir"], path.parent, "source_dir")
        provenance.update(source_kind="live_choices", selection_sha256=selection_digest)
    else:
        raise ValueError("Expected exported references or live reference choices")
    if not references:
        raise ValueError("No selected chair references")
    if data_root is not None:
        root = Path(data_root).expanduser().resolve()
    groups = {}
    for chair, name in references.items():
        filename = root / name
        if not filename.is_file():
            raise FileNotFoundError(f"Missing selected reference: {filename}")
        clip = load_motion(filename)
        if not {"object_vertices", "object_faces", "object_scale", "object_pos", "object_quat"} <= clip.keys():
            raise ValueError(f"Selected reference has no complete chair geometry: {filename}")
        key = shape_key(clip)
        if key in groups:
            raise ValueError(f"Multiple chair references have duplicate mesh geometry: {chair}")
        groups[key] = dict(shape_key=key, clips=[dict(path=name, sha256=file_digest(filename),
            geometry_key=scaled_key(clip), collision_key=collision_key(clip))])
    provenance.update(references=references, explicit_references=explicit,
                      implicit_single_references=implicit)
    return root, [groups[key] for key in sorted(groups)], provenance
