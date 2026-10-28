"""Reproducible evaluation seeds and conservative, chair-balanced summaries.

Successful trials are divided by *planned* trials, including interruptions and
unevaluable contact labels. Coverage is reported alongside each success rate.
Chairs, rather than frames, environments or reference clips, are the units of
the task-level macro average. This module does not import simulation libraries.
"""
from __future__ import annotations

from collections import defaultdict
import json
from numbers import Integral
import os
from pathlib import Path
import tempfile


def _seed(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < 2**32:
        raise ValueError("Seeds must be integers in [0, 2**32)")
    return int(value)


def planned_seeds(base_seed: int, episodes: int, explicit=None) -> list[int]:
    """Use the same seed list for each reference; reset the RNG for each trial.

    Explicit seeds must match the requested trial count and must be distinct.
    Generated seeds wrap in uint32 space, preserving uniqueness.
    """
    base_seed = _seed(base_seed)
    if isinstance(episodes, bool) or not isinstance(episodes, Integral) or not 1 <= episodes <= 2**32:
        raise ValueError("Evaluation episodes must be a positive integer <= 2**32")
    seeds = ([(base_seed + index) % 2**32 for index in range(episodes)]
             if explicit is None else [_seed(value) for value in explicit])
    if len(seeds) != episodes or len(set(seeds)) != len(seeds):
        raise ValueError("Explicit seeds must be distinct and match the episode count")
    return seeds


_COUNTS = ("planned_trials", "executed_trials", "unexecuted_trials", "invalid_trials",
           "full_reference_eligible_trials", "full_reference_invalid_trials", "full_reference_successes",
           "stable_contact_eligible_trials", "stable_contact_invalid_trials", "stable_contact_successes")
_INVALID_OUTCOMES = {"error", "invalid", "physics_error", "physics_overflow", "nonfinite",
                     "interrupted", "manual_interruption", "manual_interrupted",
                     "viewer_closed", "manual_reset", "simulation_error", "evaluation_error"}


def _rates(counts):
    denominator = counts["planned_trials"]
    return dict(counts, **{
        f"{metric}_{suffix}": counts[f"{metric}_{count}"] / denominator if denominator else None
        for metric in ("full_reference", "stable_contact")
        for suffix, count in (("success_rate", "successes"), ("coverage", "eligible_trials"))
    })


def _trial_counts(condition, fallback_seeds):
    seeds = condition.get("planned_seeds", fallback_seeds)
    if seeds is None:
        raise ValueError("Each condition needs planned_seeds or protocol.seeds[condition]")
    seeds = [_seed(seed) for seed in seeds]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Planned seeds must be distinct within each reference and condition")
    records = condition.get("episodes", [])
    observed = [_seed(record["seed"]) for record in records]
    if len(observed) != len(set(observed)) or not set(observed) <= set(seeds):
        raise ValueError("Episode seeds must be distinct members of the planned seed list")
    counts = dict.fromkeys(_COUNTS, 0)
    counts.update(planned_trials=len(seeds), executed_trials=len(records),
                  unexecuted_trials=len(seeds) - len(records))
    for record in records:
        valid = record.get("valid", True) and record.get("outcome") not in _INVALID_OUTCOMES
        counts["invalid_trials"] += not valid
        # A partial-phase demonstration must never count as a full-reference
        # success, even if a caller accidentally marks its completion flag true.
        starts_at_zero = record.get("initial", {}).get("frame") == 0
        full_eligible = bool(valid and starts_at_zero and record.get("full_reference_eligible", True))
        full_success = bool(full_eligible and record.get("full_reference_success", False)
                            and record.get("reference_completed", False)
                            and record.get("outcome") == "reference_complete")
        stable_eligible = bool(full_eligible and record.get("stable_contact_eligible", False))
        stable_success = bool(stable_eligible and full_success and record.get("stable_contact_success", False))
        counts["full_reference_eligible_trials"] += full_eligible
        counts["full_reference_invalid_trials"] += not full_eligible
        counts["full_reference_successes"] += full_success
        counts["stable_contact_eligible_trials"] += stable_eligible
        counts["stable_contact_invalid_trials"] += not stable_eligible
        counts["stable_contact_successes"] += stable_success
    return _rates(counts)


def _aggregate(rows):
    """Pool trials within each chair, then give every chair equal weight."""
    counts = {key: sum(stats[key] for _, stats in rows) for key in _COUNTS}
    by_shape = defaultdict(list)
    for reference, stats in rows:
        by_shape[reference["shape_key"]].append(stats)
    chairs = {
        key: _rates({field: sum(stats[field] for stats in values) for field in _COUNTS})
        for key, values in sorted(by_shape.items())
    }
    result = dict(_rates(counts), chair_count=len(chairs), reference_count=len(rows))
    for metric in ("full_reference", "stable_contact"):
        for suffix in ("success_rate", "coverage"):
            key = f"{metric}_{suffix}"
            values = [chair[key] for chair in chairs.values() if chair[key] is not None]
            result[f"chair_macro_{key}"] = sum(values) / len(values) if values else None
    result["chairs"] = chairs
    return result


def build_report_summary(report: dict) -> dict:
    """Summarize schema-2 records without dropping invalid/unrun planned trials.

    ``protocol.seeds`` maps condition names to seed lists. An individual
    condition can override this with ``planned_seeds``. Initialize the full
    reference roster with empty episode lists before evaluation starts so
    interruptions leave the planned denominator intact.

    ``stable_contact_eligible`` means that the stable-contact definition can be
    evaluated for the reference, not that the robot survived. Failed episodes
    remain evaluated failures when the definition and simulator data are valid.
    """
    if report.get("schema_version", report.get("schema")) != 2:
        raise ValueError("Expected evaluation report schema_version=2")
    seed_map = report.get("protocol", {}).get("seeds", {})
    references, rows = [], []
    for reference in report.get("references", []):
        conditions = reference.get("conditions", {})
        names = list(dict.fromkeys([*seed_map, *conditions]))
        item = {key: reference.get(key) for key in (
            "reference", "sha256", "shape_key", "geometry_partition",
            "reference_partition", "visited_in_checkpoint")}
        if not item["shape_key"]:
            raise ValueError("Every evaluation reference needs a shape_key for chair-balanced summaries")
        item["conditions"] = {}
        for name in names:
            condition = conditions.get(name, {})
            stats = _trial_counts(condition, seed_map.get(name))
            item["conditions"][name] = stats
            rows.append((item, name, stats))
        references.append(item)

    names = list(dict.fromkeys(name for _, name, _ in rows))
    task = {"task": report.get("task", "g1_sitting"), "conditions": {
        name: _aggregate([(ref, stats) for ref, condition, stats in rows if condition == name])
        for name in names
    }}
    for partition_field in ("geometry_partition", "reference_partition"):
        groups = {}
        for partition in ("train", "test", "external"):
            groups[partition] = {
                name: _aggregate([(ref, stats) for ref, condition, stats in rows
                                  if condition == name and (ref.get(partition_field) or "external") == partition])
                for name in names
            }
        task[partition_field + "s"] = groups
    return {
        "denominator": "planned_trials",
        "aggregation": "Pool trials within each shape, then average shapes equally; no frame-level confidence intervals.",
        "references": references,
        "task": task,
    }


def atomic_write_report(path, report: dict) -> None:
    """Write one complete JSON snapshot, preserving the previous one on error."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
