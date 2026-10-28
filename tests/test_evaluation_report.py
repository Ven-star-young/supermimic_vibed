"""Evaluation units, planned denominators and reproducibility are contracts."""
import json

import pytest

from supermimic.rl.evaluation_report import (
    atomic_write_report, build_report_summary, planned_seeds,
)


def episode(seed, success=True, stable=True, **overrides):
    return dict(seed=seed, initial={"frame": 0}, valid=True,
                outcome="reference_complete" if success else "tracking_failure",
                reference_completed=success, full_reference_success=success,
                stable_contact_success=success and stable,
                stable_contact_eligible=True) | overrides


def reference(name, shape, records, **overrides):
    return dict(reference=name, sha256=name, shape_key=shape,
                geometry_partition="train", reference_partition="train",
                conditions={"clean": {"episodes": records}}) | overrides


def report(references, seeds=(0,)):
    return dict(schema_version=2, task="g1_sitting", protocol={"seeds": {"clean": list(seeds)}},
                references=references)


def test_seed_protocol_is_order_independent_and_rejects_fake_replicates():
    assert planned_seeds(2**32 - 1, 3) == [2**32 - 1, 0, 1]
    assert planned_seeds(0, 2, [13, 51]) == [13, 51]
    for base, count, explicit in ((-1, 1, None), (True, 1, None), (0, 0, None),
                                  (0, 2, [1]), (0, 2, [1, 1]), (0, 1, [2**32])):
        with pytest.raises(ValueError):
            planned_seeds(base, count, explicit)


def test_missing_trials_invalid_contacts_and_errors_do_not_inflate_success():
    data = report([reference("chair", "A", [episode(0),
        episode(1, stable_contact_eligible=False),
        episode(2, success=False, valid=False, outcome="physics_overflow")])], seeds=range(4))
    summary = build_report_summary(data)["references"][0]["conditions"]["clean"]
    assert summary["planned_trials"] == 4
    assert summary["executed_trials"] == 3
    assert summary["unexecuted_trials"] == 1
    assert summary["invalid_trials"] == 1
    assert summary["full_reference_invalid_trials"] == 1
    assert summary["stable_contact_invalid_trials"] == 2
    assert summary["full_reference_success_rate"] == .5
    assert summary["full_reference_coverage"] == .5
    assert summary["stable_contact_success_rate"] == .25
    assert summary["stable_contact_coverage"] == .25


def test_failed_valid_episode_is_evaluated_not_discarded():
    stats = build_report_summary(report([reference("a", "A", [episode(0, success=False)])]))
    stats = stats["task"]["conditions"]["clean"]
    assert stats["full_reference_coverage"] == stats["stable_contact_coverage"] == 1
    assert stats["full_reference_success_rate"] == stats["stable_contact_success_rate"] == 0


@pytest.mark.parametrize("outcome", ["viewer_closed", "manual_reset", "interrupted"])
def test_human_interruptions_remain_planned_but_are_not_evaluated_policy_failures(outcome):
    data = report([reference("a", "A", [episode(0, success=False, outcome=outcome)])])
    stats = build_report_summary(data)["task"]["conditions"]["clean"]
    assert stats["planned_trials"] == stats["executed_trials"] == stats["invalid_trials"] == 1
    assert stats["full_reference_coverage"] == stats["stable_contact_coverage"] == 0
    assert stats["full_reference_success_rate"] == stats["stable_contact_success_rate"] == 0


@pytest.mark.parametrize("override", [
    {"initial": {"frame": 20}}, {"outcome": "step_limit"},
    {"outcome": "timeout"}, {"outcome": "manual_interruption"},
    {"reference_completed": False},
])
def test_partial_or_interrupted_reference_cannot_count_as_success(override):
    stats = build_report_summary(report([reference("a", "A", [episode(0, **override)])]))
    stats = stats["task"]["conditions"]["clean"]
    assert stats["full_reference_successes"] == stats["stable_contact_successes"] == 0


def test_macro_is_per_chair_and_partitions_remain_separate():
    data = report([reference("a1", "A", [episode(0)]),
                   reference("a2", "A", [episode(0)]),
                   reference("b", "B", [episode(0, success=False)],
                             geometry_partition="test", reference_partition="external")])
    result = build_report_summary(data)["task"]
    assert result["conditions"]["clean"]["full_reference_success_rate"] == pytest.approx(2 / 3)
    assert result["conditions"]["clean"]["chair_macro_full_reference_success_rate"] == .5
    assert result["conditions"]["clean"]["chair_count"] == 2
    assert result["geometry_partitions"]["train"]["clean"]["full_reference_success_rate"] == 1
    assert result["geometry_partitions"]["test"]["clean"]["full_reference_success_rate"] == 0
    assert result["reference_partitions"]["test"]["clean"]["planned_trials"] == 0
    assert result["reference_partitions"]["test"]["clean"]["full_reference_success_rate"] is None
    assert result["reference_partitions"]["external"]["clean"]["planned_trials"] == 1


def test_unstarted_conditions_and_references_preserve_the_plan():
    data = report([reference("a", "A", [episode(0)]), reference("b", "B", [])])
    data["protocol"]["seeds"]["perturbed"] = [7, 8]
    summary = build_report_summary(data)
    assert summary["task"]["conditions"]["clean"]["full_reference_success_rate"] == .5
    perturb = summary["task"]["conditions"]["perturbed"]
    assert perturb["planned_trials"] == 4 and perturb["executed_trials"] == 0
    assert perturb["stable_contact_success_rate"] == perturb["stable_contact_coverage"] == 0


@pytest.mark.parametrize("records", [[episode(0), episode(0)], [episode(9)]])
def test_duplicate_or_unplanned_trials_are_rejected(records):
    with pytest.raises(ValueError, match="Episode seeds"):
        build_report_summary(report([reference("a", "A", records)]))


def test_atomic_json_report_refuses_nonfinite_data_without_losing_previous_result(tmp_path):
    path = tmp_path / "report.json"
    atomic_write_report(path, {"task": "坐椅子", "count": 1})
    with pytest.raises(ValueError):
        atomic_write_report(path, {"reward": float("nan")})
    assert json.loads(path.read_text()) == {"task": "坐椅子", "count": 1}
    assert len(list(tmp_path.iterdir())) == 1
