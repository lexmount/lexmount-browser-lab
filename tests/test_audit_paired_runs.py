from __future__ import annotations

import pytest

from lexbrowser_eval.lexbench.audit_paired_runs import (
    answer_similarity,
    error_signature,
    evidence_bucket,
    load_dataset,
    load_evaluations,
    load_json_object,
    load_json_records,
    load_results,
    paired_task_ids,
    raw_log_indicators,
)


def test_answer_similarity_normalizes_whitespace() -> None:
    assert answer_similarity("same answer", "same\nanswer") == 1.0


def test_raw_log_indicators_detect_browser_access_failures() -> None:
    result = {
        "error": None,
        "action_history": [
            "Navigation failed: net::ERR_HTTP_RESPONSE_CODE_FAILURE (HTTP ERROR 403)"
        ],
    }

    assert raw_log_indicators(result) == [
        "http_access_denied",
        "network_navigation",
    ]


def test_raw_log_indicators_ignore_status_digits_inside_urls() -> None:
    result = {"error": None, "action_history": ["Opened https://example.test/id/abc403def"]}

    assert raw_log_indicators(result) == []


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "Agent stopped before completion after 6 steps without reporting done",
            "agent_stopped_without_done",
        ),
        (
            "Navigation failed: Expected at least one handler to return a non-None result",
            "browser_state_handler_failure",
        ),
        ("Navigation failed: target closed", "navigation_failure"),
        ("Timeout after 600 seconds", "timeout"),
        ("Session create failed", "session_lifecycle_failure"),
    ],
)
def test_error_signature_redacts_runtime_details(message: str, expected: str) -> None:
    assert error_signature({"error": message}) == expected


def test_evidence_bucket_keeps_mixed_causes_explicit() -> None:
    arm = {
        "failure_category": "E1",
        "failure_codes": ["E1", "M1"],
        "signals": {"network_navigation": True},
        "raw_log_indicators": ["captcha_or_bot"],
        "agent_done": "done",
    }

    assert evidence_bucket(arm) == "mixed"


def test_load_json_records_reports_malformed_line(tmp_path) -> None:
    path = tmp_path / "records.json"
    path.write_text('{"task_id": 1}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"records\.json:2: invalid JSON record"):
        load_json_records(path)


def test_load_json_object_reports_source_path(tmp_path) -> None:
    path = tmp_path / "summary.json"
    path.write_text("not-json", encoding="utf-8")

    with pytest.raises(ValueError, match=r"summary\.json: invalid JSON object"):
        load_json_object(path)


def test_load_json_object_requires_object(tmp_path) -> None:
    path = tmp_path / "summary.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match=r"summary\.json: expected a JSON object"):
        load_json_object(path)


def test_load_json_records_rejects_non_object_record(tmp_path) -> None:
    path = tmp_path / "records.json"
    path.write_text('[{"id": 1}, 2]', encoding="utf-8")

    with pytest.raises(ValueError, match=r"record 2 is not a JSON object"):
        load_json_records(path)


def test_load_results_reports_malformed_result_path(tmp_path) -> None:
    result_path = tmp_path / "tasks" / "42" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text("not-json", encoding="utf-8")

    with pytest.raises(ValueError, match=r"tasks/42/result\.json: invalid JSON object"):
        load_results(tmp_path)


def test_load_dataset_requires_id(tmp_path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text('[{"task": "missing id"}]', encoding="utf-8")

    with pytest.raises(ValueError, match=r"record 1 is missing 'id'"):
        load_dataset(path)


def test_load_evaluations_requires_task_id(tmp_path) -> None:
    path = tmp_path / "evaluations.json"
    path.write_text('[{"success": true}]', encoding="utf-8")

    with pytest.raises(ValueError, match=r"record 1 is missing 'task_id'"):
        load_evaluations(path)


def test_load_evaluations_rejects_duplicate_task_ids(tmp_path) -> None:
    path = tmp_path / "evaluations.json"
    path.write_text('[{"task_id": 1}, {"task_id": "1"}]', encoding="utf-8")

    with pytest.raises(ValueError, match=r"duplicate 'task_id' value 1"):
        load_evaluations(path)


def test_paired_task_ids_reports_incomplete_coverage() -> None:
    summary = {"per_task": {"1": {"success": False}}}

    with pytest.raises(ValueError, match="local evaluations: 1"):
        paired_task_ids(
            {"1": {}}, summary, summary, {"1": {}}, {}, {"1": {}}, {"1": {}}
        )


def test_paired_task_ids_rejects_summary_coverage_mismatch() -> None:
    lexmount_summary = {
        "per_task": {"1": {"success": False}, "2": {"success": False}}
    }
    local_summary = {"per_task": {"1": {"success": False}}}
    complete = {"1": {}, "2": {}}

    with pytest.raises(ValueError, match=r"lexmount only: 2; local only: none"):
        paired_task_ids(
            complete,
            lexmount_summary,
            local_summary,
            complete,
            complete,
            complete,
            complete,
        )


def test_paired_task_ids_reports_missing_dataset_row() -> None:
    summary = {"per_task": {"1": {"success": False}}}
    complete = {"1": {}}

    with pytest.raises(ValueError, match="dataset: 1"):
        paired_task_ids({}, summary, summary, complete, complete, complete, complete)


def test_paired_task_ids_can_return_complete_subset_when_requested() -> None:
    summary = {"per_task": {"1": {"success": False}, "2": {"success": False}}}
    complete = {"1": {}, "2": {}}

    assert paired_task_ids(
        complete,
        summary,
        summary,
        {"1": {}},
        complete,
        complete,
        complete,
        allow_incomplete=True,
    ) == ["1"]
