from __future__ import annotations

import pytest

from lexbrowser_eval.lexbench.analyze_mechanism_repeats import analyze_mechanism_repeats


def summary(records: dict[str, int]) -> dict:
    return {"per_task": {task_id: {"success": bool(value)} for task_id, value in records.items()}}


def selection() -> dict:
    return {
        "selection": {
            "stable_lexmount_only": [
                {"task_id": "1", "pair_outcomes": "XX", "language": "en"}
            ],
            "stable_local_only": [
                {"task_id": "2", "pair_outcomes": "LL", "language": "zh"}
            ],
        }
    }


def test_analyze_mechanism_repeats_tracks_followup_retention() -> None:
    lexmount = [
        summary({"1": 1, "2": 0, "99": 1}),
        summary({"1": 1, "2": 0}),
        summary({"1": 1, "2": 1}),
        summary({"1": 0, "2": 0}),
    ]
    local = [
        summary({"1": 0, "2": 1}),
        summary({"1": 0, "2": 1}),
        summary({"1": 0, "2": 1}),
        summary({"1": 1, "2": 1}),
    ]

    audits = [
        {
            "discordant": [
                {
                    "task_id": "1",
                    "outcome": "lexmount_only",
                    "target_website": "example.com",
                    "evidence_bucket": "site_or_access_environment",
                    "local": {
                        "failure_category": "E1",
                        "error_signature": "navigation_failure",
                        "raw_log_indicators": ["network_navigation"],
                    },
                },
                {
                    "task_id": "99",
                    "outcome": "local_only",
                    "evidence_bucket": "unresolved",
                    "lexmount": {"failure_category": None},
                },
            ]
        }
    ] * 4
    result = analyze_mechanism_repeats(
        selection(), lexmount, local, labels=["a", "b", "c", "d"], audits=audits
    )

    assert result["success_attempts"] == {"lexmount": 4, "local": 5}
    assert result["per_task"][0]["pattern"] == "XXXL"
    assert result["per_task"][0]["target_website"] == "example.com"
    assert result["category_summary"]["stable_lexmount_only"] == {
        "tasks": 1,
        "followup_observations": 2,
        "followup_outcomes": {"X": 1, "L": 1},
        "unanimous_all_repeats": 0,
        "expected_outcome": "X",
        "followup_expected_observations": 1,
        "followup_expected_rate": 0.5,
        "tasks_expected_in_all_repeats": 0,
        "tasks_expected_in_all_followups": 0,
    }
    assert result["repeatability"]["contains_both_lexmount_only_and_local_only"] == 1
    assert result["evidence"]["discordant_observations"] == 4
    assert result["evidence"]["loser_error_signatures"]["lexmount_only"] == {
        "navigation_failure": 4
    }
    assert result["evidence"]["loser_raw_log_indicators"]["lexmount_only"] == {
        "network_navigation": 4
    }


def test_analyze_mechanism_repeats_rejects_selection_mismatch() -> None:
    runs = [summary({"1": 1, "2": 0})] * 3
    with pytest.raises(ValueError, match="does not match selected pattern"):
        analyze_mechanism_repeats(
            selection(), runs, runs, labels=["a", "b", "c"]
        )


def test_analyze_mechanism_repeats_rejects_empty_selection_groups() -> None:
    empty_selection = {"selection": {"stable_lexmount_only": []}}

    with pytest.raises(ValueError, match="selection contains no task records"):
        analyze_mechanism_repeats(
            empty_selection,
            [summary({"1": 1})] * 3,
            [summary({"1": 0})] * 3,
            labels=["a", "b", "c"],
        )
