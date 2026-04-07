from __future__ import annotations

import json

from copilotcode_sdk.exercise import (
    ExerciseReport,
    SubsystemResult,
    SUBSYSTEM_CHECKLIST,
    build_exercise_prompt,
    parse_exercise_report,
)


# ---------------------------------------------------------------------------
# SubsystemResult
# ---------------------------------------------------------------------------


def test_subsystem_result_fields() -> None:
    r = SubsystemResult(
        name="events",
        status="pass",
        detail="EventBus created, event emitted and received.",
        duration_seconds=0.12,
    )
    assert r.name == "events"
    assert r.status == "pass"
    assert r.error is None
    assert r.duration_seconds == 0.12


def test_subsystem_result_with_error() -> None:
    r = SubsystemResult(
        name="memory",
        status="error",
        detail="Import failed.",
        error="ModuleNotFoundError: no module named 'copilotcode_sdk.memory'",
    )
    assert r.status == "error"
    assert "ModuleNotFoundError" in (r.error or "")


# ---------------------------------------------------------------------------
# ExerciseReport
# ---------------------------------------------------------------------------


def test_exercise_report_counts() -> None:
    report = ExerciseReport(
        product_name="CopilotCode",
        session_id="test-123",
        timestamp="2026-04-05T00:00:00Z",
        subsystems=[
            SubsystemResult(name="a", status="pass", detail="ok"),
            SubsystemResult(name="b", status="pass", detail="ok"),
            SubsystemResult(name="c", status="fail", detail="nope"),
            SubsystemResult(name="d", status="skip", detail="env"),
            SubsystemResult(name="e", status="error", detail="boom", error="err"),
        ],
    )
    assert report.passed == 2
    assert report.failed == 1
    assert report.skipped == 1
    assert report.errored == 1
    assert report.ok is False


def test_exercise_report_ok_when_all_pass() -> None:
    report = ExerciseReport(
        product_name="CopilotCode",
        session_id="s",
        timestamp="t",
        subsystems=[
            SubsystemResult(name="a", status="pass", detail="ok"),
            SubsystemResult(name="b", status="skip", detail="env"),
        ],
    )
    assert report.ok is True


def test_exercise_report_to_dict() -> None:
    report = ExerciseReport(
        product_name="CopilotCode",
        session_id="s-1",
        timestamp="2026-04-05T00:00:00Z",
        total_duration_seconds=5.5,
        summary="All good.",
        subsystems=[
            SubsystemResult(name="config", status="pass", detail="defaults verified"),
        ],
    )
    d = report.to_dict()
    assert d["product_name"] == "CopilotCode"
    assert d["ok"] is True
    assert d["passed"] == 1
    assert d["total_duration_seconds"] == 5.5
    assert len(d["subsystems"]) == 1
    assert d["subsystems"][0]["name"] == "config"
    # Round-trip through JSON
    assert json.loads(json.dumps(d)) == d


def test_exercise_report_to_text() -> None:
    report = ExerciseReport(
        product_name="CopilotCode",
        session_id="s-1",
        timestamp="2026-04-05T00:00:00Z",
        summary="Done.",
        subsystems=[
            SubsystemResult(name="events", status="pass", detail="ok"),
            SubsystemResult(name="diff", status="fail", detail="wrong output", error="mismatch"),
        ],
    )
    text = report.to_text()
    assert "Exercise Report" in text
    assert "[+] events" in text
    assert "[!] diff" in text
    assert "error: mismatch" in text
    assert "Summary: Done." in text


# ---------------------------------------------------------------------------
# build_exercise_prompt
# ---------------------------------------------------------------------------


def test_build_exercise_prompt_contains_all_subsystems() -> None:
    prompt = build_exercise_prompt()
    for item in SUBSYSTEM_CHECKLIST:
        assert item["name"] in prompt, f"Missing subsystem: {item['name']}"
    assert "<exercise-report>" in prompt
    assert "self-aware" in prompt


def test_build_exercise_prompt_custom_checklist() -> None:
    custom = [
        {"name": "alpha", "description": "Test alpha."},
        {"name": "beta", "description": "Test beta."},
    ]
    prompt = build_exercise_prompt(custom)
    assert "alpha" in prompt
    assert "beta" in prompt
    # Default subsystems should NOT appear
    assert "prompt_compiler" not in prompt


# ---------------------------------------------------------------------------
# parse_exercise_report
# ---------------------------------------------------------------------------


def test_parse_exercise_report_valid_json() -> None:
    raw = """Some preamble text.
<exercise-report>
{
  "subsystems": [
    {"name": "config", "status": "pass", "detail": "defaults ok", "duration_seconds": 0.1, "error": null},
    {"name": "events", "status": "fail", "detail": "timeout", "duration_seconds": 1.2, "error": "timed out"}
  ],
  "summary": "1 pass, 1 fail"
}
</exercise-report>
trailing text"""
    report = parse_exercise_report(
        raw,
        product_name="Test",
        session_id="s1",
        timestamp="t1",
        total_duration_seconds=2.0,
    )
    assert report.product_name == "Test"
    assert report.session_id == "s1"
    assert len(report.subsystems) == 2
    assert report.subsystems[0].name == "config"
    assert report.subsystems[0].status == "pass"
    assert report.subsystems[1].error == "timed out"
    assert report.summary == "1 pass, 1 fail"


def test_parse_exercise_report_no_tags() -> None:
    raw = json.dumps({
        "subsystems": [
            {"name": "diff", "status": "pass", "detail": "ok"}
        ],
        "summary": "all good",
    })
    report = parse_exercise_report(raw)
    assert len(report.subsystems) == 1
    assert report.subsystems[0].name == "diff"


def test_parse_exercise_report_malformed() -> None:
    raw = "This is not JSON at all, just random text."
    report = parse_exercise_report(raw)
    assert len(report.subsystems) == 1
    assert report.subsystems[0].status == "error"
    assert "parse_error" in report.subsystems[0].name


# ---------------------------------------------------------------------------
# CLI parser wiring
# ---------------------------------------------------------------------------


def test_cli_exercise_parser() -> None:
    from copilotcode_sdk.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["exercise", "--json", "--timeout", "300"])
    assert args.command == "exercise"
    assert args.json is True
    assert args.timeout == 300.0
    assert args.subsystems is None


def test_cli_exercise_parser_subsystems() -> None:
    from copilotcode_sdk.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["exercise", "--subsystems", "config", "events", "diff"])
    assert args.subsystems == ["config", "events", "diff"]
