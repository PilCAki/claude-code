from __future__ import annotations

import json

from copilotcode_sdk.exercise import (
    ExerciseReport,
    SubsystemResult,
    SUBSYSTEM_CHECKLIST,
    ORCHESTRATION_SCENARIOS,
    ADVANCED_ORCHESTRATION_SCENARIOS,
    CASCADE_ORCHESTRATION_SCENARIOS,
    build_exercise_prompt,
    build_orchestration_prompt,
    build_advanced_orchestration_prompt,
    build_cascade_orchestration_prompt,
    build_exercise_config,
    build_cascade_config,
    parse_exercise_report,
    _capture_ground_truth,
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


# ---------------------------------------------------------------------------
# Orchestration scenarios
# ---------------------------------------------------------------------------


def test_orchestration_scenarios_exist() -> None:
    assert len(ORCHESTRATION_SCENARIOS) == 6
    names = [s["name"] for s in ORCHESTRATION_SCENARIOS]
    assert "event_bus_lifecycle" in names
    assert "file_change_tracking" in names
    assert "git_context_staleness" in names
    assert "tool_call_accumulation" in names
    assert "multi_turn_conversation" in names
    assert "error_handling" in names


def test_build_orchestration_prompt_self_aware() -> None:
    prompt = build_orchestration_prompt()
    assert "self-aware" in prompt
    assert "PROVOKE" in prompt
    assert "OBSERVE" in prompt
    assert "exercise-report" in prompt
    for scenario in ORCHESTRATION_SCENARIOS:
        assert scenario["name"] in prompt


def test_exercise_report_ground_truth() -> None:
    report = ExerciseReport(
        product_name="Test",
        session_id="s1",
        timestamp="t1",
        mode="orchestration",
        ground_truth={"event_counts": {"tool_called": 5}, "total_events": 8},
    )
    d = report.to_dict()
    assert d["mode"] == "orchestration"
    assert d["ground_truth"]["total_events"] == 8
    assert d["ground_truth"]["event_counts"]["tool_called"] == 5


def test_exercise_report_no_ground_truth() -> None:
    report = ExerciseReport(
        product_name="Test",
        session_id="s1",
        timestamp="t1",
        mode="subsystem",
    )
    d = report.to_dict()
    assert "ground_truth" not in d


def test_cli_exercise_mode_flag() -> None:
    from copilotcode_sdk.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["exercise", "--mode", "orchestration"])
    assert args.mode == "orchestration"

    args2 = parser.parse_args(["exercise"])
    assert args2.mode == "full"


# ---------------------------------------------------------------------------
# Advanced orchestration scenarios
# ---------------------------------------------------------------------------


def test_advanced_orchestration_scenarios_exist() -> None:
    assert len(ADVANCED_ORCHESTRATION_SCENARIOS) == 7
    names = [s["name"] for s in ADVANCED_ORCHESTRATION_SCENARIOS]
    assert "extraction_nudge_trigger" in names
    assert "git_safety_gate" in names
    assert "tool_result_caching" in names
    assert "task_reminder_injection" in names
    assert "context_size_warning" in names
    assert "read_before_write_enforcement" in names
    assert "repeat_loop_detection" in names


def test_build_advanced_orchestration_prompt_self_aware() -> None:
    prompt = build_advanced_orchestration_prompt()
    assert "self-aware" in prompt
    assert "PROVOKE" in prompt
    assert "OBSERVE" in prompt
    assert "lowered" in prompt.lower()
    assert "exercise-report" in prompt
    for scenario in ADVANCED_ORCHESTRATION_SCENARIOS:
        assert scenario["name"] in prompt


def test_build_exercise_config_lowered_thresholds() -> None:
    cfg = build_exercise_config()
    assert cfg.extraction_tool_call_interval == 5
    assert cfg.extraction_char_threshold == 10_000
    assert cfg.extraction_min_turn_gap == 2
    assert cfg.task_reminder_turns == 3
    assert cfg.task_reminder_cooldown_turns == 3
    assert cfg.enable_tool_result_cache is True
    assert cfg.reminder_reinjection_interval == 5
    assert cfg.max_context_chars == 50_000


def test_build_exercise_config_preserves_base() -> None:
    from copilotcode_sdk.config import CopilotCodeConfig

    base = CopilotCodeConfig(model="claude-sonnet-4-20250514")
    cfg = build_exercise_config(base)
    assert cfg.model == "claude-sonnet-4-20250514"
    assert cfg.extraction_tool_call_interval == 5  # overridden


def test_capture_ground_truth_with_raw_hooks() -> None:
    from copilotcode_sdk.events import EventBus

    class FakeSession:
        event_bus = EventBus()
        _state = None
        _completed_skills = None
        _raw_hooks = {
            "get_tool_call_count": lambda: 7,
            "get_recent_shell": lambda: [("echo hello", "hello")],
            "get_read_file_state": lambda: {"/tmp/foo.txt": 1234.0},
            "get_estimated_context_chars": lambda: 45000,
            "get_tool_result_cache_size": lambda: 3,
            "get_token_budget": lambda: None,
            "get_file_changes": lambda: {},
        }

    gt = _capture_ground_truth(FakeSession())
    assert gt["tool_call_count"] == 7
    assert len(gt["recent_shell"]) == 1
    assert "/tmp/foo.txt" in gt["read_file_state_keys"]
    assert gt["estimated_context_chars"] == 45000
    assert gt["tool_result_cache_size"] == 3


def test_cli_exercise_advanced_mode() -> None:
    from copilotcode_sdk.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["exercise", "--mode", "advanced"])
    assert args.mode == "advanced"


def test_basic_scenarios_unchanged() -> None:
    assert len(ORCHESTRATION_SCENARIOS) == 6
    names = [s["name"] for s in ORCHESTRATION_SCENARIOS]
    assert names == [
        "event_bus_lifecycle",
        "file_change_tracking",
        "git_context_staleness",
        "tool_call_accumulation",
        "multi_turn_conversation",
        "error_handling",
    ]


def test_skill_shorthand_expansion() -> None:
    from copilotcode_sdk.hooks import _expand_skill_shorthand

    # Known skill should expand
    result = _expand_skill_shorthand("/verify")
    assert "Use the `verify` skill" in result

    # Known skill with arguments should expand and include args
    result = _expand_skill_shorthand("/verify check the code")
    assert "Use the `verify` skill" in result
    assert "check the code" in result

    # Unknown skill should not expand
    result = _expand_skill_shorthand("/unknown_skill_xyz")
    assert result == "/unknown_skill_xyz"

    # Non-shorthand prompt should pass through
    result = _expand_skill_shorthand("just a normal prompt")
    assert result == "just a normal prompt"


# ---------------------------------------------------------------------------
# Cascade orchestration scenarios
# ---------------------------------------------------------------------------


def test_capture_ground_truth_compaction_warned() -> None:
    from copilotcode_sdk.events import EventBus

    class FakeSession:
        event_bus = EventBus()
        _state = None
        _completed_skills = None
        _raw_hooks = {
            "get_tool_call_count": lambda: 0,
            "get_recent_shell": lambda: [],
            "get_read_file_state": lambda: {},
            "get_estimated_context_chars": lambda: 0,
            "get_tool_result_cache_size": lambda: 0,
            "get_compaction_warned": lambda: True,
            "get_token_budget": lambda: None,
            "get_file_changes": lambda: {},
        }

    gt = _capture_ground_truth(FakeSession())
    assert gt["compaction_warned"] is True


def test_cascade_orchestration_scenarios_exist() -> None:
    assert len(CASCADE_ORCHESTRATION_SCENARIOS) == 7
    names = [s["name"] for s in CASCADE_ORCHESTRATION_SCENARIOS]
    assert "context_accounting_desync" in names
    assert "injection_priority_collision" in names
    assert "cache_invalidation_cascade" in names
    assert "error_skip_vs_session_continuity" in names
    assert "task_reminder_vs_extraction_race" in names
    assert "repeat_detection_ring_overflow" in names
    assert "context_warning_single_fire" in names


def test_build_cascade_config_thresholds() -> None:
    cfg = build_cascade_config()
    assert cfg.extraction_tool_call_interval == 3
    assert cfg.extraction_char_threshold == 5_000
    assert cfg.extraction_min_turn_gap == 1
    assert cfg.task_reminder_turns == 2
    assert cfg.task_reminder_cooldown_turns == 2
    assert cfg.enable_tool_result_cache is True
    assert cfg.reminder_reinjection_interval == 3
    assert cfg.max_context_chars == 20_000
    assert cfg.noisy_tool_char_limit == 5_000


def test_build_cascade_config_preserves_base() -> None:
    from copilotcode_sdk.config import CopilotCodeConfig

    base = CopilotCodeConfig(model="claude-opus-4-20250514")
    cfg = build_cascade_config(base)
    assert cfg.model == "claude-opus-4-20250514"
    assert cfg.extraction_tool_call_interval == 3  # overridden


def test_build_cascade_orchestration_prompt_self_aware() -> None:
    prompt = build_cascade_orchestration_prompt()
    assert "self-aware" in prompt
    assert "PROVOKE" in prompt
    assert "OBSERVE" in prompt
    assert "cascade" in prompt.lower()
    assert "exercise-report" in prompt
    for scenario in CASCADE_ORCHESTRATION_SCENARIOS:
        assert scenario["name"] in prompt


def test_cascade_mode_in_exercise_mode() -> None:
    from copilotcode_sdk.exercise import ExerciseMode
    import typing
    args = typing.get_args(ExerciseMode)
    assert "cascade" in args


def test_cli_exercise_cascade_mode() -> None:
    from copilotcode_sdk.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["exercise", "--mode", "cascade"])
    assert args.mode == "cascade"
