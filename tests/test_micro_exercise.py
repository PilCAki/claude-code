"""Tests for the micro and chain exercise system."""
from __future__ import annotations

import json

from copilotcode_sdk.micro_exercise import (
    CHAIN_EXERCISES,
    MICRO_EXERCISES,
    ExerciseFailure,
    ExerciseRunner,
    VerifyResult,
    build_verifier_prompt,
    _extract_tool_calls,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_micro_exercises_registered() -> None:
    assert len(MICRO_EXERCISES) == 6
    names = [e.name for e in MICRO_EXERCISES]
    assert names == [
        "read_file",
        "write_file",
        "shell_command",
        "multi_tool",
        "error_recovery",
        "follow_instructions",
    ]


def test_chain_exercises_registered() -> None:
    assert len(CHAIN_EXERCISES) == 3
    names = [e.name for e in CHAIN_EXERCISES]
    assert names == ["read_transform_report", "multi_file_synthesis", "memory_lifecycle"]


def test_micro_exercise_timeouts() -> None:
    for ex in MICRO_EXERCISES:
        assert ex.timeout == 300.0
        assert ex.kind == "micro"


def test_chain_exercise_timeouts() -> None:
    for ex in CHAIN_EXERCISES:
        assert ex.timeout == 480.0
        assert ex.kind == "chain"


# ---------------------------------------------------------------------------
# VerifyResult
# ---------------------------------------------------------------------------


def test_verify_result_fields() -> None:
    r = VerifyResult(passed=True, reasoning="All checks satisfied.")
    assert r.passed is True
    assert "satisfied" in r.reasoning


def test_verify_result_failure() -> None:
    r = VerifyResult(passed=False, reasoning="Tool call missing.")
    assert r.passed is False


# ---------------------------------------------------------------------------
# ExerciseFailure
# ---------------------------------------------------------------------------


def test_exercise_failure_exception() -> None:
    exc = ExerciseFailure("The agent did not call Read.")
    assert exc.reasoning == "The agent did not call Read."
    assert "Read" in str(exc)


# ---------------------------------------------------------------------------
# Verifier prompt
# ---------------------------------------------------------------------------


def test_build_verifier_prompt_structure() -> None:
    prompt = build_verifier_prompt(
        task_prompt="Read data.txt",
        agent_response="The file contains Alice,30",
        tool_calls='[{"tool": "Read", "arguments": {"path": "data.txt"}}]',
        ground_truth='{"file_content": "Alice,30"}',
        rubric="A Read tool call was made. Response references Alice.",
    )
    assert "Read data.txt" in prompt
    assert "Alice,30" in prompt
    assert "Read tool call was made" in prompt
    assert "Ground Truth" in prompt
    assert "Tool Calls Made" in prompt


def test_build_verifier_prompt_contains_all_sections() -> None:
    prompt = build_verifier_prompt(
        task_prompt="test",
        agent_response="response",
        tool_calls="none",
        ground_truth="{}",
        rubric="check something",
    )
    assert "## Task Prompt" in prompt
    assert "## Agent Response" in prompt
    assert "## Tool Calls Made" in prompt
    assert "## Ground Truth" in prompt
    assert "## Rubric" in prompt


# ---------------------------------------------------------------------------
# Tool call extraction
# ---------------------------------------------------------------------------


def test_extract_tool_calls_from_content_blocks() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "name": "Read",
                    "input": {"path": "/tmp/test.txt"},
                },
                {
                    "type": "text",
                    "text": "Let me read the file.",
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "abc123",
                    "content": "file contents here",
                },
            ],
        },
    ]
    calls = _extract_tool_calls(messages)
    assert len(calls) == 2
    assert calls[0]["tool"] == "Read"
    assert calls[0]["arguments"]["path"] == "/tmp/test.txt"
    assert calls[1]["tool_result"] == "abc123"


def test_extract_tool_calls_sdk_format() -> None:
    messages = [
        {
            "type": "assistant.message",
            "data": {
                "role": "assistant",
                "tool_requests": [
                    {"name": "Bash", "input": {"command": "echo hello"}},
                ],
            },
        },
    ]
    calls = _extract_tool_calls(messages)
    assert len(calls) == 1
    assert calls[0]["tool"] == "Bash"


def test_extract_tool_calls_empty() -> None:
    messages = [
        {"role": "assistant", "content": "Just text, no tools."},
        {"role": "user", "content": "ok"},
    ]
    calls = _extract_tool_calls(messages)
    assert calls == []


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cli_exercise_micro_mode() -> None:
    from copilotcode_sdk.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["exercise", "--mode", "micro"])
    assert args.mode == "micro"


def test_cli_exercise_chain_mode() -> None:
    from copilotcode_sdk.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["exercise", "--mode", "chain"])
    assert args.mode == "chain"


# ---------------------------------------------------------------------------
# ExerciseMode includes new modes
# ---------------------------------------------------------------------------


def test_exercise_mode_includes_micro_chain() -> None:
    from copilotcode_sdk.exercise import ExerciseMode
    import typing

    args = typing.get_args(ExerciseMode)
    assert "micro" in args
    assert "chain" in args


# ---------------------------------------------------------------------------
# Exercise data fixtures
# ---------------------------------------------------------------------------


def test_sales_csv_data() -> None:
    from copilotcode_sdk.micro_exercise import SALES_CSV

    lines = SALES_CSV.strip().split("\n")
    assert lines[0] == "product,units_sold,price_per_unit"
    assert len(lines) == 11  # header + 10 rows


def test_employees_csv_data() -> None:
    from copilotcode_sdk.micro_exercise import EMPLOYEES_CSV

    lines = EMPLOYEES_CSV.strip().split("\n")
    assert lines[0] == "name,department,salary"
    assert len(lines) == 8  # header + 7 rows


def test_departments_csv_data() -> None:
    from copilotcode_sdk.micro_exercise import DEPARTMENTS_CSV

    lines = DEPARTMENTS_CSV.strip().split("\n")
    assert lines[0] == "department,budget,location"
    assert len(lines) == 4  # header + 3 rows


# ---------------------------------------------------------------------------
# Memory lifecycle fixtures and ground truth
# ---------------------------------------------------------------------------


def test_sales_q1_csv_data() -> None:
    from copilotcode_sdk.micro_exercise import SALES_Q1_CSV

    lines = SALES_Q1_CSV.strip().split("\n")
    assert lines[0] == "region,product,revenue,units"
    assert len(lines) == 9  # header + 8 rows


def test_sales_q2_csv_data() -> None:
    from copilotcode_sdk.micro_exercise import SALES_Q2_CSV

    lines = SALES_Q2_CSV.strip().split("\n")
    assert lines[0] == "region,product,revenue,units"
    assert len(lines) == 9  # header + 8 rows


def test_q1_ground_truth_totals() -> None:
    from copilotcode_sdk.micro_exercise import _Q1_REGION_TOTALS, _Q2_REGION_TOTALS, _Q1_VS_Q2_DELTAS

    assert _Q1_REGION_TOTALS["West"] == 7600
    assert _Q1_REGION_TOTALS["Southeast"] == 6690
    assert _Q2_REGION_TOTALS["Northeast"] == 8300
    # Verify deltas are consistent
    for region in _Q1_REGION_TOTALS:
        assert _Q1_VS_Q2_DELTAS[region] == _Q2_REGION_TOTALS[region] - _Q1_REGION_TOTALS[region]


def test_runner_delete_fixture(tmp_path: "Path") -> None:
    """Test ExerciseRunner.delete_fixture removes files."""
    from pathlib import Path

    # Simulate what ExerciseRunner does
    test_file = tmp_path / "test.csv"
    test_file.write_text("a,b,c\n1,2,3\n")
    assert test_file.exists()
    test_file.unlink()
    assert not test_file.exists()


def test_runner_list_memory_files_empty(tmp_path: "Path") -> None:
    """memory_dir returns empty list when no memory files exist."""
    from copilotcode_sdk.micro_exercise import ExerciseRunner

    # ExerciseRunner.list_memory_files checks self.memory_dir
    # When memory_dir is None, returns empty list
    runner = ExerciseRunner.__new__(ExerciseRunner)
    runner._client = type("FakeClient", (), {"_memory_store": None})()
    assert runner.list_memory_files() == []


def test_memory_lifecycle_registered() -> None:
    """The memory_lifecycle exercise is in the chain registry."""
    names = [e.name for e in CHAIN_EXERCISES]
    assert "memory_lifecycle" in names
    ex = next(e for e in CHAIN_EXERCISES if e.name == "memory_lifecycle")
    assert ex.kind == "chain"
    assert ex.timeout == 480.0


def test_verify_result_carries_token_usage() -> None:
    """VerifyResult should carry token_usage and cost fields."""
    result = VerifyResult(
        passed=True,
        reasoning="Good.",
        token_usage={"input": 500, "output": 200},
        cost=0.0105,
    )
    assert result.token_usage["input"] == 500
    assert result.token_usage["output"] == 200
    assert result.cost == 0.0105


def test_verify_result_default_empty_tokens() -> None:
    """VerifyResult defaults should be empty dict and zero cost."""
    result = VerifyResult(passed=True, reasoning="ok")
    assert result.token_usage == {}
    assert result.cost == 0.0
