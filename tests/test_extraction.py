from __future__ import annotations

from copilotcode_sdk.extraction import should_extract


def test_should_extract_false_below_tool_call_threshold() -> None:
    assert should_extract(tool_call_count=10, total_chars=0, last_extraction_turn=0, current_turn=15) is False


def test_should_extract_true_at_tool_call_threshold() -> None:
    assert should_extract(tool_call_count=20, total_chars=0, last_extraction_turn=0, current_turn=15) is True


def test_should_extract_true_at_char_threshold() -> None:
    assert should_extract(tool_call_count=5, total_chars=50_000, last_extraction_turn=0, current_turn=15) is True


def test_should_extract_false_if_too_recent() -> None:
    assert should_extract(tool_call_count=20, total_chars=60_000, last_extraction_turn=10, current_turn=15) is False


def test_should_extract_true_with_custom_thresholds() -> None:
    assert should_extract(
        tool_call_count=5, total_chars=0, last_extraction_turn=0, current_turn=15,
        tool_call_interval=5, char_threshold=100, min_turn_gap=3,
    ) is True


def test_should_extract_respects_custom_min_turn_gap() -> None:
    assert should_extract(
        tool_call_count=20, total_chars=0, last_extraction_turn=12, current_turn=14, min_turn_gap=5,
    ) is False


from copilotcode_sdk.extraction import build_extraction_prompt


def test_build_extraction_prompt_contains_required_sections() -> None:
    prompt = build_extraction_prompt(memory_dir="/tmp/mem", project_root="/tmp/project")
    assert "durable" in prompt.lower()
    assert "/tmp/mem" in prompt
    assert "user" in prompt
    assert "feedback" in prompt
    assert "project" in prompt
    assert "reference" in prompt


def test_build_extraction_prompt_includes_project_root() -> None:
    prompt = build_extraction_prompt(memory_dir="/data/mem", project_root="/data/repo")
    assert "/data/repo" in prompt
