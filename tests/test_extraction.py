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
