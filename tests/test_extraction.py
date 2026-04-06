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
    assert "checkpoint" in prompt.lower()
    assert "/tmp/mem" in prompt
    assert "/tmp/project" in prompt
    assert "MEMORY.md" in prompt
    # Should NOT contain opt-out language
    assert "that is fine" not in prompt
    assert "do not force" not in prompt
    # Should contain inline example with frontmatter
    assert "---\nname:" in prompt
    assert "type: project" in prompt


def test_build_extraction_prompt_includes_project_root() -> None:
    prompt = build_extraction_prompt(memory_dir="/data/mem", project_root="/data/repo")
    assert "/data/repo" in prompt


from copilotcode_sdk.extraction import build_session_end_extraction_prompt


def test_build_session_end_extraction_prompt_is_urgent() -> None:
    prompt = build_session_end_extraction_prompt(memory_dir="/tmp/mem", project_root="/tmp/project")
    assert "/tmp/mem" in prompt
    assert "/tmp/project" in prompt
    assert "complete" in prompt.lower()
    assert "at least one" in prompt.lower()
    # Should NOT contain opt-out language
    assert "that is fine" not in prompt


def test_build_session_end_extraction_prompt_lists_what_to_capture() -> None:
    prompt = build_session_end_extraction_prompt(memory_dir="/m", project_root="/p")
    assert "schema" in prompt.lower()
    assert "column" in prompt.lower()
    assert "data quality" in prompt.lower()
    assert "metrics" in prompt.lower()


# ---------------------------------------------------------------------------
# Wave 2: Enforce extraction mode
# ---------------------------------------------------------------------------

from copilotcode_sdk.extraction import build_enforce_extraction_prompt, ExtractionMode


def test_extraction_mode_type_accepts_both_values() -> None:
    nudge: ExtractionMode = "nudge"
    enforce: ExtractionMode = "enforce"
    assert nudge == "nudge"
    assert enforce == "enforce"


def test_build_enforce_extraction_prompt_contains_required_parts() -> None:
    prompt = build_enforce_extraction_prompt(
        memory_dir="/tmp/mem",
        session_memory_path="/tmp/mem/session_memory.md",
    )
    assert "enforcement" in prompt.lower() or "enforce" in prompt.lower()
    assert "MUST" in prompt
    assert "/tmp/mem/session_memory.md" in prompt


def test_build_enforce_extraction_prompt_includes_format_guidance() -> None:
    prompt = build_enforce_extraction_prompt(
        memory_dir="/data/mem",
        session_memory_path="/data/mem/session_memory.md",
    )
    # Should have structured entry format instructions
    assert "# <Title>" in prompt or "Title" in prompt
    assert "---" in prompt  # separator guidance
