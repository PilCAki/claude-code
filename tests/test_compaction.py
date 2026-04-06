from __future__ import annotations

from copilotcode_sdk.compaction import (
    CompactionResult,
    build_compaction_prompt,
    build_handoff_context,
    format_transcript_for_compaction,
    parse_compaction_response,
)


# ---------------------------------------------------------------------------
# Existing tests (updated for 9-section structure)
# ---------------------------------------------------------------------------


def test_build_compaction_prompt_contains_nine_sections() -> None:
    prompt = build_compaction_prompt()
    assert "Primary request and intent" in prompt
    assert "Key technical concepts" in prompt
    assert "Files and code sections" in prompt
    assert "Errors and fixes" in prompt
    assert "Problem solving" in prompt
    assert "All user messages" in prompt
    assert "Pending tasks" in prompt
    assert "Current Work" in prompt
    assert "Optional Next Step" in prompt


def test_build_handoff_context_includes_all_sections() -> None:
    context = build_handoff_context(
        compaction_summary="User is analyzing Q1 data.",
        skill_catalog_text="| Skill | Desc |\n|---|---|\n| intake | Ingest |",
        instruction_content="Use DuckDB for queries.",
        memory_index="- [profile](profile.md) - user prefs",
    )
    assert "Q1 data" in context
    assert "intake" in context
    assert "DuckDB" in context
    assert "profile" in context


def test_build_handoff_context_omits_empty_sections() -> None:
    context = build_handoff_context(
        compaction_summary="Analyzing data.",
        skill_catalog_text="",
        instruction_content="",
        memory_index="",
    )
    assert "Analyzing data." in context
    assert "Skill Catalog" not in context
    assert "Workspace Instructions" not in context
    assert "Memory Index" not in context


# ---------------------------------------------------------------------------
# Wave 4.4: 9-section prompt structure
# ---------------------------------------------------------------------------


def test_full_mode_has_9_sections() -> None:
    prompt = build_compaction_prompt(mode="full")
    # Sections 1-7 common
    for i in range(1, 8):
        assert f"{i}." in prompt, f"Missing section {i}"
    # Sections 8-9 full mode
    assert "8. **Current Work**" in prompt
    assert "9. **Optional Next Step**" in prompt


def test_up_to_sections_8_9() -> None:
    prompt = build_compaction_prompt(mode="up_to", up_to_turn=10)
    assert "8. **Work Completed**" in prompt
    assert "9. **Context for Continuing Work**" in prompt
    assert "Current Work" not in prompt
    assert "Optional Next Step" not in prompt


def test_partial_from_sections_8_9() -> None:
    prompt = build_compaction_prompt(mode="partial", preserve_recent=5)
    # partial uses the same sections as full
    assert "8. **Current Work**" in prompt
    assert "9. **Optional Next Step**" in prompt


def test_no_tools_trailer_present() -> None:
    prompt = build_compaction_prompt()
    assert "REMINDER: Do NOT call any tools" in prompt
    assert "<analysis>" in prompt
    assert "<summary>" in prompt


def test_all_modes_have_trailer() -> None:
    for mode in ("full", "partial", "up_to"):
        prompt = build_compaction_prompt(mode=mode)
        assert "REMINDER: Do NOT call any tools" in prompt, (
            f"{mode} mode missing no-tools trailer"
        )


def test_extra_instructions_appended() -> None:
    prompt = build_compaction_prompt(extra_instructions="Preserve all SQL queries verbatim.")
    assert "Preserve all SQL queries verbatim." in prompt
    assert "Additional compaction guidance" in prompt


# ---------------------------------------------------------------------------
# Wave 4.4: Parser
# ---------------------------------------------------------------------------


def test_parse_with_both_tags() -> None:
    text = (
        "Some preamble\n"
        "<analysis>\nThis is the analysis.\n</analysis>\n"
        "<summary>\nThis is the summary.\n</summary>\n"
        "trailing text"
    )
    result = parse_compaction_response(text)
    assert result.analysis == "This is the analysis."
    assert result.summary == "This is the summary."
    assert result.raw_response == text


def test_parse_no_analysis() -> None:
    text = "<summary>Just summary content.</summary>"
    result = parse_compaction_response(text)
    assert result.analysis == ""
    assert result.summary == "Just summary content."


def test_parse_no_summary() -> None:
    text = "Plain text without any tags, just a free-form summary."
    result = parse_compaction_response(text)
    assert result.analysis == ""
    assert result.summary == text.strip()


def test_compaction_result_frozen() -> None:
    result = CompactionResult(analysis="a", summary="s", raw_response="r")
    try:
        result.analysis = "new"  # type: ignore[misc]
        assert False, "Should have raised"
    except AttributeError:
        pass


# ---------------------------------------------------------------------------
# Wave 4.4: Handoff context enhancements
# ---------------------------------------------------------------------------


def test_handoff_context_recent_turns_note() -> None:
    context = build_handoff_context(
        compaction_summary="Summary here.",
        recent_turns_note="3 recent turns preserved after this summary.",
    )
    assert "Recent Turns" in context
    assert "3 recent turns preserved" in context


def test_handoff_context_all_sections() -> None:
    context = build_handoff_context(
        compaction_summary="Summary.",
        skill_catalog_text="skills",
        instruction_content="instructions",
        memory_index="memories",
        recent_turns_note="recent",
    )
    assert "Compaction Summary" in context
    assert "Skill Catalog" in context
    assert "Workspace Instructions" in context
    assert "Memory Index" in context
    assert "Recent Turns" in context


def test_build_compaction_prompt_extra_instructions() -> None:
    prompt = build_compaction_prompt(extra_instructions="Keep all SQL.")
    # Extra instructions come before the trailer
    trailer_pos = prompt.index("REMINDER: Do NOT call any tools")
    extra_pos = prompt.index("Keep all SQL.")
    assert extra_pos < trailer_pos


# ---------------------------------------------------------------------------
# format_transcript_for_compaction
# ---------------------------------------------------------------------------


def test_format_transcript_basic() -> None:
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
        {"role": "user", "content": "what's up"},
    ]
    result = format_transcript_for_compaction(messages)
    assert result.startswith("<transcript>\n")
    assert result.endswith("\n</transcript>")
    assert "[user]: hello" in result
    assert "[assistant]: hi there" in result
    assert "[user]: what's up" in result


def test_format_transcript_truncates_long_messages() -> None:
    long_msg = "x" * 5000
    messages = [{"role": "user", "content": long_msg}]
    result = format_transcript_for_compaction(messages, max_per_message=3000)
    # Content should be truncated to 3000 chars + ellipsis
    line = result.split("\n")[1]  # skip <transcript>
    content_part = line[len("[user]: "):]
    assert len(content_part) == 3001  # 3000 chars + "…"
    assert content_part.endswith("…")


def test_format_transcript_respects_max_chars() -> None:
    messages = [
        {"role": "user", "content": "A" * 100},
        {"role": "assistant", "content": "B" * 100},
        {"role": "user", "content": "C" * 100},
    ]
    # With a tight max_chars, oldest messages get dropped
    result = format_transcript_for_compaction(messages, max_chars=150)
    # Should keep the most recent message(s) and drop oldest
    assert "[user]: " + "C" * 100 in result
    assert "[user]: " + "A" * 100 not in result


def test_format_transcript_empty() -> None:
    result = format_transcript_for_compaction([])
    assert result == "<transcript>\n\n</transcript>"


def test_format_transcript_content_blocks() -> None:
    """Messages with list content blocks are handled correctly."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Here is the answer."},
                {"type": "tool_use", "id": "t1"},
            ],
        },
    ]
    result = format_transcript_for_compaction(messages)
    assert "[assistant]: Here is the answer." in result


def test_format_transcript_accepts_session_event_like_objects() -> None:
    class EventLike:
        def to_dict(self) -> dict[str, object]:
            return {"role": "assistant", "content": "live event text"}

    result = format_transcript_for_compaction([EventLike()])

    assert "[assistant]: live event text" in result
