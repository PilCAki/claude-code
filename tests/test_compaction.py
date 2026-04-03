from __future__ import annotations

from copilotcode_sdk.compaction import build_compaction_prompt, build_handoff_context


def test_build_compaction_prompt_contains_six_summary_points() -> None:
    prompt = build_compaction_prompt()
    assert "primary request" in prompt.lower() or "original" in prompt.lower()
    assert "key findings" in prompt.lower() or "discoveries" in prompt.lower()
    assert "current state" in prompt.lower() or "progress" in prompt.lower()
    assert "remaining" in prompt.lower() or "next" in prompt.lower()


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
