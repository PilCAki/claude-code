from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from copilotcode_sdk.suggestions import (
    build_predictive_suggestions,
    build_prompt_suggestions,
    format_suggestions_prompt,
    merge_suggestions,
)
from copilotcode_sdk.tasks import TaskRecord, TaskStatus


def _sample_skill_map() -> dict[str, dict[str, str]]:
    return {
        "intake": {"name": "intake", "type": "data-intake", "requires": "none"},
        "analysis": {"name": "analysis", "type": "biz-analysis", "requires": "data-intake"},
    }


class TestBuildPromptSuggestions:
    def test_empty_inputs(self) -> None:
        suggestions = build_prompt_suggestions()
        assert suggestions == []

    def test_skill_with_met_prerequisites(self) -> None:
        suggestions = build_prompt_suggestions(
            skill_map=_sample_skill_map(),
            completed_skills=set(),
        )
        assert any("intake" in s for s in suggestions)

    def test_skill_after_completion(self) -> None:
        suggestions = build_prompt_suggestions(
            skill_map=_sample_skill_map(),
            completed_skills={"intake"},
        )
        assert any("analysis" in s for s in suggestions)

    def test_all_skills_complete(self) -> None:
        suggestions = build_prompt_suggestions(
            skill_map=_sample_skill_map(),
            completed_skills={"intake", "analysis"},
        )
        assert any("complete" in s.lower() for s in suggestions)

    def test_task_in_progress(self) -> None:
        task = TaskRecord(id=1, subject="Write tests", status=TaskStatus.in_progress)
        suggestions = build_prompt_suggestions(open_tasks=[task])
        assert any("Continue" in s for s in suggestions)

    def test_task_pending(self) -> None:
        task = TaskRecord(id=1, subject="Write tests", status=TaskStatus.pending)
        suggestions = build_prompt_suggestions(open_tasks=[task])
        assert any("Start" in s or "Write tests" in s for s in suggestions)

    def test_almost_done_hint(self) -> None:
        task = TaskRecord(id=5, subject="Last one", status=TaskStatus.pending)
        suggestions = build_prompt_suggestions(
            open_tasks=[task],
            completed_task_count=4,
        )
        assert any("80%" in s or "Almost" in s for s in suggestions)

    def test_long_session_memory_nudge(self) -> None:
        suggestions = build_prompt_suggestions(session_turn=35)
        assert any("memory" in s.lower() for s in suggestions)

    def test_edit_heavy_suggests_verification(self) -> None:
        suggestions = build_prompt_suggestions(
            session_turn=10,
            recent_tools=["edit", "edit", "edit", "edit"],
        )
        assert any("test" in s.lower() or "verify" in s.lower() for s in suggestions)

    def test_early_session_skill_review(self) -> None:
        suggestions = build_prompt_suggestions(
            session_turn=1,
            skill_map=_sample_skill_map(),
            completed_skills=set(),
        )
        assert any("skill catalog" in s.lower() or "review" in s.lower() for s in suggestions)

    def test_max_five_suggestions(self) -> None:
        task = TaskRecord(id=1, subject="Task", status=TaskStatus.in_progress)
        suggestions = build_prompt_suggestions(
            skill_map=_sample_skill_map(),
            completed_skills=set(),
            open_tasks=[task],
            completed_task_count=4,
            session_turn=35,
            recent_tools=["edit"] * 5,
        )
        assert len(suggestions) <= 5

    def test_no_duplicates(self) -> None:
        suggestions = build_prompt_suggestions(
            skill_map=_sample_skill_map(),
            completed_skills=set(),
            session_turn=1,
        )
        assert len(suggestions) == len(set(suggestions))


class TestFormatSuggestionsPrompt:
    def test_empty(self) -> None:
        assert format_suggestions_prompt([]) == ""

    def test_formats_numbered_list(self) -> None:
        result = format_suggestions_prompt(["Do X", "Do Y"])
        assert "**Suggested next steps:**" in result
        assert "1. Do X" in result
        assert "2. Do Y" in result


# ---------------------------------------------------------------------------
# Predictive suggestions
# ---------------------------------------------------------------------------


class TestBuildPredictiveSuggestions:
    def test_parses_numbered_response(self) -> None:
        mock_session = MagicMock()
        mock_session.send_and_wait = AsyncMock(
            return_value="1. Fix the failing test in auth.py\n2. Add error handling to the API endpoint\n3. Update the README"
        )
        mock_session.destroy = AsyncMock()

        async def create():
            return mock_session

        result = asyncio.run(build_predictive_suggestions(
            create_maintenance_session=create,
            transcript_tail="User asked to fix auth bugs.",
            session_state={"open_tasks": 3},
        ))
        assert result == [
            "Fix the failing test in auth.py",
            "Add error handling to the API endpoint",
            "Update the README",
        ]

    def test_handles_empty_response(self) -> None:
        mock_session = MagicMock()
        mock_session.send_and_wait = AsyncMock(return_value="")
        mock_session.destroy = AsyncMock()

        async def create():
            return mock_session

        result = asyncio.run(build_predictive_suggestions(
            create_maintenance_session=create,
            transcript_tail="",
            session_state={},
        ))
        assert result == []

    def test_handles_session_creation_error(self) -> None:
        async def create():
            raise RuntimeError("connection failed")

        result = asyncio.run(build_predictive_suggestions(
            create_maintenance_session=create,
            transcript_tail="",
            session_state={},
        ))
        assert result == []


class TestMergeSuggestions:
    def test_interleaves_and_deduplicates(self) -> None:
        result = merge_suggestions(
            heuristic=["B", "C", "D"],
            predictive=["A", "B"],
        )
        # Predictive first, then heuristic, deduplicated
        assert result == ["A", "B", "C", "D"]

    def test_respects_max_total(self) -> None:
        result = merge_suggestions(
            heuristic=["H1", "H2", "H3"],
            predictive=["P1", "P2", "P3"],
            max_total=4,
        )
        assert len(result) == 4
        assert result[0] == "P1"  # predictive first
