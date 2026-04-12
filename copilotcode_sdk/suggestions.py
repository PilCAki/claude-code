"""Prompt suggestion system.

Proposes next actions based on completed work, skill catalog state,
task progress, and session context.  Optionally, a predictive model
can be used for richer, context-aware suggestions.
"""
from __future__ import annotations

import re
from typing import Any, Awaitable, Callable, Mapping, Sequence

from .tasks import TaskRecord, TaskStatus


def build_prompt_suggestions(
    *,
    skill_map: Mapping[str, Mapping[str, str]] | None = None,
    completed_skills: set[str] | None = None,
    open_tasks: Sequence[TaskRecord] | None = None,
    completed_task_count: int = 0,
    session_turn: int = 0,
    recent_tools: Sequence[str] | None = None,
) -> list[str]:
    """Build a list of contextual prompt suggestions.

    Returns up to 5 short actionable suggestion strings that the UI
    can present to the user as next-step hints.
    """
    suggestions: list[str] = []
    _completed_skills = completed_skills or set()
    _skill_map = skill_map or {}
    _open_tasks = list(open_tasks or [])
    _recent = list(recent_tools or [])

    # 1. Skill-based suggestions
    if _skill_map:
        _add_skill_suggestions(
            suggestions, _skill_map, _completed_skills,
        )

    # 2. Task-based suggestions
    if _open_tasks:
        _add_task_suggestions(
            suggestions, _open_tasks, completed_task_count,
        )

    # 3. Session-phase suggestions
    _add_session_suggestions(
        suggestions, session_turn, _recent, _completed_skills, _skill_map,
    )

    # Deduplicate preserving order, cap at 5
    seen: set[str] = set()
    unique: list[str] = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique[:5]


def format_suggestions_prompt(suggestions: list[str]) -> str:
    """Format suggestions as a prompt-injectable string."""
    if not suggestions:
        return ""
    lines = ["**Suggested next steps:**"]
    for i, suggestion in enumerate(suggestions, 1):
        lines.append(f"{i}. {suggestion}")
    return "\n".join(lines)


def _add_skill_suggestions(
    suggestions: list[str],
    skill_map: Mapping[str, Mapping[str, str]],
    completed_skills: set[str],
) -> None:
    """Add suggestions based on skill catalog state."""
    completed_types = {
        skill_map[name].get("type", "")
        for name in completed_skills
        if name in skill_map
    }

    available: list[str] = []
    for name, fm in skill_map.items():
        if name in completed_skills:
            continue
        requires = fm.get("requires", "").strip().lower()
        if not requires or requires == "none" or requires in completed_types:
            available.append(name)

    if available:
        if len(available) == 1:
            suggestions.append(
                f"Run the `{available[0]}` skill — its prerequisites are met."
            )
        else:
            names = ", ".join(f"`{n}`" for n in available[:3])
            suggestions.append(
                f"Available skills with met prerequisites: {names}"
            )

    if completed_skills and len(completed_skills) == len(skill_map):
        suggestions.append(
            "All skills complete. Consider saving learnings to memory."
        )


def _add_task_suggestions(
    suggestions: list[str],
    open_tasks: Sequence[TaskRecord],
    completed_count: int,
) -> None:
    """Add suggestions based on task state."""
    pending = [t for t in open_tasks if t.status == TaskStatus.pending]
    in_progress = [t for t in open_tasks if t.status == TaskStatus.in_progress]

    if in_progress:
        task = in_progress[0]
        suggestions.append(
            f"Continue working on task #{task.id}: {task.subject}"
        )
    elif pending:
        task = pending[0]
        suggestions.append(
            f"Start the next task: #{task.id} — {task.subject}"
        )

    total = completed_count + len(open_tasks)
    if total > 0 and completed_count > 0:
        pct = int(completed_count / total * 100)
        if pct >= 80 and pending:
            suggestions.append(
                f"Almost done ({pct}% complete) — {len(pending)} task(s) remaining."
            )


def _add_session_suggestions(
    suggestions: list[str],
    session_turn: int,
    recent_tools: Sequence[str],
    completed_skills: set[str],
    skill_map: Mapping[str, Mapping[str, str]],
) -> None:
    """Add suggestions based on session phase."""
    if session_turn <= 2:
        if skill_map and not completed_skills:
            suggestions.append(
                "Review the skill catalog to plan your approach."
            )
        return

    # Long session without memory saves
    if session_turn > 30:
        suggestions.append(
            "Consider saving durable learnings from this session to memory."
        )

    # After heavy editing, suggest verification
    recent_lower = [t.lower() for t in recent_tools]
    edit_count = sum(1 for t in recent_lower if t in ("edit", "write", "write_file"))
    if edit_count >= 3 and "execute" not in recent_lower[-3:]:
        suggestions.append(
            "You've made several edits — consider running tests to verify."
        )


# ---------------------------------------------------------------------------
# Predictive suggestions (model-based)
# ---------------------------------------------------------------------------

_PREDICTIVE_PROMPT = """\
Based on the conversation below, suggest {max_suggestions} specific next \
actions the user should take. Format: one suggestion per line, each under \
80 characters. Focus on concrete actions, not generic advice.

{transcript_tail}

Session state: {session_state}
"""


async def build_predictive_suggestions(
    *,
    create_maintenance_session: Callable[[], Awaitable[Any]],
    transcript_tail: str,
    session_state: dict[str, Any],
    max_suggestions: int = 3,
) -> list[str]:
    """Use a lightweight maintenance session to predict next actions.

    Forks a no-tool session, sends the recent conversation context,
    and parses the numbered suggestions from the response.
    Returns an empty list on any error.
    """
    try:
        session = await create_maintenance_session()
        try:
            prompt = _PREDICTIVE_PROMPT.format(
                max_suggestions=max_suggestions,
                transcript_tail=transcript_tail[:15_000],
                session_state=str(session_state)[:2_000],
            )
            result = await session.send_and_wait(prompt, timeout=3600.0)
            text = ""
            if isinstance(result, str):
                text = result
            elif isinstance(result, dict):
                text = str(result.get("content", ""))
            return _parse_numbered_suggestions(text, max_suggestions)
        finally:
            destroy = getattr(session, "destroy", None)
            if destroy:
                await destroy()
    except Exception:
        return []


def _parse_numbered_suggestions(text: str, max_count: int = 3) -> list[str]:
    """Parse numbered lines like '1. Do X' into a list of suggestion strings."""
    lines = text.strip().splitlines()
    suggestions: list[str] = []
    for line in lines:
        # Strip leading number + dot/paren: "1. Do X" → "Do X"
        cleaned = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()
        if cleaned and len(cleaned) <= 120:
            suggestions.append(cleaned)
        if len(suggestions) >= max_count:
            break
    return suggestions


def merge_suggestions(
    heuristic: list[str],
    predictive: list[str],
    *,
    max_total: int = 5,
) -> list[str]:
    """Merge heuristic and predictive suggestions, deduplicating.

    Predictive suggestions come first (they're typically more contextual),
    followed by heuristic ones. Duplicates are removed preserving order.
    """
    seen: set[str] = set()
    merged: list[str] = []
    for s in (*predictive, *heuristic):
        if s not in seen:
            seen.add(s)
            merged.append(s)
    return merged[:max_total]
