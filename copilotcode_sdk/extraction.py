from __future__ import annotations

from typing import Literal, Sequence

ExtractionMode = Literal["nudge", "enforce"]

DEFAULT_TOOL_CALL_INTERVAL = 20
DEFAULT_CHAR_THRESHOLD = 50_000
DEFAULT_MIN_TURN_GAP = 10


def should_extract(
    *,
    tool_call_count: int,
    total_chars: int,
    last_extraction_turn: int,
    current_turn: int,
    tool_call_interval: int = DEFAULT_TOOL_CALL_INTERVAL,
    char_threshold: int = DEFAULT_CHAR_THRESHOLD,
    min_turn_gap: int = DEFAULT_MIN_TURN_GAP,
) -> bool:
    if current_turn - last_extraction_turn < min_turn_gap:
        return False
    if tool_call_count >= tool_call_interval:
        return True
    if total_chars >= char_threshold:
        return True
    return False


def build_extraction_prompt(*, memory_dir: str, project_root: str) -> str:
    """Periodic backstop reminder to save durable learnings.

    The system prompt already contains standing instructions for proactive
    memory saving.  This fires as a periodic nudge for agents that forget.
    Includes a concrete example to remove ambiguity about the mechanics.
    """
    return (
        "**Memory checkpoint.** Review what you have learned in this session. "
        "If you discovered anything durable about this project — data structure, "
        "column meanings, schema relationships, data quality issues, key decisions, "
        "user preferences — save it to memory now.\n\n"
        f"Memory directory: `{memory_dir}`\n"
        f"Project root: `{project_root}`\n\n"
        "For example, if you learned that the dataset has 166K rows of claims data "
        "with a 37% realization rate, you would:\n\n"
        f"1. Write `{memory_dir}/project_dataset_structure.md`:\n"
        "```markdown\n"
        "---\n"
        "name: Dataset structure\n"
        "description: eCW claims dataset — 166K rows, key columns, realization rate\n"
        "type: project\n"
        "---\n\n"
        "The eCW transactions dataset contains ~166K claim line items...\n"
        "```\n\n"
        "2. Add to `MEMORY.md`: `- [Dataset structure](project_dataset_structure.md) "
        "— eCW claims: 166K rows, 37% realization rate`\n\n"
        "Refer to the auto memory section of your instructions for memory types "
        "and format. Continue with your current work after saving."
    )


def build_enforce_extraction_prompt(*, memory_dir: str, session_memory_path: str) -> str:
    """Enforcement-mode extraction: writes directly to session memory.

    Unlike the nudge mode, this tells the agent to write structured entries
    to the session memory file, which can later be promoted to durable memory.
    """
    return (
        "**Memory enforcement checkpoint.** You MUST save at least one learning "
        "from this session RIGHT NOW. Do not skip this step.\n\n"
        "Write a structured entry to the session memory file:\n"
        f"  Path: `{session_memory_path}`\n\n"
        "Format each entry as:\n"
        "```\n"
        "# <Title>\n"
        "<What you learned, with specific details>\n"
        "```\n\n"
        "Separate multiple entries with `---` on its own line.\n\n"
        "Focus on: data structure insights, schema relationships, "
        "key metrics, user preferences, or decisions that future sessions need.\n\n"
        "Continue with your current work after writing."
    )


def build_session_end_extraction_prompt(*, memory_dir: str, project_root: str) -> str:
    """Urgent extraction prompt fired when all skills are complete.

    This is the last chance to persist learnings before the session ends.
    """
    return (
        "**All skills are complete — final memory checkpoint.** Before this session "
        "ends, save any durable learnings you have not yet persisted. Key things "
        "to capture:\n"
        "- Schema structure and column interpretations\n"
        "- Data quality findings (nulls, duplicates, unexpected values)\n"
        "- Key metrics you computed and verified\n"
        "- Decisions or assumptions a future session should know about\n"
        "- File paths and formats of important artifacts produced\n\n"
        f"Memory directory: `{memory_dir}`\n"
        f"Project root: `{project_root}`\n\n"
        "Write at least one memory file. Future sessions on this project will "
        "benefit from what you learned today."
    )


# ---------------------------------------------------------------------------
# Session memory update prompt (for autonomous maintenance pass)
# ---------------------------------------------------------------------------

SESSION_MEMORY_SECTIONS: tuple[str, ...] = (
    "Session Title",
    "Current State",
    "Task Specification",
    "Files and Functions",
    "Workflow",
    "Errors & Corrections",
    "Codebase/System Documentation",
    "Learnings",
    "Key Results",
    "Worklog",
)


def build_session_memory_update_prompt(
    *,
    existing_memory: str,
    sections: Sequence[str] = SESSION_MEMORY_SECTIONS,
    max_total_tokens: int = 12_000,
    max_section_tokens: int = 2_000,
) -> str:
    """Build the prompt for the session-memory maintenance pass.

    This prompt is sent to a short-lived maintenance session (not the main
    conversation) to update the structured session notes file.
    """
    section_template = "\n".join(
        f"## {name}\n*Update with relevant information from the conversation.*"
        for name in sections
    )

    return (
        "You are a session-memory maintenance agent. You are NOT part of the "
        "main conversation. Your only job is to update the structured session "
        "notes below based on new conversation activity.\n\n"
        "Current session notes:\n"
        f"<current_notes_content>\n{existing_memory}\n</current_notes_content>\n\n"
        "If the notes are empty, initialize them with this structure:\n"
        f"```markdown\n{section_template}\n```\n\n"
        "Rules:\n"
        "- Preserve the section headers exactly as shown.\n"
        "- Update only the content under each section, not the headers.\n"
        "- Be specific: include file paths, function names, metric values, "
        "error messages.\n"
        "- Remove stale information that is no longer accurate.\n"
        "- Keep each section concise — no more than a few paragraphs.\n"
        f"- Stay within ~{max_section_tokens:,} tokens per section "
        f"and ~{max_total_tokens:,} tokens total.\n\n"
        "Respond with the complete updated notes document. "
        "Do not include any explanation outside the notes."
    )
