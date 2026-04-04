from __future__ import annotations

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
    """
    return (
        "**Memory checkpoint.** Review what you have learned in this session. "
        "If you discovered anything durable about this project — data structure, "
        "column meanings, schema relationships, data quality issues, key decisions, "
        "user preferences — save it to memory now.\n\n"
        f"Memory directory: `{memory_dir}`\n"
        f"Project root: `{project_root}`\n\n"
        "Write a `.md` file with frontmatter (name, description, type) and add a "
        "pointer to `MEMORY.md`. Refer to the Memory Guidance section of your "
        "instructions for memory types and format. Continue with your current "
        "work after saving."
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
