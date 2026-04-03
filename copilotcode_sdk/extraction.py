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
    return (
        "**Session memory extraction checkpoint.** "
        "Review what you have learned in this session so far and save any durable "
        "learnings to the memory system. Durable learnings are facts, preferences, "
        "or context that would be valuable in future conversations about this project.\n\n"
        "Memory types to consider:\n"
        "- **user**: Role, preferences, expertise level\n"
        "- **feedback**: Corrections or confirmed approaches\n"
        "- **project**: Ongoing work, goals, decisions, timelines\n"
        "- **reference**: Pointers to external resources\n\n"
        f"Memory directory: `{memory_dir}`\n"
        f"Project root: `{project_root}`\n\n"
        "Only save things that are not already derivable from the code or git history. "
        "Skip ephemeral task details. If you have nothing durable to save, that is fine — "
        "do not force it. Continue with your current work after this checkpoint."
    )
