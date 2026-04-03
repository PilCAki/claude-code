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
