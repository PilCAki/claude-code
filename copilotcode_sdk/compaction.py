from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Sequence

CompactionMode = Literal["full", "partial", "up_to"]

_NO_TOOLS_TRAILER = (
    "REMINDER: Do NOT call any tools. Respond with plain text only — "
    "an `<analysis>` block followed by a `<summary>` block."
)

_SECTIONS_COMMON = """\
1. **Primary request and intent** — what the user originally asked for and why
2. **Key technical concepts** — domain terms, architecture patterns, algorithms discussed
3. **Files and code sections** — exact file paths, function/class names, relevant snippets
4. **Errors and fixes** — what went wrong, what was tried, what resolved it
5. **Problem solving** — reasoning chains, trade-offs considered, decisions made
6. **All user messages** — preserve every non-tool user message (requirements, feedback, corrections)
7. **Pending tasks** — what still needs to be done, in priority order\
"""

_SECTION_8_FULL = (
    "8. **Current Work** — what is actively being worked on right now, "
    "including partial progress and next immediate step"
)
_SECTION_9_FULL = (
    "9. **Optional Next Step** — if obvious, what the assistant should do next "
    "after resuming from this summary"
)

_SECTION_8_UP_TO = (
    "8. **Work Completed** — what was accomplished in the summarized portion"
)
_SECTION_9_UP_TO = (
    "9. **Context for Continuing Work** — anything from the summarized portion "
    "that is needed to understand the preserved turns that follow"
)


def build_compaction_prompt(
    *,
    mode: CompactionMode = "full",
    preserve_recent: int = 0,
    up_to_turn: int = 0,
    extra_instructions: str = "",
) -> str:
    """Build a compaction prompt for the given mode.

    Args:
        mode: ``"full"`` summarizes the entire conversation, ``"partial"``
            summarizes only older messages (preserving the most recent
            *preserve_recent* turns), ``"up_to"`` summarizes everything up to
            *up_to_turn*.
        preserve_recent: Number of recent turns to preserve (for partial mode).
        up_to_turn: Turn number to summarize up to (for up_to mode).
        extra_instructions: Project-specific compaction guidance appended to
            the prompt.
    """
    parts: list[str] = []

    if mode == "full":
        parts.append(
            "The conversation context is being compacted. Produce a structured summary "
            "that preserves everything a fresh context window would need to continue "
            "this work seamlessly. Cover these sections:\n\n"
        )
    elif mode == "partial":
        parts.append(
            f"The conversation is approaching its context limit. Summarize all messages "
            f"EXCEPT the most recent {preserve_recent} turns. Those recent turns will be "
            f"preserved verbatim. Your summary should cover only the older portion. "
            f"Cover these sections:\n\n"
        )
    elif mode == "up_to":
        parts.append(
            f"Summarize the conversation up to and including turn {up_to_turn}. "
            f"Messages after turn {up_to_turn} will be preserved verbatim. "
            f"Cover these sections:\n\n"
        )

    parts.append(_SECTIONS_COMMON)
    parts.append("\n")

    # Mode-dependent sections 8-9
    if mode == "up_to":
        parts.append(_SECTION_8_UP_TO)
        parts.append("\n")
        parts.append(_SECTION_9_UP_TO)
    else:
        parts.append(_SECTION_8_FULL)
        parts.append("\n")
        parts.append(_SECTION_9_FULL)

    parts.append(
        "\n\nBe specific and concrete. Include file paths, metric values, and exact names. "
        "A vague summary is worse than no summary."
    )

    if extra_instructions:
        parts.append(f"\n\nAdditional compaction guidance:\n{extra_instructions}")

    parts.append(f"\n\n{_NO_TOOLS_TRAILER}")

    return "".join(parts)


def format_transcript_for_compaction(
    messages: Sequence[Any],
    *,
    max_chars: int = 60_000,
    max_per_message: int = 3_000,
) -> str:
    """Format conversation messages into a ``<transcript>`` block for compaction.

    Messages are formatted as ``[role]: content`` lines.  Individual messages
    are truncated to *max_per_message* characters.  If the total exceeds
    *max_chars*, the **oldest** messages are dropped so that the most-recent
    context is preserved.
    """
    formatted: list[str] = []
    for msg in messages:
        msg = _ensure_message_dict(msg)
        role = msg.get("role") or msg.get("type") or "unknown"
        content = msg.get("content", "")
        if isinstance(content, list):
            # Content blocks: extract text parts
            content = " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        if not isinstance(content, str):
            content = str(content)
        if len(content) > max_per_message:
            content = content[:max_per_message] + "…"
        formatted.append(f"[{role}]: {content}")

    # Drop oldest messages until we fit within max_chars
    while formatted and sum(len(line) for line in formatted) > max_chars:
        formatted.pop(0)

    body = "\n".join(formatted)
    return f"<transcript>\n{body}\n</transcript>"


def _ensure_message_dict(message: Any) -> dict[str, Any]:
    """Normalize SDK event objects into dicts for transcript formatting."""
    if isinstance(message, dict):
        return message

    to_dict = getattr(message, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, dict):
            return converted
        return {"content": str(converted)}

    try:
        converted = vars(message)
    except TypeError:
        return {"content": str(message)}
    return converted if isinstance(converted, dict) else {"content": str(converted)}


def build_handoff_context(
    *,
    compaction_summary: str,
    skill_catalog_text: str = "",
    instruction_content: str = "",
    memory_index: str = "",
    recent_turns_note: str = "",
) -> str:
    sections = [
        "## Session Continuation\n",
        "The previous context was compacted. Below is the preserved context.\n",
        "### Compaction Summary\n",
        compaction_summary,
    ]

    if skill_catalog_text:
        sections.append("\n### Skill Catalog\n")
        sections.append(skill_catalog_text)

    if instruction_content:
        sections.append("\n### Workspace Instructions\n")
        sections.append(instruction_content)

    if memory_index:
        sections.append("\n### Memory Index\n")
        sections.append(memory_index)

    if recent_turns_note:
        sections.append("\n### Recent Turns\n")
        sections.append(recent_turns_note)

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Compaction response parser
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CompactionResult:
    """Parsed compaction response."""

    analysis: str
    summary: str
    raw_response: str


_ANALYSIS_RE = re.compile(r"<analysis>(.*?)</analysis>", re.DOTALL)
_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)


def parse_compaction_response(text: str) -> CompactionResult:
    """Extract ``<analysis>`` and ``<summary>`` from a compaction response.

    Falls back gracefully: if tags are missing, the full text is used as
    the summary (and analysis is empty).
    """
    analysis_match = _ANALYSIS_RE.search(text)
    summary_match = _SUMMARY_RE.search(text)

    analysis = analysis_match.group(1).strip() if analysis_match else ""
    summary = summary_match.group(1).strip() if summary_match else text.strip()

    return CompactionResult(
        analysis=analysis,
        summary=summary,
        raw_response=text,
    )
