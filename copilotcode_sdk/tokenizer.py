"""Token counting with optional tiktoken support.

Uses tiktoken when available for accurate counts, falls back to a
character-based heuristic (~4 chars per token for English text).
"""
from __future__ import annotations

from typing import Any

_encoder: Any = None
_tiktoken_available: bool | None = None


def _get_encoder() -> Any:
    """Lazily load the tiktoken encoder."""
    global _encoder, _tiktoken_available
    if _tiktoken_available is False:
        return None
    if _encoder is not None:
        return _encoder
    try:
        import tiktoken
        # cl100k_base is the encoding used by Claude models
        _encoder = tiktoken.get_encoding("cl100k_base")
        _tiktoken_available = True
        return _encoder
    except (ImportError, Exception):
        _tiktoken_available = False
        return None


def count_tokens(text: str) -> int:
    """Count tokens in text.

    Uses tiktoken if available, otherwise estimates at ~4 chars per token.
    """
    encoder = _get_encoder()
    if encoder is not None:
        return len(encoder.encode(text))
    return estimate_tokens(text)


def estimate_tokens(text: str) -> int:
    """Estimate token count using character heuristic (~4 chars/token)."""
    return max(1, len(text) // 4) if text else 0


def count_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Count total tokens across a list of messages.

    Each message contributes its content tokens plus ~4 tokens of overhead
    for role/formatting.  Handles both plain dicts and SDK
    ``SessionEvent.to_dict()`` output where content lives in ``data``.
    """
    total = 0
    for msg in messages:
        # Support SDK event dicts: {"type": "...", "data": {"content": ...}}
        content = msg.get("content", "")
        if not content:
            data = msg.get("data")
            if isinstance(data, dict):
                content = data.get("content", "") or data.get("message", "") or ""
        if isinstance(content, str):
            total += count_tokens(content) + 4
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "")
                    if text:
                        total += count_tokens(text)
            total += 4
        else:
            total += estimate_tokens(str(content)) + 4
    return total


def has_tiktoken() -> bool:
    """Check if tiktoken is available."""
    _get_encoder()
    return _tiktoken_available is True
