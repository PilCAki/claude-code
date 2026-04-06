"""Tests for copilotcode_sdk.tokenizer module."""
from __future__ import annotations

from copilotcode_sdk.tokenizer import (
    count_message_tokens,
    count_tokens,
    estimate_tokens,
    has_tiktoken,
)


def test_estimate_tokens_empty() -> None:
    assert estimate_tokens("") == 0


def test_estimate_tokens_short() -> None:
    # "hello" is 5 chars → 5//4 = 1
    assert estimate_tokens("hello") >= 1


def test_estimate_tokens_longer() -> None:
    text = "x" * 400
    # ~100 tokens at 4 chars/token
    assert 90 <= estimate_tokens(text) <= 110


def test_count_tokens_returns_positive() -> None:
    result = count_tokens("Hello, world!")
    assert result > 0


def test_count_tokens_empty() -> None:
    result = count_tokens("")
    assert result == 0


def test_count_message_tokens_basic() -> None:
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    total = count_message_tokens(messages)
    assert total > 0
    # Should be at least the overhead (4 per message = 8) plus content tokens
    assert total >= 8


def test_count_message_tokens_with_blocks() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Here is some code."},
                {"type": "tool_use", "id": "123"},
            ],
        },
    ]
    total = count_message_tokens(messages)
    assert total > 0


def test_count_message_tokens_empty() -> None:
    assert count_message_tokens([]) == 0


def test_has_tiktoken_is_bool() -> None:
    result = has_tiktoken()
    assert isinstance(result, bool)


def test_count_tokens_consistency() -> None:
    """count_tokens should return consistent results for the same input."""
    text = "The quick brown fox jumps over the lazy dog."
    r1 = count_tokens(text)
    r2 = count_tokens(text)
    assert r1 == r2
