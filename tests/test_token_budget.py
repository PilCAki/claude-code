"""Tests for copilotcode_sdk.token_budget module."""
from __future__ import annotations

import pytest
from copilotcode_sdk.token_budget import (
    TokenBudget,
    parse_token_budget,
    strip_budget_directive,
    format_budget_status,
)


# --- parse_token_budget shorthand ---

def test_parse_shorthand_500k():
    result = parse_token_budget("+500k")
    assert result is not None
    assert result.tokens == 500_000

def test_parse_shorthand_1_5m():
    result = parse_token_budget("+1.5m")
    assert result is not None
    assert result.tokens == 1_500_000

def test_parse_shorthand_2b():
    result = parse_token_budget("+2b")
    assert result is not None
    assert result.tokens == 2_000_000_000


# --- parse_token_budget verbose ---

def test_parse_verbose_use():
    result = parse_token_budget("use 1M tokens")
    assert result is not None
    assert result.tokens == 1_000_000

def test_parse_verbose_spend():
    result = parse_token_budget("spend 500k tokens")
    assert result is not None
    assert result.tokens == 500_000


# --- parse_token_budget returns None ---

def test_parse_no_budget():
    assert parse_token_budget("just a regular message") is None
    assert parse_token_budget("") is None
    assert parse_token_budget("500k") is None  # missing +


# --- case insensitive ---

def test_parse_case_insensitive_upper():
    result = parse_token_budget("+500K")
    assert result is not None
    assert result.tokens == 500_000

def test_parse_case_insensitive_mixed():
    result = parse_token_budget("+1.5M")
    assert result is not None
    assert result.tokens == 1_500_000


# --- strip_budget_directive ---

def test_strip_shorthand():
    assert strip_budget_directive("Fix the bug +500k") == "Fix the bug"

def test_strip_verbose():
    assert strip_budget_directive("use 1M tokens and fix the bug") == "and fix the bug"


# --- TokenBudget.remaining ---

def test_remaining():
    b = TokenBudget(raw="+500k", tokens=500_000, consumed=200_000)
    assert b.remaining == 300_000

def test_remaining_over_consumed():
    b = TokenBudget(raw="+500k", tokens=500_000, consumed=600_000)
    assert b.remaining == 0


# --- TokenBudget.progress ---

def test_progress_zero():
    b = TokenBudget(raw="+500k", tokens=500_000, consumed=0)
    assert b.progress == pytest.approx(0.0)

def test_progress_half():
    b = TokenBudget(raw="+500k", tokens=500_000, consumed=250_000)
    assert b.progress == pytest.approx(0.5)

def test_progress_full():
    b = TokenBudget(raw="+500k", tokens=500_000, consumed=500_000)
    assert b.progress == pytest.approx(1.0)

def test_progress_zero_tokens():
    b = TokenBudget(raw="+0k", tokens=0, consumed=0)
    assert b.progress == 1.0


# --- TokenBudget.exhausted ---

def test_exhausted_true():
    b = TokenBudget(raw="+500k", tokens=500_000, consumed=500_000)
    assert b.exhausted is True

def test_exhausted_over():
    b = TokenBudget(raw="+500k", tokens=500_000, consumed=600_000)
    assert b.exhausted is True

def test_exhausted_false():
    b = TokenBudget(raw="+500k", tokens=500_000, consumed=100_000)
    assert b.exhausted is False


# --- format_budget_status ---

def test_format_in_progress():
    b = TokenBudget(raw="+500k", tokens=500_000, consumed=250_000)
    status = format_budget_status(b)
    assert "250k/500k" in status
    assert "50%" in status
    assert "exhausted" not in status.lower()

def test_format_exhausted():
    b = TokenBudget(raw="+500k", tokens=500_000, consumed=500_000)
    status = format_budget_status(b)
    assert "exhausted" in status.lower()
    assert "100%" in status


# --- Edge cases ---

def test_parse_zero_k():
    result = parse_token_budget("+0k")
    assert result is not None
    assert result.tokens == 0

def test_parse_very_large_budget():
    result = parse_token_budget("+999b")
    assert result is not None
    assert result.tokens == 999_000_000_000
