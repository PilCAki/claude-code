"""Tests for copilotcode_sdk.session_state module."""
from __future__ import annotations

import pytest

from copilotcode_sdk.session_state import (
    RequiresActionDetails,
    SessionState,
    SessionStatus,
)


def test_initial_state_is_idle_with_zero_counters():
    state = SessionState()
    assert state.status == SessionStatus.idle
    assert state.action_details is None
    assert state.turn_count == 0
    assert state.total_input_tokens == 0
    assert state.total_output_tokens == 0
    assert state.total_cache_read_tokens == 0
    assert state.total_cache_creation_tokens == 0


def test_start_turn_transitions_to_running_and_increments():
    state = SessionState()
    state.start_turn()
    assert state.status == SessionStatus.running
    assert state.turn_count == 1
    state.start_turn()
    assert state.turn_count == 2


def test_end_turn_transitions_to_idle():
    state = SessionState()
    state.start_turn()
    state.end_turn()
    assert state.status == SessionStatus.idle


def test_require_action_transitions_with_details():
    state = SessionState()
    state.start_turn()
    details = RequiresActionDetails(tool_name="bash", action_description="Run ls")
    state.require_action(details)
    assert state.status == SessionStatus.requires_action
    assert state.action_details is details


def test_transition_to_clears_action_details_when_not_requires_action():
    state = SessionState()
    details = RequiresActionDetails(tool_name="bash", action_description="Run ls")
    state.require_action(details)
    assert state.action_details is not None
    state.transition_to(SessionStatus.running)
    assert state.action_details is None


def test_listener_called_on_state_change():
    state = SessionState()
    calls: list[tuple] = []

    def listener(s, status, details):
        calls.append((s, status, details))

    state.add_listener(listener)
    state.start_turn()
    assert len(calls) == 1
    assert calls[0][1] == SessionStatus.running


def test_listener_not_called_when_same_state():
    state = SessionState()
    state.transition_to(SessionStatus.running)
    calls: list[tuple] = []

    def listener(s, status, details):
        calls.append((s, status, details))

    state.add_listener(listener)
    # Transition to the same state again -- should NOT fire
    state.transition_to(SessionStatus.running)
    assert len(calls) == 0


def test_remove_listener():
    state = SessionState()
    calls: list[tuple] = []

    def listener(s, status, details):
        calls.append((s, status, details))

    state.add_listener(listener)
    state.remove_listener(listener)
    state.start_turn()
    assert len(calls) == 0


def test_listener_exception_does_not_crash():
    state = SessionState()

    def bad_listener(s, status, details):
        raise RuntimeError("boom")

    calls: list[tuple] = []

    def good_listener(s, status, details):
        calls.append((s, status, details))

    state.add_listener(bad_listener)
    state.add_listener(good_listener)
    state.start_turn()  # should not raise
    assert len(calls) == 1


def test_record_usage_accumulates_tokens():
    state = SessionState()
    state.record_usage(input_tokens=10, output_tokens=5)
    state.record_usage(input_tokens=20, output_tokens=15, cache_read_tokens=3, cache_creation_tokens=2)
    assert state.total_input_tokens == 30
    assert state.total_output_tokens == 20
    assert state.total_cache_read_tokens == 3
    assert state.total_cache_creation_tokens == 2


def test_total_tokens_property():
    state = SessionState()
    state.record_usage(input_tokens=100, output_tokens=50)
    assert state.total_tokens == 150


def test_to_dict_includes_all_fields():
    state = SessionState()
    state.start_turn()
    state.record_usage(input_tokens=10, output_tokens=5)
    d = state.to_dict()
    assert d["status"] == "running"
    assert d["turn_count"] == 1
    assert d["total_input_tokens"] == 10
    assert d["total_output_tokens"] == 5
    assert d["total_tokens"] == 15
    assert "action_details" not in d


def test_to_dict_includes_action_details_when_present():
    state = SessionState()
    details = RequiresActionDetails(tool_name="bash", action_description="Run ls")
    state.require_action(details)
    d = state.to_dict()
    assert d["status"] == "requires_action"
    assert d["action_details"]["tool_name"] == "bash"
    assert d["action_details"]["action_description"] == "Run ls"
