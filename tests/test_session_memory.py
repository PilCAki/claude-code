from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from copilotcode_sdk.extraction import (
    SESSION_MEMORY_SECTIONS,
    build_session_memory_update_prompt,
)
from copilotcode_sdk.memory import MemoryStore
from copilotcode_sdk.session_memory import SessionMemoryController


def _make_controller(
    tmp_path: Path, **kwargs,
) -> tuple[MemoryStore, SessionMemoryController]:
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    defaults = {
        "min_init_tokens": 10_000,
        "min_update_tokens": 5_000,
        "tool_calls_between_updates": 3,
        "timeout_seconds": 5.0,
        "promote_on_destroy": True,
    }
    defaults.update(kwargs)
    return store, SessionMemoryController(store, **defaults)


# ---------------------------------------------------------------------------
# should_extract threshold tests
# ---------------------------------------------------------------------------


def test_should_extract_below_init_threshold(tmp_path: Path) -> None:
    _, ctrl = _make_controller(tmp_path)
    assert ctrl.should_extract(5_000) is False
    assert ctrl.state.initialized is False


def test_should_extract_initializes_at_threshold(tmp_path: Path) -> None:
    _, ctrl = _make_controller(tmp_path, tool_calls_between_updates=0)
    # At 10K tokens with 0 tool call threshold, should fire
    assert ctrl.should_extract(10_000) is True
    assert ctrl.state.initialized is True


def test_should_extract_only_initializes_once(tmp_path: Path) -> None:
    _, ctrl = _make_controller(tmp_path, tool_calls_between_updates=0)
    # First: initializes and fires
    assert ctrl.should_extract(10_000) is True
    # Simulate extraction completed
    ctrl.state.tokens_at_last_extraction = 10_000
    # Below update threshold, should not fire
    assert ctrl.should_extract(12_000) is False


def test_should_extract_requires_token_growth_and_tool_calls(tmp_path: Path) -> None:
    _, ctrl = _make_controller(tmp_path, tool_calls_between_updates=3)
    ctrl.state.initialized = True
    ctrl.state.tokens_at_last_extraction = 10_000

    # Enough tokens but not enough tool calls
    ctrl.state.tool_calls_since_last_extraction = 2
    assert ctrl.should_extract(16_000) is False

    # Enough tool calls
    ctrl.state.tool_calls_since_last_extraction = 3
    assert ctrl.should_extract(16_000) is True


def test_should_extract_natural_break(tmp_path: Path) -> None:
    _, ctrl = _make_controller(tmp_path, tool_calls_between_updates=10)
    ctrl.state.initialized = True
    ctrl.state.tokens_at_last_extraction = 10_000
    ctrl.state.tool_calls_since_last_extraction = 1  # below threshold

    # Natural break: no tool calls in last turn
    assert ctrl.should_extract(16_000, has_tool_calls_in_last_turn=False) is True


def test_sequential_guard_prevents_concurrent(tmp_path: Path) -> None:
    _, ctrl = _make_controller(tmp_path, tool_calls_between_updates=0)
    ctrl.state.initialized = True
    ctrl.state.extraction_started_at = time.monotonic()

    assert ctrl.should_extract(100_000) is False


# ---------------------------------------------------------------------------
# run_extraction tests
# ---------------------------------------------------------------------------


def test_run_extraction_writes_session_memory(tmp_path: Path) -> None:
    store, ctrl = _make_controller(tmp_path)
    ctrl.state.initialized = True

    mock_session = MagicMock()
    mock_session.send_and_wait = AsyncMock(return_value={"content": "## Session Title\nTest session"})
    mock_session.destroy = AsyncMock()

    async def create_session():
        return mock_session

    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    result = asyncio.run(ctrl.run_extraction(create_session, messages, context_tokens=15_000))

    assert result is True
    assert ctrl.state.extraction_started_at is None  # cleared
    assert ctrl.state.tokens_at_last_extraction == 15_000
    assert ctrl.state.tool_calls_since_last_extraction == 0
    # Session memory file should have content
    content = store.read_session_memory()
    assert "Session Title" in content


def test_run_extraction_clears_guard_on_error(tmp_path: Path) -> None:
    store, ctrl = _make_controller(tmp_path)
    ctrl.state.initialized = True

    async def create_session():
        raise RuntimeError("connection failed")

    result = asyncio.run(ctrl.run_extraction(create_session, [], context_tokens=15_000))

    assert result is False
    assert ctrl.state.extraction_started_at is None  # cleared despite error


def test_run_extraction_updates_state(tmp_path: Path) -> None:
    store, ctrl = _make_controller(tmp_path)
    ctrl.state.initialized = True
    ctrl.state.tool_calls_since_last_extraction = 5

    mock_session = MagicMock()
    mock_session.send_and_wait = AsyncMock(return_value="Updated notes")
    mock_session.destroy = AsyncMock()

    async def create_session():
        return mock_session

    messages = [{"role": "assistant", "id": "msg-123", "content": "done"}]
    asyncio.run(ctrl.run_extraction(create_session, messages, context_tokens=20_000))

    assert ctrl.state.tokens_at_last_extraction == 20_000
    assert ctrl.state.tool_calls_since_last_extraction == 0
    assert ctrl.state.last_summarized_message_id == "msg-123"


# ---------------------------------------------------------------------------
# finalize tests
# ---------------------------------------------------------------------------


def test_finalize_promotes_memory(tmp_path: Path) -> None:
    store, ctrl = _make_controller(tmp_path, promote_on_destroy=True)
    # Write some session memory to promote
    store.append_session_memory("## Learnings\nImportant finding.")

    result = asyncio.run(ctrl.finalize())

    # promote_session_memory returns list of paths
    assert isinstance(result, list)


def test_finalize_skips_when_disabled(tmp_path: Path) -> None:
    store, ctrl = _make_controller(tmp_path, promote_on_destroy=False)
    store.append_session_memory("## Learnings\nSomething.")

    result = asyncio.run(ctrl.finalize())

    assert result == []


# ---------------------------------------------------------------------------
# Prompt tests
# ---------------------------------------------------------------------------


def test_update_prompt_has_all_sections() -> None:
    prompt = build_session_memory_update_prompt(existing_memory="")
    for section in SESSION_MEMORY_SECTIONS:
        assert section in prompt, f"Missing section: {section}"


def test_update_prompt_includes_existing_content() -> None:
    existing = "## Session Title\nPrevious content here."
    prompt = build_session_memory_update_prompt(existing_memory=existing)
    assert "Previous content here." in prompt
    assert "<current_notes_content>" in prompt


# ---------------------------------------------------------------------------
# record_tool_call
# ---------------------------------------------------------------------------


def test_record_tool_call_increments(tmp_path: Path) -> None:
    _, ctrl = _make_controller(tmp_path)
    assert ctrl.state.tool_calls_since_last_extraction == 0
    ctrl.record_tool_call()
    ctrl.record_tool_call()
    assert ctrl.state.tool_calls_since_last_extraction == 2


# ---------------------------------------------------------------------------
# Gap 1.1: In-place overwrite and transcript slicing
# ---------------------------------------------------------------------------


def test_write_session_memory_overwrites(tmp_path: Path) -> None:
    """write_session_memory replaces the file entirely, not appending."""
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    store.write_session_memory("## First version\nContent A.")
    store.write_session_memory("## Second version\nContent B.")
    content = store.read_session_memory()
    assert "Content B" in content
    assert "Content A" not in content
    assert "---" not in content  # no entry separator


def test_run_extraction_overwrites_not_appends(tmp_path: Path) -> None:
    """run_extraction should overwrite session memory, not append entries."""
    store, ctrl = _make_controller(tmp_path)
    ctrl.state.initialized = True

    # Pre-existing session memory
    store.write_session_memory("## Old notes\nStale content.")

    mock_session = MagicMock()
    mock_session.send_and_wait = AsyncMock(return_value="## Updated notes\nFresh content.")
    mock_session.destroy = AsyncMock()

    async def create_session():
        return mock_session

    messages = [{"role": "user", "id": "m1", "content": "hello"}]
    asyncio.run(ctrl.run_extraction(create_session, messages, context_tokens=15_000))

    content = store.read_session_memory()
    assert "Fresh content" in content
    assert "Stale content" not in content  # old content replaced, not appended


def test_format_transcript_tail_slices_from_message_id() -> None:
    """When last_summarized_message_id is set, only messages after it are included."""
    from copilotcode_sdk.session_memory import _format_transcript_tail

    messages = [
        {"role": "user", "id": "m1", "content": "first message"},
        {"role": "assistant", "id": "m2", "content": "second message"},
        {"role": "user", "id": "m3", "content": "third message"},
        {"role": "assistant", "id": "m4", "content": "fourth message"},
    ]
    result = _format_transcript_tail(messages, last_summarized_message_id="m2")
    assert "third message" in result
    assert "fourth message" in result
    assert "first message" not in result
    assert "second message" not in result


def test_format_transcript_tail_fallback_without_id() -> None:
    """Without a message ID, all messages are included (tail behavior)."""
    from copilotcode_sdk.session_memory import _format_transcript_tail

    messages = [
        {"role": "user", "id": "m1", "content": "msg one"},
        {"role": "assistant", "id": "m2", "content": "msg two"},
    ]
    result = _format_transcript_tail(messages, last_summarized_message_id=None)
    assert "msg one" in result
    assert "msg two" in result


def test_format_transcript_tail_unknown_id_falls_back() -> None:
    """When the ID isn't found, include all messages (graceful fallback)."""
    from copilotcode_sdk.session_memory import _format_transcript_tail

    messages = [
        {"role": "user", "id": "m1", "content": "only msg"},
    ]
    result = _format_transcript_tail(messages, last_summarized_message_id="nonexistent")
    assert "only msg" in result


# ---------------------------------------------------------------------------
# Bug fix tests: _accumulate_usage + record_tool_call wiring + destroy
# ---------------------------------------------------------------------------


def test_accumulate_usage_handles_session_event_objects(tmp_path: Path) -> None:
    """_accumulate_usage should convert SessionEvent (with to_dict()) before extracting usage."""
    from copilotcode_sdk.client import CopilotCodeSession

    fake_session = MagicMock()
    fake_session.session_id = "test"
    fake_session.workspace_path = str(tmp_path)

    session = CopilotCodeSession.__new__(CopilotCodeSession)
    session._session = fake_session
    session._smc = None
    session._copilot_client = None
    session._subagent_context = None
    session._model = None
    session._cumulative_cost = MagicMock(
        input_cost=0, output_cost=0, cache_read_cost=0, cache_creation_cost=0, total=0,
    )

    # Simulate session state
    from copilotcode_sdk.session_state import SessionState
    session._state = SessionState()
    session._event_bus = MagicMock()

    # Create a fake SessionEvent with to_dict() that returns usage
    event = MagicMock()
    event.to_dict.return_value = {
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
    }

    session._accumulate_usage(event)

    # Should have called to_dict() and recorded usage
    event.to_dict.assert_called_once()
    assert session._state.total_input_tokens == 100
    assert session._state.total_output_tokens == 50


def test_accumulate_usage_still_handles_plain_dicts(tmp_path: Path) -> None:
    """_accumulate_usage should continue to work with plain dicts."""
    from copilotcode_sdk.client import CopilotCodeSession
    from copilotcode_sdk.session_state import SessionState

    session = CopilotCodeSession.__new__(CopilotCodeSession)
    session._session = MagicMock()
    session._smc = None
    session._model = None
    session._cumulative_cost = MagicMock(
        input_cost=0, output_cost=0, cache_read_cost=0, cache_creation_cost=0, total=0,
    )
    session._state = SessionState()
    session._event_bus = MagicMock()

    session._accumulate_usage({"usage": {"input_tokens": 200, "output_tokens": 75}})

    assert session._state.total_input_tokens == 200
    assert session._state.total_output_tokens == 75


def test_maybe_run_session_memory_calls_record_tool_call(tmp_path: Path) -> None:
    """_maybe_run_session_memory should call record_tool_call when last turn has tool calls."""
    store, ctrl = _make_controller(tmp_path, tool_calls_between_updates=10)
    ctrl.state.initialized = True
    ctrl.state.tokens_at_last_extraction = 0

    from copilotcode_sdk.client import CopilotCodeSession, _last_turn_has_tool_calls
    from copilotcode_sdk.session_state import SessionState

    session = CopilotCodeSession.__new__(CopilotCodeSession)
    session._session = MagicMock()
    session._smc = ctrl
    session._copilot_client = MagicMock()
    session._subagent_context = None
    session._model = None
    session._state = SessionState()
    session._event_bus = MagicMock()

    # Mock get_messages to return messages with tool calls
    messages_with_tools = [
        {"role": "user", "content": "do something"},
        {
            "type": "assistant.message",
            "data": {
                "role": "assistant",
                "tool_requests": [{"name": "read_file", "id": "t1"}],
            },
        },
    ]
    session._session.get_messages = AsyncMock(return_value=messages_with_tools)

    assert ctrl.state.tool_calls_since_last_extraction == 0
    asyncio.run(session._maybe_run_session_memory())
    assert ctrl.state.tool_calls_since_last_extraction == 1


def test_maybe_run_session_memory_no_record_without_tool_calls(tmp_path: Path) -> None:
    """_maybe_run_session_memory should not call record_tool_call when no tool calls."""
    store, ctrl = _make_controller(tmp_path)
    ctrl.state.initialized = True

    from copilotcode_sdk.client import CopilotCodeSession
    from copilotcode_sdk.session_state import SessionState

    session = CopilotCodeSession.__new__(CopilotCodeSession)
    session._session = MagicMock()
    session._smc = ctrl
    session._copilot_client = MagicMock()
    session._subagent_context = None
    session._model = None
    session._state = SessionState()
    session._event_bus = MagicMock()

    # Messages without tool calls
    session._session.get_messages = AsyncMock(return_value=[
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ])

    asyncio.run(session._maybe_run_session_memory())
    assert ctrl.state.tool_calls_since_last_extraction == 0


def test_destroy_calls_finalize(tmp_path: Path) -> None:
    """destroy() should force extraction then call _smc.finalize()."""
    from copilotcode_sdk.client import CopilotCodeSession
    from copilotcode_sdk.session_state import SessionState

    session = CopilotCodeSession.__new__(CopilotCodeSession)
    session._session = MagicMock()
    session._session.destroy = AsyncMock()
    session._session.session_id = "test-session"
    session._session.get_messages = AsyncMock(return_value=[
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "I helped with the task"},
    ])
    session._event_bus = MagicMock()
    session._copilot_client = MagicMock()
    session._subagent_context = None

    mock_smc = MagicMock()
    mock_smc.finalize = AsyncMock(return_value=[])
    mock_smc.run_extraction = AsyncMock(return_value=True)
    session._smc = mock_smc

    asyncio.run(session.destroy())

    # Should force extraction THEN finalize
    mock_smc.run_extraction.assert_awaited_once()
    mock_smc.finalize.assert_awaited_once()
    session._session.destroy.assert_awaited_once()


def test_destroy_forces_extraction_even_without_thresholds(tmp_path: Path) -> None:
    """destroy() forces extraction regardless of should_extract() thresholds.

    After extraction, finalize() promotes session memory to durable files and
    clears the session memory file.  So we verify that promoted durable memories
    exist (meaning extraction + promotion both worked).
    """
    from copilotcode_sdk.client import CopilotCodeSession

    store, ctrl = _make_controller(tmp_path, min_init_tokens=999_999)  # absurdly high

    session = CopilotCodeSession.__new__(CopilotCodeSession)
    session._session = MagicMock()
    session._session.destroy = AsyncMock()
    session._session.session_id = "test-session"
    session._session.get_messages = AsyncMock(return_value=[
        {"role": "user", "content": "do stuff"},
        {"role": "assistant", "content": "done"},
    ])
    session._event_bus = MagicMock()
    session._subagent_context = None

    # Mock the copilot client and maintenance session
    mock_maintenance = MagicMock()
    mock_maintenance.send_and_wait = AsyncMock(return_value="## Session notes\nLearned things.")
    mock_maintenance.destroy = AsyncMock()

    mock_copilot = MagicMock()
    mock_copilot.create_session = AsyncMock(return_value=mock_maintenance)
    session._copilot_client = mock_copilot

    session._smc = ctrl

    asyncio.run(session.destroy())

    # Extraction fired + finalize promoted to durable memory + cleared session file.
    # Session memory should be empty (cleared after promotion).
    assert store.read_session_memory() == ""

    # Durable memory files should exist with the promoted content.
    durable_files = list(store.memory_dir.glob("*.md"))
    assert len(durable_files) > 0, "No durable memory files were created by promotion"
    combined = "\n".join(f.read_text(encoding="utf-8") for f in durable_files)
    assert "Learned things" in combined


def test_disconnect_does_not_call_finalize(tmp_path: Path) -> None:
    """disconnect() should NOT call _smc.finalize() — only destroy() does."""
    from copilotcode_sdk.client import CopilotCodeSession

    session = CopilotCodeSession.__new__(CopilotCodeSession)
    session._session = MagicMock()
    session._session.disconnect = AsyncMock()

    mock_smc = MagicMock()
    mock_smc.finalize = AsyncMock(return_value=[])
    session._smc = mock_smc

    asyncio.run(session.disconnect())

    mock_smc.finalize.assert_not_awaited()
    session._session.disconnect.assert_awaited_once()
