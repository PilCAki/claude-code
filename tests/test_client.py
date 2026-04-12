from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from copilotcode_sdk import CopilotCodeClient, CopilotCodeConfig, CopilotCodeSession
from copilotcode_sdk.memory import MemoryStore
from copilotcode_sdk.client import (
    _directory_check,
    _finalize_smoke_report,
    _nearest_existing_parent,
    _write_transcript_artifact,
)
from copilotcode_sdk.events import turn_completed as _turn_completed_event
from copilotcode_sdk.reports import CheckResult, PreflightReport, SmokeTestReport


class FakeSession:
    def __init__(self) -> None:
        self.session_id = "fake-session"
        self.workspace_path = "fake-workspace"
        self.sent: list[tuple[str, list[dict[str, object]] | None, str | None]] = []
        self.destroyed = False
        self.disconnected = False

    async def send(
        self,
        prompt: str,
        *,
        attachments=None,
        mode=None,
    ) -> str:
        self.sent.append((prompt, attachments, mode))
        return "message-id"

    async def send_and_wait(
        self,
        prompt: str,
        *,
        attachments=None,
        mode=None,
        timeout=60.0,
    ) -> dict[str, object]:
        self.sent.append((prompt, attachments, mode))
        return {"status": "ok", "timeout": timeout}

    async def get_messages(self) -> list[dict[str, str]]:
        return [{"type": "assistant", "content": "done"}]

    async def disconnect(self) -> None:
        self.disconnected = True

    async def destroy(self) -> None:
        self.destroyed = True

    def on(self, callback):
        """Stub for SDK session event subscription."""
        return lambda: None  # unsubscribe noop


class FakeCopilotClient:
    def __init__(self) -> None:
        self.create_calls: list[dict[str, object]] = []
        self.resume_calls: list[tuple[str, dict[str, object]]] = []
        self.session = FakeSession()

    async def create_session(self, **kwargs):
        self.create_calls.append(kwargs)
        return self.session

    async def resume_session(self, session_id, **kwargs):
        self.resume_calls.append((session_id, kwargs))
        return self.session


class ExplodingSession(FakeSession):
    async def send_and_wait(
        self,
        prompt: str,
        *,
        attachments=None,
        mode=None,
        timeout=60.0,
    ) -> dict[str, object]:
        raise RuntimeError("session failed")


def test_create_session_wires_prompt_agents_skills_and_hooks(tmp_path: Path) -> None:
    fake_client = FakeCopilotClient()
    config = CopilotCodeConfig(
        model="claude-sonnet-4.6",
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        enabled_skills=("verify", "remember"),
        default_agent="planner",
    )
    client = CopilotCodeClient(config, copilot_client=fake_client)

    session = asyncio.run(client.create_session(session_id="session-123"))
    call = fake_client.create_calls[0]

    assert isinstance(session, CopilotCodeSession)
    assert call["session_id"] == "session-123"
    assert call["system_message"]["mode"] == "append"
    assert "## Core Operating Rules" in call["system_message"]["content"]
    assert "# CopilotCode" in call["system_message"]["content"]
    assert call["working_directory"] == str(tmp_path.resolve())
    assert call["agent"] == "planner"
    assert callable(call["on_permission_request"])
    assert "on_pre_tool_use" in call["hooks"]
    assert any(agent["name"] == "verifier" for agent in call["custom_agents"])
    assert call["disabled_skills"] == ["batch", "debug", "simplify", "skillify"]


def test_resume_session_uses_same_wiring(tmp_path: Path) -> None:
    fake_client = FakeCopilotClient()
    config = CopilotCodeConfig(
        model="claude-sonnet-4.6",
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
    )
    client = CopilotCodeClient(config, copilot_client=fake_client)

    session = asyncio.run(client.resume_session("resume-456"))

    assert isinstance(session, CopilotCodeSession)
    assert fake_client.resume_calls[0][0] == "resume-456"
    assert fake_client.resume_calls[0][1]["working_directory"] == str(tmp_path.resolve())


def test_session_memory_helpers_round_trip(tmp_path: Path) -> None:
    fake_client = FakeCopilotClient()
    config = CopilotCodeConfig(
        model="claude-sonnet-4.6",
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
    )
    client = CopilotCodeClient(config, copilot_client=fake_client)
    session = asyncio.run(client.create_session())

    path = session.remember(
        title="Style Preference",
        description="Keep responses concise.",
        memory_type="user",
        content="The user prefers concise responses with concrete outcomes.",
    )
    relevant = session.relevant_memories("concise responses")

    assert path.exists()
    assert relevant
    assert relevant[0].name == "Style Preference"


def test_resolved_cli_path_prefers_explicit_path_over_shell_lookup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    explicit_cli = tmp_path / "copilot-custom.exe"
    explicit_cli.write_text("", encoding="utf-8")
    client = CopilotCodeClient(CopilotCodeConfig(model="claude-sonnet-4.6", working_directory=tmp_path, cli_path=str(explicit_cli)))

    monkeypatch.setattr("copilotcode_sdk.client.shutil.which", lambda value: None)

    assert client._resolved_cli_path(None) == str(explicit_cli)


def test_directory_helpers_cover_non_directory_and_parent_resolution(tmp_path: Path) -> None:
    file_path = tmp_path / "not-a-dir"
    file_path.write_text("x", encoding="utf-8")

    report = _directory_check("app_config_directory", file_path)

    assert report.status == "error"
    assert "not a directory" in report.message
    assert _nearest_existing_parent(tmp_path / "a" / "b" / "c") == tmp_path


def test_transcript_and_report_helpers_write_expected_payloads(tmp_path: Path) -> None:
    preflight = PreflightReport(
        product_name="CopilotCode",
        require_auth=False,
        working_directory=str(tmp_path),
        app_config_directory="cfg",
        memory_directory="mem",
        copilot_config_directory="copilot",
        cli_path=None,
        checks=(CheckResult(name="python_sdk", status="ok", message="ok"),),
    )
    report = SmokeTestReport(
        product_name="CopilotCode",
        live=True,
        success=True,
        preflight=preflight,
        session_created=True,
        prompt_roundtrip=True,
        session_id="session-xyz",
        workspace_path="workspace",
        prompt="Reply OK",
    )

    transcript_path = _write_transcript_artifact(
        report,
        [{"type": "assistant", "content": "ok"}],
        tmp_path / "transcripts",
    )
    final = _finalize_smoke_report(
        report,
        save_report_path=tmp_path / "report.json",
    )

    assert transcript_path is not None and transcript_path.exists()
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert payload["session_id"] == "session-xyz"
    assert final.report_path == str((tmp_path / "report.json").resolve())


def test_transcript_artifact_normalizes_session_event_like_objects(tmp_path: Path) -> None:
    class EventLike:
        def to_dict(self) -> dict[str, object]:
            return {
                "type": "assistant.message",
                "data": {"role": "assistant", "content": "ok from live event"},
            }

    preflight = PreflightReport(
        product_name="CopilotCode",
        require_auth=False,
        working_directory=str(tmp_path),
        app_config_directory="cfg",
        memory_directory="mem",
        copilot_config_directory="copilot",
        cli_path=None,
        checks=(CheckResult(name="python_sdk", status="ok", message="ok"),),
    )
    report = SmokeTestReport(
        product_name="CopilotCode",
        live=True,
        success=True,
        preflight=preflight,
        session_created=True,
        prompt_roundtrip=True,
        session_id="session-live",
        workspace_path="workspace",
        prompt="Reply OK",
    )

    transcript_path = _write_transcript_artifact(
        report,
        [EventLike()],
        tmp_path / "transcripts",
    )

    assert transcript_path is not None and transcript_path.exists()
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert payload["messages"][0]["type"] == "assistant.message"
    assert payload["messages"][0]["data"]["content"] == "ok from live event"


def test_session_kwargs_passes_skill_directories_to_system_message(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "test-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-skill\ndescription: A test.\ntype: test-type\nrequires: none\n---\n\n# Test\n",
        encoding="utf-8",
    )

    config = CopilotCodeConfig(
        model="claude-sonnet-4.6",
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        enabled_skills=(),
        extra_skill_directories=[str(tmp_path / "skills")],
    )
    client = CopilotCodeClient(config)
    kwargs = client._session_kwargs()

    # Skill catalog is non-cacheable, so it's in the dynamic render, not the
    # cacheable system_message.  Verify the assembler has it.
    assert client.assembler is not None
    assert "Available Skills" in client.assembler.render_dynamic()
    assert "test-skill" in client.assembler.render_dynamic()


def test_passive_skill_detection_disabled(tmp_path: Path) -> None:
    """Writing files to a skill's output dir does NOT mark it complete.

    Skill completion is handled exclusively by the CompleteSkill tool.
    """
    skill_dir = tmp_path / "skills" / "test-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-skill\ndescription: A test.\ntype: test-type\noutputs: outputs/test/\nrequires: none\n---\n\n# Test\n",
        encoding="utf-8",
    )

    config = CopilotCodeConfig(
        model="claude-sonnet-4.6",
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        enabled_skills=(),
        extra_skill_directories=[str(tmp_path / "skills")],
    )
    client = CopilotCodeClient(config)
    kwargs = client._session_kwargs()
    hooks = kwargs["hooks"]

    result = hooks["on_post_tool_use"](
        {
            "toolName": "write_file",
            "toolArgs": {"path": str(tmp_path / "outputs" / "test" / "result.csv")},
            "toolResult": "File written.",
        },
        {},
    )

    # Passive detection disabled — writing files does NOT trigger completion
    assert result is None


# ---------------------------------------------------------------------------
# Wave 1+2: Scaffold-to-runtime wiring tests
# ---------------------------------------------------------------------------


def test_assembler_stored_on_client(tmp_path: Path) -> None:
    """Client should store the assembler after _session_kwargs() is called."""
    config = CopilotCodeConfig(
        model="claude-sonnet-4.6",
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
    )
    client = CopilotCodeClient(config)

    assert client.assembler is None  # not yet built

    client._session_kwargs()

    assert client.assembler is not None
    assert client.assembler.has("intro")
    assert client.assembler.has("core_rules")


def test_cacheable_only_is_shorter_than_full_render(tmp_path: Path) -> None:
    """render(cacheable_only=True) should be shorter than render() when dynamic sections exist."""
    skill_dir = tmp_path / "skills" / "test-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-skill\ndescription: A test.\ntype: test-type\nrequires: none\n---\n\n# Test\n",
        encoding="utf-8",
    )
    config = CopilotCodeConfig(
        model="claude-sonnet-4.6",
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        enabled_skills=(),
        extra_skill_directories=[str(tmp_path / "skills")],
    )
    client = CopilotCodeClient(config)
    kwargs = client._session_kwargs()

    asm = client.assembler
    assert asm is not None
    full = asm.render()
    cacheable = asm.render(cacheable_only=True)

    assert len(cacheable) < len(full)


def test_system_message_uses_cacheable_only(tmp_path: Path) -> None:
    """The system_message in session kwargs should match render(cacheable_only=True)."""
    config = CopilotCodeConfig(
        model="claude-sonnet-4.6",
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
    )
    client = CopilotCodeClient(config)
    kwargs = client._session_kwargs()

    assert client.assembler is not None
    expected = client.assembler.render(cacheable_only=True)
    assert kwargs["system_message"]["content"] == expected


def test_assembler_passed_to_hooks(tmp_path: Path) -> None:
    """Hooks should receive the assembler and inject dynamic content."""
    from copilotcode_sdk.prompt_compiler import PromptAssembler

    config = CopilotCodeConfig(
        model="claude-sonnet-4.6",
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
    )
    client = CopilotCodeClient(config)
    kwargs = client._session_kwargs()
    hooks = kwargs["hooks"]

    # Session start should work without error and include context
    result = hooks["on_session_start"]({}, {})
    assert "additionalContext" in result


def test_mcp_servers_in_session_start_context(tmp_path: Path) -> None:
    """MCP server configuration should flow through to session start context."""
    config = CopilotCodeConfig(
        model="claude-sonnet-4.6",
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        mcp_servers=[
            {"name": "ClientMCP", "description": "Client MCP server.",
             "tools": [{"name": "client_tool", "description": "A tool."}]},
        ],
    )
    client = CopilotCodeClient(config)
    kwargs = client._session_kwargs()
    hooks = kwargs["hooks"]

    result = hooks["on_session_start"]({}, {})

    assert "ClientMCP" in result["additionalContext"]


# ---------------------------------------------------------------------------
# Wave 4.4: compact_for_handoff
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compact_for_handoff_persists_artifact(tmp_path: Path) -> None:
    """compact_for_handoff should write summary to <memory_home>/compaction/."""
    from unittest.mock import AsyncMock, MagicMock

    mem_dir = tmp_path / ".mem"
    store = MemoryStore(tmp_path, mem_dir)
    store.ensure()

    # Build a mock SDK session
    mock_session = AsyncMock()
    mock_session.session_id = "sess-42"
    mock_session.workspace_path = str(tmp_path)
    mock_session.get_messages = AsyncMock(return_value=[])
    mock_session.send_and_wait = AsyncMock(return_value=None)
    mock_session.destroy = AsyncMock()

    # Build a mock maintenance session
    maintenance_session = AsyncMock()
    maintenance_session.send_and_wait = AsyncMock(return_value=None)
    maintenance_session.get_messages = AsyncMock(return_value=[
        {"role": "assistant", "content": "<analysis>Reasoning here.</analysis>\n<summary>Compact summary.</summary>"},
    ])
    maintenance_session.destroy = AsyncMock()

    # Mock copilot client that creates our maintenance session
    mock_copilot_client = AsyncMock()
    mock_copilot_client.create_session = AsyncMock(return_value=maintenance_session)

    from copilotcode_sdk.client import CopilotCodeSession
    wrapped = CopilotCodeSession(
        mock_session, store,
        copilot_client=mock_copilot_client,
    )

    result = await wrapped.compact_for_handoff()

    assert result.analysis == "Reasoning here."
    assert result.summary == "Compact summary."

    artifact = store.memory_dir / "compaction" / "sess-42.md"
    assert artifact.exists()
    assert artifact.read_text(encoding="utf-8") == "Compact summary."


@pytest.mark.asyncio
async def test_compact_for_handoff_accepts_session_event_like_messages(tmp_path: Path) -> None:
    from unittest.mock import AsyncMock

    class EventLike:
        def __init__(self, payload: dict[str, object]) -> None:
            self._payload = payload

        def to_dict(self) -> dict[str, object]:
            return self._payload

    mem_dir = tmp_path / ".mem"
    store = MemoryStore(tmp_path, mem_dir)
    store.ensure()

    parent_session = AsyncMock()
    parent_session.session_id = "sess-live"
    parent_session.workspace_path = str(tmp_path)
    parent_session.get_messages = AsyncMock(return_value=[
        EventLike({"role": "user", "content": "Please summarize this live transcript."}),
    ])
    parent_session.destroy = AsyncMock()

    maintenance_session = AsyncMock()
    maintenance_session.send_and_wait = AsyncMock(return_value=None)
    maintenance_session.get_messages = AsyncMock(return_value=[
        EventLike({
            "role": "assistant",
            "content": "<analysis>Checked the transcript.</analysis>\n<summary>Live-safe summary.</summary>",
        }),
    ])
    maintenance_session.destroy = AsyncMock()

    mock_copilot_client = AsyncMock()
    mock_copilot_client.create_session = AsyncMock(return_value=maintenance_session)

    wrapped = CopilotCodeSession(
        parent_session,
        store,
        copilot_client=mock_copilot_client,
    )

    result = await wrapped.compact_for_handoff()

    assert result.summary == "Live-safe summary."
    sent_prompt = maintenance_session.send_and_wait.await_args.args[0]
    assert "[user]: Please summarize this live transcript." in sent_prompt


# ---------------------------------------------------------------------------
# Wave 6: Session state and cost wiring
# ---------------------------------------------------------------------------


class FakeSessionWithUsage(FakeSession):
    """FakeSession that returns usage data from send_and_wait."""

    async def send_and_wait(
        self,
        prompt: str,
        *,
        attachments=None,
        mode=None,
        timeout=60.0,
    ) -> dict[str, object]:
        self.sent.append((prompt, attachments, mode))
        return {
            "status": "ok",
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 500,
                "cache_read_input_tokens": 200,
                "cache_creation_input_tokens": 100,
            },
        }


def test_session_state_tracks_turns(tmp_path: Path) -> None:
    """Session state should track turn count across send_and_wait calls."""
    from copilotcode_sdk.client import CopilotCodeSession
    from copilotcode_sdk.session_state import SessionStatus

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store)

    assert session.state.turn_count == 0
    assert session.state.status == SessionStatus.idle

    asyncio.run(session.send_and_wait("hello"))
    assert session.state.turn_count == 1
    assert session.state.status == SessionStatus.idle  # back to idle after turn

    asyncio.run(session.send_and_wait("world"))
    assert session.state.turn_count == 2


def test_session_state_accumulates_usage(tmp_path: Path) -> None:
    """Session state should accumulate token usage from results with usage data."""
    from copilotcode_sdk.client import CopilotCodeSession

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSessionWithUsage()
    session = CopilotCodeSession(fake, store, model="claude-sonnet-4.6")

    asyncio.run(session.send_and_wait("hello"))

    assert session.state.total_input_tokens == 1000
    assert session.state.total_output_tokens == 500
    assert session.state.total_tokens == 1500

    asyncio.run(session.send_and_wait("more"))
    assert session.state.total_input_tokens == 2000
    assert session.state.total_output_tokens == 1000


def test_session_cost_accumulates(tmp_path: Path) -> None:
    """Cumulative cost should grow with each send_and_wait."""
    from copilotcode_sdk.client import CopilotCodeSession

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSessionWithUsage()
    session = CopilotCodeSession(fake, store, model="claude-sonnet-4.6")

    assert session.cumulative_cost.total == 0.0

    asyncio.run(session.send_and_wait("hello"))
    cost1 = session.cumulative_cost.total
    assert cost1 > 0  # Should have calculated some cost

    asyncio.run(session.send_and_wait("world"))
    cost2 = session.cumulative_cost.total
    assert cost2 > cost1  # Cost should grow


def test_session_cost_zero_without_model(tmp_path: Path) -> None:
    """Without a model configured, cost should remain zero."""
    from copilotcode_sdk.client import CopilotCodeSession

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSessionWithUsage()
    session = CopilotCodeSession(fake, store)  # no model

    asyncio.run(session.send_and_wait("hello"))
    assert session.cumulative_cost.total == 0.0


def test_session_state_survives_error(tmp_path: Path) -> None:
    """Session state should return to idle even if send_and_wait raises."""
    from copilotcode_sdk.client import CopilotCodeSession
    from copilotcode_sdk.session_state import SessionStatus

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = ExplodingSession()
    session = CopilotCodeSession(fake, store)

    with pytest.raises(RuntimeError):
        asyncio.run(session.send_and_wait("boom"))

    assert session.state.status == SessionStatus.idle
    assert session.state.turn_count == 1  # Turn was started before error


# ---------------------------------------------------------------------------
# Wave 6: Model switching
# ---------------------------------------------------------------------------


def test_model_switching(tmp_path: Path) -> None:
    from copilotcode_sdk.client import CopilotCodeSession

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store, model="claude-opus-4.6")

    assert session.active_model == "claude-opus-4.6"

    prev = session.switch_model("claude-haiku-4.5")
    assert prev is None
    assert session.active_model == "claude-haiku-4.5"

    prev = session.switch_model(None)
    assert prev == "claude-haiku-4.5"


def test_toggle_fast_mode(tmp_path: Path) -> None:
    from copilotcode_sdk.client import CopilotCodeSession

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store, model="claude-opus-4.6")

    # Toggle on
    is_fast = session.toggle_fast_mode("claude-haiku-4.5")
    assert is_fast is True
    assert session.active_model == "claude-haiku-4.5"

    # Toggle off
    is_fast = session.toggle_fast_mode("claude-haiku-4.5")
    assert is_fast is False


def test_stream_fallback(tmp_path: Path) -> None:
    """When underlying session has no stream(), stream() falls back to send_and_wait."""
    from copilotcode_sdk.client import CopilotCodeSession

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store)

    chunks = []

    async def _collect():
        async for chunk in session.stream("hello"):
            chunks.append(chunk)

    asyncio.run(_collect())
    assert len(chunks) == 1
    assert chunks[0]["type"] == "message_stop"


# ---------------------------------------------------------------------------
# Session task convenience methods (Wave 3.2 items 12-13)
# ---------------------------------------------------------------------------


def test_session_task_store_property(tmp_path: Path) -> None:
    from copilotcode_sdk.client import CopilotCodeSession
    from copilotcode_sdk.tasks import TaskStore

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    ts = TaskStore(task_list_id="list-1", task_root=tmp_path / "tasks")

    session = CopilotCodeSession(fake, store, task_store=ts)
    assert session.task_store is ts
    assert session.task_list_id == "list-1"


def test_session_task_store_none_by_default(tmp_path: Path) -> None:
    from copilotcode_sdk.client import CopilotCodeSession

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    session = CopilotCodeSession(FakeSession(), store)
    assert session.task_store is None
    assert session.task_list_id is None
    assert session.list_tasks() == []
    assert session.get_task(1) is None
    assert session.get_task_output(1) is None


def test_session_create_and_get_task(tmp_path: Path) -> None:
    from copilotcode_sdk.client import CopilotCodeSession
    from copilotcode_sdk.tasks import TaskStore

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    ts = TaskStore(task_list_id="sess-1", task_root=tmp_path / "tasks")

    session = CopilotCodeSession(FakeSession(), store, task_store=ts)
    task = session.create_task("Build feature", description="The big one")
    assert task.subject == "Build feature"
    assert task.id == 1

    fetched = session.get_task(1)
    assert fetched is not None
    assert fetched.subject == "Build feature"


def test_session_update_task(tmp_path: Path) -> None:
    from copilotcode_sdk.client import CopilotCodeSession
    from copilotcode_sdk.tasks import TaskStore, TaskStatus

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    ts = TaskStore(task_list_id="sess-2", task_root=tmp_path / "tasks")

    session = CopilotCodeSession(FakeSession(), store, task_store=ts)
    session.create_task("Test task")
    updated = session.update_task(1, status="completed")
    assert updated is not None
    assert updated.status == TaskStatus.completed


def test_session_list_tasks(tmp_path: Path) -> None:
    from copilotcode_sdk.client import CopilotCodeSession
    from copilotcode_sdk.tasks import TaskStore

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    ts = TaskStore(task_list_id="sess-3", task_root=tmp_path / "tasks")

    session = CopilotCodeSession(FakeSession(), store, task_store=ts)
    session.create_task("Task A")
    session.create_task("Task B")
    tasks = session.list_tasks()
    assert len(tasks) == 2


def test_session_get_task_output(tmp_path: Path) -> None:
    from copilotcode_sdk.client import CopilotCodeSession
    from copilotcode_sdk.tasks import TaskStore

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    ts = TaskStore(task_list_id="sess-4", task_root=tmp_path / "tasks")

    session = CopilotCodeSession(FakeSession(), store, task_store=ts)
    session.create_task("Output task")
    ts.write_task_output(1, "Result data")
    assert session.get_task_output(1) == "Result data"


def test_session_create_task_raises_without_store(tmp_path: Path) -> None:
    from copilotcode_sdk.client import CopilotCodeSession

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    session = CopilotCodeSession(FakeSession(), store)
    with pytest.raises(RuntimeError, match="No task store"):
        session.create_task("Should fail")


def test_session_update_task_raises_without_store(tmp_path: Path) -> None:
    from copilotcode_sdk.client import CopilotCodeSession

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    session = CopilotCodeSession(FakeSession(), store)
    with pytest.raises(RuntimeError, match="No task store"):
        session.update_task(1, status="completed")


def test_resolve_task_store_per_session(tmp_path: Path) -> None:
    """CopilotCodeClient._resolve_task_store creates per-list stores."""
    config = CopilotCodeConfig(
        model="claude-sonnet-4.6",
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        enable_tasks_v2=True,
    )
    client = CopilotCodeClient(config)
    ts = client._resolve_task_store("my-session")
    assert ts is not None
    assert ts.task_list_id == "my-session"


def test_resolve_task_store_none_when_disabled(tmp_path: Path) -> None:
    config = CopilotCodeConfig(
        model="claude-sonnet-4.6",
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        enable_tasks_v2=False,
    )
    client = CopilotCodeClient(config)
    assert client._resolve_task_store("anything") is None


def test_resolve_task_store_fallback_to_shared(tmp_path: Path) -> None:
    config = CopilotCodeConfig(
        model="claude-sonnet-4.6",
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        enable_tasks_v2=True,
    )
    client = CopilotCodeClient(config)
    ts = client._resolve_task_store(None)
    assert ts is client.task_store


# ---------------------------------------------------------------------------
# Gap 2.2: Resume restores cacheable prefix
# ---------------------------------------------------------------------------


def test_resume_session_sets_cacheable_prefix(tmp_path: Path) -> None:
    """Resumed sessions should have subagent_context (set_cacheable_prefix called)."""
    from copilotcode_sdk.client import CopilotCodeSession

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    config = CopilotCodeConfig(model="claude-sonnet-4.6", working_directory=tmp_path, memory_root=tmp_path / ".mem")
    fake_client = FakeCopilotClient()
    client = CopilotCodeClient(config, copilot_client=fake_client)

    session = asyncio.run(client.resume_session("sess-123"))
    # The assembler is rebuilt in _session_kwargs, then set_cacheable_prefix is called
    assert session.subagent_context is not None
    assert session.subagent_context.cacheable_prefix != ""


# ---------------------------------------------------------------------------
# Gap 2.3: fork_child returns ChildSession with constraints
# ---------------------------------------------------------------------------


def test_fork_child_returns_child_session(tmp_path: Path) -> None:
    from copilotcode_sdk.client import CopilotCodeSession
    from copilotcode_sdk.subagent import EnforcedChildSession, SubagentSpec

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    fake_client = FakeCopilotClient()
    session = CopilotCodeSession(fake, store, copilot_client=fake_client)
    session.set_cacheable_prefix("test prefix")

    spec = SubagentSpec(
        role="researcher",
        system_prompt_suffix="Search only.",
        tools=("read", "grep"),
        max_turns=5,
        timeout_seconds=15.0,
    )
    child = asyncio.run(session.fork_child(spec))

    assert isinstance(child, EnforcedChildSession)
    assert child.spec.timeout_seconds == 15.0
    assert child.spec.max_turns == 5
    assert child.session is not None
    assert child.session_id is not None


def test_fork_child_passes_tool_allowlist(tmp_path: Path) -> None:
    from copilotcode_sdk.client import CopilotCodeSession
    from copilotcode_sdk.subagent import SubagentSpec

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    fake_client = FakeCopilotClient()
    session = CopilotCodeSession(fake, store, copilot_client=fake_client)
    session.set_cacheable_prefix("prefix")

    spec = SubagentSpec(role="r", system_prompt_suffix="s", tools=("read", "write"))
    asyncio.run(session.fork_child(spec))

    # Check that create_session was called with available_tools
    assert len(fake_client.create_calls) == 1
    call_kwargs = fake_client.create_calls[0]
    assert "available_tools" in call_kwargs
    assert "read" in call_kwargs["available_tools"]
    assert "write" in call_kwargs["available_tools"]


def test_fork_child_no_tools_when_spec_empty(tmp_path: Path) -> None:
    from copilotcode_sdk.client import CopilotCodeSession
    from copilotcode_sdk.subagent import SubagentSpec

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    fake_client = FakeCopilotClient()
    session = CopilotCodeSession(fake, store, copilot_client=fake_client)
    session.set_cacheable_prefix("prefix")

    spec = SubagentSpec(role="r", system_prompt_suffix="s")  # no tools
    asyncio.run(session.fork_child(spec))

    call_kwargs = fake_client.create_calls[0]
    assert "tools" not in call_kwargs


# ── Wave 6b: Model switching events ──────────────────────────────────────


def test_switch_model_emits_event(tmp_path: Path) -> None:
    from copilotcode_sdk.events import EventType

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store, model="claude-sonnet-4.6")

    events: list = []
    session.event_bus.subscribe(lambda e: events.append(e))

    session.switch_model("claude-opus-4.6")

    model_events = [e for e in events if e.type == EventType.model_switched]
    assert len(model_events) == 1
    assert model_events[0].data["from"] == "claude-sonnet-4.6"
    assert model_events[0].data["to"] == "claude-opus-4.6"


def test_switch_model_none_reverts_to_default(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store, model="claude-sonnet-4.6")

    session.switch_model("claude-opus-4.6")
    session.switch_model(None)

    assert session._model_override is None


def test_switch_model_multiple_times_emits_multiple_events(tmp_path: Path) -> None:
    from copilotcode_sdk.events import EventType

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store, model="claude-sonnet-4.6")

    events: list = []
    session.event_bus.subscribe(lambda e: events.append(e))

    session.switch_model("claude-opus-4.6")
    session.switch_model("claude-haiku-4.5")

    model_events = [e for e in events if e.type == EventType.model_switched]
    assert len(model_events) == 2
    assert model_events[0].data["to"] == "claude-opus-4.6"
    assert model_events[1].data["from"] == "claude-opus-4.6"
    assert model_events[1].data["to"] == "claude-haiku-4.5"


def test_switch_model_none_does_not_emit_event(tmp_path: Path) -> None:
    from copilotcode_sdk.events import EventType

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store, model="claude-sonnet-4.6")

    events: list = []
    session.event_bus.subscribe(lambda e: events.append(e))

    session.switch_model(None)

    model_events = [e for e in events if e.type == EventType.model_switched]
    assert len(model_events) == 0


def test_toggle_fast_mode_with_none(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store, model="claude-sonnet-4.6")

    result = session.toggle_fast_mode(None)
    assert result is False
    assert session._model_override is None


# ── Wave 6c: Cost event logging ──────────────────────────────────────────


def test_cost_event_emitted_on_send(tmp_path: Path) -> None:
    from copilotcode_sdk.events import EventType

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store, model="claude-sonnet-4.6")

    events: list = []
    session.event_bus.subscribe(lambda e: events.append(e))

    # Simulate SDK usage event
    class FakeData:
        input_tokens = 1000
        output_tokens = 500
        cache_read_tokens = 0
        cache_write_tokens = 0
        model = "claude-sonnet-4.6"

    class FakeEvent:
        type = "assistant.usage"
        data = FakeData()

    session._on_sdk_event(FakeEvent())

    cost_events = [e for e in events if e.type == EventType.cost_accumulated]
    assert len(cost_events) == 1
    assert cost_events[0].data["model"] == "claude-sonnet-4.6"
    assert cost_events[0].data["turn_cost"] > 0
    assert cost_events[0].data["total_cost"] > 0


def test_cost_event_total_grows(tmp_path: Path) -> None:
    from copilotcode_sdk.events import EventType

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store, model="claude-sonnet-4.6")

    events: list = []
    session.event_bus.subscribe(lambda e: events.append(e))

    class FakeData:
        input_tokens = 500
        output_tokens = 200
        cache_read_tokens = 0
        cache_write_tokens = 0
        model = "claude-sonnet-4.6"

    class FakeEvent:
        type = "assistant.usage"
        data = FakeData()

    session._on_sdk_event(FakeEvent())
    session._on_sdk_event(FakeEvent())

    cost_events = [e for e in events if e.type == EventType.cost_accumulated]
    assert len(cost_events) == 2
    assert cost_events[1].data["total_cost"] > cost_events[0].data["total_cost"]


def test_cost_event_not_emitted_without_model(tmp_path: Path) -> None:
    from copilotcode_sdk.events import EventType

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store, model=None)

    events: list = []
    session.event_bus.subscribe(lambda e: events.append(e))

    class FakeData:
        input_tokens = 100
        output_tokens = 50
        cache_read_tokens = 0
        cache_write_tokens = 0
        model = None

    class FakeEvent:
        type = "assistant.usage"
        data = FakeData()

    session._on_sdk_event(FakeEvent())

    cost_events = [e for e in events if e.type == EventType.cost_accumulated]
    assert len(cost_events) == 0


def test_cost_uses_active_model_pricing(tmp_path: Path) -> None:
    from copilotcode_sdk.events import EventType

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store, model="claude-opus-4.6")

    events: list = []
    session.event_bus.subscribe(lambda e: events.append(e))

    class OpusData:
        input_tokens = 1000
        output_tokens = 500
        cache_read_tokens = 0
        cache_write_tokens = 0
        model = "claude-opus-4.6"

    class OpusEvent:
        type = "assistant.usage"
        data = OpusData()

    session._on_sdk_event(OpusEvent())
    opus_cost = events[-1].data["turn_cost"]

    session.switch_model("claude-haiku-4.5")

    class HaikuData:
        input_tokens = 1000
        output_tokens = 500
        cache_read_tokens = 0
        cache_write_tokens = 0
        model = "claude-haiku-4.5"

    class HaikuEvent:
        type = "assistant.usage"
        data = HaikuData()

    session._on_sdk_event(HaikuEvent())
    haiku_cost = [e for e in events if e.type == EventType.cost_accumulated][-1].data["turn_cost"]

    assert haiku_cost < opus_cost


def test_cost_accumulation_correct_values(tmp_path: Path) -> None:
    from copilotcode_sdk.model_cost import calculate_cost

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store, model="claude-sonnet-4.6")

    class FakeData:
        input_tokens = 1000
        output_tokens = 500
        cache_read_tokens = 100
        cache_write_tokens = 50
        model = "claude-sonnet-4.6"

    class FakeEvent:
        type = "assistant.usage"
        data = FakeData()

    session._on_sdk_event(FakeEvent())

    expected = calculate_cost(
        "claude-sonnet-4.6",
        input_tokens=1000,
        output_tokens=500,
        cache_read_tokens=100,
        cache_creation_tokens=50,
    )
    assert abs(session._cumulative_cost.total - expected.total) < 1e-10


def test_turn_completed_event_emitted(tmp_path: Path) -> None:
    from copilotcode_sdk.events import EventType

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store, model="claude-sonnet-4.6")

    events: list = []
    session.event_bus.subscribe(lambda e: events.append(e))

    # Simulate usage event to record tokens
    class FakeData:
        input_tokens = 1000
        output_tokens = 500
        cache_read_tokens = 0
        cache_write_tokens = 0
        model = "claude-sonnet-4.6"

    class FakeEvent:
        type = "assistant.usage"
        data = FakeData()

    session._on_sdk_event(FakeEvent())

    # Now call _emit_turn_completed
    session._state._turn_count = 1
    session._event_bus.emit(_turn_completed_event(
        turn_count=1,
        input_tokens=1000,
        output_tokens=500,
    ))

    turn_events = [e for e in events if e.type == EventType.turn_completed]
    assert len(turn_events) == 1
    assert turn_events[0].data["turn_count"] == 1
    assert turn_events[0].data["input_tokens"] == 1000
    assert turn_events[0].data["output_tokens"] == 500


# ── Wave 6d: SDK event path token tracking ───────────────────────────────


def test_sdk_usage_event_tracks_tokens(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store, model="claude-sonnet-4.6")

    class FakeData:
        input_tokens = 500
        output_tokens = 200
        cache_read_tokens = 50
        cache_write_tokens = 25
        model = "claude-sonnet-4.6"

    class FakeEvent:
        type = "assistant.usage"
        data = FakeData()

    session._on_sdk_event(FakeEvent())

    assert session._state.total_input_tokens == 500
    assert session._state.total_output_tokens == 200


def test_sdk_usage_event_emits_cost_event(tmp_path: Path) -> None:
    from copilotcode_sdk.events import EventType

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store, model="claude-sonnet-4.6")

    events: list = []
    session.event_bus.subscribe(lambda e: events.append(e))

    class FakeData:
        input_tokens = 500
        output_tokens = 200
        cache_read_tokens = 50
        cache_write_tokens = 25
        model = "claude-sonnet-4.6"

    class FakeEvent:
        type = "assistant.usage"
        data = FakeData()

    session._on_sdk_event(FakeEvent())

    cost_events = [e for e in events if e.type == EventType.cost_accumulated]
    assert len(cost_events) == 1
    assert cost_events[0].data["turn_cost"] > 0


def test_sdk_usage_event_accumulates_across_calls(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store, model="claude-sonnet-4.6")

    class FakeData:
        input_tokens = 500
        output_tokens = 200
        cache_read_tokens = 50
        cache_write_tokens = 25
        model = "claude-sonnet-4.6"

    class FakeEvent:
        type = "assistant.usage"
        data = FakeData()

    session._on_sdk_event(FakeEvent())
    session._on_sdk_event(FakeEvent())

    assert session._state.total_input_tokens == 1000
    assert session._state.total_output_tokens == 400
    assert session._cumulative_cost.total > 0


def test_sdk_usage_event_ignores_non_usage(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store, model="claude-sonnet-4.6")

    class FakeEvent:
        type = "other.event"
        data = None

    session._on_sdk_event(FakeEvent())

    assert session._state.total_input_tokens == 0
    assert session._state.total_output_tokens == 0


def test_sdk_usage_event_with_enum_type(tmp_path: Path) -> None:
    from enum import Enum

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store, model="claude-sonnet-4.6")

    class FakeEventType(Enum):
        USAGE = "assistant.usage"

    class FakeData:
        input_tokens = 300
        output_tokens = 100
        cache_read_tokens = 0
        cache_write_tokens = 0
        model = "claude-sonnet-4.6"

    class FakeEvent:
        type = FakeEventType.USAGE
        data = FakeData()

    session._on_sdk_event(FakeEvent())

    assert session._state.total_input_tokens == 300
    assert session._state.total_output_tokens == 100


# ── Wave 7: Cost event carries token counts and source lineage ───────────


def test_cost_event_includes_token_counts(tmp_path: Path) -> None:
    from copilotcode_sdk.events import EventType

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store, model="claude-sonnet-4.6")

    events: list = []
    session.event_bus.subscribe(lambda e: events.append(e))

    class FakeData:
        input_tokens = 800
        output_tokens = 300
        cache_read_tokens = 100
        cache_write_tokens = 50
        model = "claude-sonnet-4.6"

    class FakeEvent:
        type = "assistant.usage"
        data = FakeData()

    session._on_sdk_event(FakeEvent())

    cost_events = [e for e in events if e.type == EventType.cost_accumulated]
    assert len(cost_events) == 1
    d = cost_events[0].data
    assert d["input_tokens"] == 800
    assert d["output_tokens"] == 300
    assert d["cache_read_tokens"] == 100
    assert d["cache_write_tokens"] == 50


def test_cost_event_includes_source(tmp_path: Path) -> None:
    from copilotcode_sdk.events import EventType

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store, model="claude-sonnet-4.6", source="verifier")

    events: list = []
    session.event_bus.subscribe(lambda e: events.append(e))

    class FakeData:
        input_tokens = 100
        output_tokens = 50
        cache_read_tokens = 0
        cache_write_tokens = 0
        model = "claude-sonnet-4.6"

    class FakeEvent:
        type = "assistant.usage"
        data = FakeData()

    session._on_sdk_event(FakeEvent())

    cost_events = [e for e in events if e.type == EventType.cost_accumulated]
    assert cost_events[0].data["source"] == "verifier"


def test_cost_event_default_source_is_main(tmp_path: Path) -> None:
    from copilotcode_sdk.events import EventType

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store, model="claude-sonnet-4.6")

    events: list = []
    session.event_bus.subscribe(lambda e: events.append(e))

    class FakeData:
        input_tokens = 100
        output_tokens = 50
        cache_read_tokens = 0
        cache_write_tokens = 0
        model = "claude-sonnet-4.6"

    class FakeEvent:
        type = "assistant.usage"
        data = FakeData()

    session._on_sdk_event(FakeEvent())

    cost_events = [e for e in events if e.type == EventType.cost_accumulated]
    assert cost_events[0].data["source"] == "main"


def test_cost_logger_fires_on_event(tmp_path: Path, caplog) -> None:
    """The auto-subscribed cost logger should write to copilotcode.cost logger."""
    import logging

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store, model="claude-sonnet-4.6")

    class FakeData:
        input_tokens = 500
        output_tokens = 200
        cache_read_tokens = 0
        cache_write_tokens = 0
        model = "claude-sonnet-4.6"

    class FakeEvent:
        type = "assistant.usage"
        data = FakeData()

    with caplog.at_level(logging.INFO, logger="copilotcode.cost"):
        session._on_sdk_event(FakeEvent())

    assert any("model=claude-sonnet-4.6" in r.message for r in caplog.records)
    assert any("in=500" in r.message for r in caplog.records)
    assert any("out=200" in r.message for r in caplog.records)


def test_accumulate_usage_includes_source(tmp_path: Path) -> None:
    """The fallback _accumulate_usage path should also include source and tokens."""
    from copilotcode_sdk.events import EventType

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store, model="claude-sonnet-4.6", source="exercise")

    events: list = []
    session.event_bus.subscribe(lambda e: events.append(e))

    # Simulate a send_and_wait result with usage data
    result = {
        "usage": {
            "input_tokens": 400,
            "output_tokens": 150,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
        }
    }
    session._accumulate_usage(result)

    cost_events = [e for e in events if e.type == EventType.cost_accumulated]
    assert len(cost_events) == 1
    d = cost_events[0].data
    assert d["source"] == "exercise"
    assert d["input_tokens"] == 400
    assert d["output_tokens"] == 150


# ── SDK event visibility: compaction & context window ─────────────────


def test_sdk_compaction_start_event(tmp_path: Path) -> None:
    """session.compaction_start events should emit compaction_started on the bus."""
    from copilotcode_sdk.events import EventType

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store, model="claude-sonnet-4.6")

    events: list = []
    session.event_bus.subscribe(lambda e: events.append(e))

    class FakeData:
        pre_compaction_tokens = 50000
        system_tokens = 5000
        tool_definitions_tokens = 3000
        pre_compaction_messages_length = 42

    class FakeEvent:
        type = "session.compaction_start"
        data = FakeData()

    session._on_sdk_event(FakeEvent())

    compaction_events = [e for e in events if e.type == EventType.compaction_started]
    assert len(compaction_events) == 1
    d = compaction_events[0].data
    assert d["pre_compaction_tokens"] == 50000
    assert d["system_tokens"] == 5000
    assert d["tool_tokens"] == 3000
    assert d["message_count"] == 42


def test_sdk_compaction_complete_event(tmp_path: Path) -> None:
    """session.compaction_complete events should emit compaction_completed on the bus."""
    from copilotcode_sdk.events import EventType

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store, model="claude-sonnet-4.6")

    events: list = []
    session.event_bus.subscribe(lambda e: events.append(e))

    class FakeTokensUsed:
        input = 1000
        output = 500
        cached_input = 200

    class FakeData:
        pre_compaction_tokens = 50000
        post_compaction_tokens = 20000
        tokens_removed = 30000
        messages_removed = 15
        success = True
        summary_content = "Session compacted successfully."
        compaction_tokens_used = FakeTokensUsed()

    class FakeEvent:
        type = "session.compaction_complete"
        data = FakeData()

    session._on_sdk_event(FakeEvent())

    compaction_events = [e for e in events if e.type == EventType.compaction_completed]
    assert len(compaction_events) == 1
    d = compaction_events[0].data
    assert d["pre_compaction_tokens"] == 50000
    assert d["post_compaction_tokens"] == 20000
    assert d["tokens_removed"] == 30000
    assert d["messages_removed"] == 15
    assert d["success"] is True
    assert d["summary"] == "Session compacted successfully."
    assert d["compaction_tokens_used"]["input"] == 1000
    assert d["compaction_tokens_used"]["output"] == 500
    assert d["compaction_tokens_used"]["cached_input"] == 200


def test_sdk_context_window_event(tmp_path: Path) -> None:
    """session.info events with context_window info_type should emit context_window_updated."""
    from copilotcode_sdk.events import EventType

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store, model="claude-sonnet-4.6")

    events: list = []
    session.event_bus.subscribe(lambda e: events.append(e))

    class FakeData:
        info_type = "context_window"
        current_tokens = 80000
        token_limit = 200000

    class FakeEvent:
        type = "session.info"
        data = FakeData()

    session._on_sdk_event(FakeEvent())

    ctx_events = [e for e in events if e.type == EventType.context_window_updated]
    assert len(ctx_events) == 1
    d = ctx_events[0].data
    assert d["current_tokens"] == 80000
    assert d["token_limit"] == 200000
    assert abs(d["utilization_ratio"] - 0.4) < 0.01


def test_sdk_session_info_non_context_ignored(tmp_path: Path) -> None:
    """session.info events with non-context info_type should not emit context events."""
    from copilotcode_sdk.events import EventType

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store, model="claude-sonnet-4.6")

    events: list = []
    session.event_bus.subscribe(lambda e: events.append(e))

    class FakeData:
        info_type = "notification"
        current_tokens = 0
        token_limit = 0

    class FakeEvent:
        type = "session.info"
        data = FakeData()

    session._on_sdk_event(FakeEvent())

    ctx_events = [e for e in events if e.type == EventType.context_window_updated]
    assert len(ctx_events) == 0


def test_sdk_usage_event_still_works_after_refactor(tmp_path: Path) -> None:
    """assistant.usage events should continue to work after the _on_sdk_event refactor."""
    from copilotcode_sdk.events import EventType

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store, model="claude-sonnet-4.6")

    events: list = []
    session.event_bus.subscribe(lambda e: events.append(e))

    class FakeData:
        input_tokens = 500
        output_tokens = 200
        cache_read_tokens = 50
        cache_write_tokens = 10
        model = "claude-sonnet-4.6"

    class FakeEvent:
        type = "assistant.usage"
        data = FakeData()

    session._on_sdk_event(FakeEvent())

    cost_events = [e for e in events if e.type == EventType.cost_accumulated]
    assert len(cost_events) == 1
    assert session._state.total_input_tokens == 500
    assert session._state.total_output_tokens == 200


# ── Child hook reliability ──────────────────────────────────────────────


def _fork_child_and_get_hooks(tmp_path: Path) -> tuple[dict, "CopilotCodeSession"]:
    """Helper: fork a child and return (hooks_dict, parent_session)."""
    from copilotcode_sdk.client import CopilotCodeSession
    from copilotcode_sdk.subagent import SubagentSpec

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    fake_client = FakeCopilotClient()
    session = CopilotCodeSession(fake, store, copilot_client=fake_client)
    session.set_cacheable_prefix("prefix")

    spec = SubagentSpec(role="worker", system_prompt_suffix="do stuff")
    asyncio.run(session.fork_child(spec))

    call_kwargs = fake_client.create_calls[0]
    return call_kwargs["hooks"], session


def test_child_pre_tool_hook_detects_repeated_identical_calls(tmp_path: Path) -> None:
    """Pre-tool hook returns additionalContext after 4 identical calls."""
    hooks, _session = _fork_child_and_get_hooks(tmp_path)
    pre_hook = hooks["on_pre_tool_use"]

    input_data = {"toolName": "Write", "toolArgs": {"path": "/tmp/x", "content": "hi"}}
    env: dict = {}

    # First 3 calls should return None
    for _ in range(3):
        result = pre_hook(input_data, env)
        assert result is None

    # 4th identical call triggers the warning
    result = pre_hook(input_data, env)
    assert result is not None
    assert "REPEATED CALL DETECTED" in result["additionalContext"]
    assert "Write" in result["additionalContext"]

    # Counter resets — next call should be None again
    result = pre_hook(input_data, env)
    assert result is None


def test_child_pre_tool_hook_different_args_no_trigger(tmp_path: Path) -> None:
    """Pre-tool hook does NOT trigger when args differ each time."""
    hooks, _session = _fork_child_and_get_hooks(tmp_path)
    pre_hook = hooks["on_pre_tool_use"]

    for i in range(6):
        input_data = {"toolName": "Write", "toolArgs": {"path": f"/tmp/x{i}", "content": "hi"}}
        result = pre_hook(input_data, {})
        assert result is None


def test_child_pre_tool_hook_emits_tool_called_event(tmp_path: Path) -> None:
    """Pre-tool hook emits tool_called event on parent's event bus."""
    from copilotcode_sdk.events import EventType

    hooks, session = _fork_child_and_get_hooks(tmp_path)
    pre_hook = hooks["on_pre_tool_use"]

    events_received: list = []
    session.event_bus.subscribe(lambda e: events_received.append(e))

    pre_hook({"toolName": "Bash", "toolArgs": {"command": "ls"}}, {})

    tool_events = [e for e in events_received if e.type == EventType.tool_called]
    assert len(tool_events) == 1
    assert tool_events[0].data["tool_name"] == "Bash"


def test_child_post_tool_hook_detects_repeated_errors(tmp_path: Path) -> None:
    """Post-tool hook returns additionalContext after 3 identical errors."""
    hooks, _session = _fork_child_and_get_hooks(tmp_path)
    post_hook = hooks["on_post_tool_use"]

    input_data = {
        "toolName": "Write",
        "toolResult": {"error": "permission denied"},
    }

    # First 2 failures return None
    for _ in range(2):
        result = post_hook(input_data, {})
        assert result is None

    # 3rd failure triggers the stuck loop warning
    result = post_hook(input_data, {})
    assert result is not None
    assert "STUCK LOOP DETECTED" in result["additionalContext"]
    assert "Write" in result["additionalContext"]


def test_child_post_tool_hook_clears_on_success(tmp_path: Path) -> None:
    """Post-tool hook clears failure count on success."""
    hooks, _session = _fork_child_and_get_hooks(tmp_path)
    post_hook = hooks["on_post_tool_use"]

    fail_data = {"toolName": "Write", "toolResult": {"error": "bad"}}
    ok_data = {"toolName": "Write", "toolResult": {"content": "ok"}}

    # 2 failures
    post_hook(fail_data, {})
    post_hook(fail_data, {})
    # Then a success — should clear the counter
    post_hook(ok_data, {})
    # 2 more failures should NOT trigger (counter was reset)
    assert post_hook(fail_data, {}) is None
    assert post_hook(fail_data, {}) is None


def test_child_event_fallback_logs_stuck_loop(tmp_path: Path, caplog) -> None:
    """Event-based fallback logs a warning after repeated tool failures."""
    import logging
    from copilotcode_sdk.client import CopilotCodeSession
    from copilotcode_sdk.subagent import SubagentSpec

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    fake_client = FakeCopilotClient()
    session = CopilotCodeSession(fake, store, copilot_client=fake_client)
    session.set_cacheable_prefix("prefix")

    spec = SubagentSpec(role="worker", system_prompt_suffix="do stuff")
    asyncio.run(session.fork_child(spec))

    # Extract the on_event proxy from create_session kwargs
    call_kwargs = fake_client.create_calls[0]
    event_proxy = call_kwargs["on_event"]

    # Simulate tool.execution_complete events with errors
    fail_event = {
        "type": "tool.execution_complete",
        "data": {"tool_name": "Write", "error": "permission denied"},
    }

    with caplog.at_level(logging.WARNING, logger="copilotcode.child_hook"):
        for _ in range(3):
            event_proxy(fail_event)

    assert any("stuck-loop detected via events" in r.message.lower() for r in caplog.records)
