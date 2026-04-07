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
    client = CopilotCodeClient(CopilotCodeConfig(working_directory=tmp_path, cli_path=str(explicit_cli)))

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
    session = CopilotCodeSession(fake, store, model="claude-sonnet-4-6")

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
    session = CopilotCodeSession(fake, store, model="claude-sonnet-4-6")

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
    session = CopilotCodeSession(fake, store, model="claude-opus-4-6")

    assert session.active_model == "claude-opus-4-6"

    prev = session.switch_model("claude-haiku-4-5-20251001")
    assert prev is None
    assert session.active_model == "claude-haiku-4-5-20251001"

    prev = session.switch_model(None)
    assert prev == "claude-haiku-4-5-20251001"


def test_toggle_fast_mode(tmp_path: Path) -> None:
    from copilotcode_sdk.client import CopilotCodeSession

    store = MemoryStore(tmp_path, tmp_path / ".mem")
    fake = FakeSession()
    session = CopilotCodeSession(fake, store, model="claude-opus-4-6")

    # Toggle on
    is_fast = session.toggle_fast_mode("claude-haiku-4-5-20251001")
    assert is_fast is True
    assert session.active_model == "claude-haiku-4-5-20251001"

    # Toggle off
    is_fast = session.toggle_fast_mode("claude-haiku-4-5-20251001")
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
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        enable_tasks_v2=False,
    )
    client = CopilotCodeClient(config)
    assert client._resolve_task_store("anything") is None


def test_resolve_task_store_fallback_to_shared(tmp_path: Path) -> None:
    config = CopilotCodeConfig(
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
    config = CopilotCodeConfig(working_directory=tmp_path, memory_root=tmp_path / ".mem")
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
