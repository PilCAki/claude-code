from __future__ import annotations

import asyncio
import json
from pathlib import Path

from copilotcode_sdk import CopilotCodeClient, CopilotCodeConfig, CopilotCodeSession
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

    system_message = kwargs["system_message"]
    assert "Available Skills" in system_message["content"]
    assert "test-skill" in system_message["content"]
