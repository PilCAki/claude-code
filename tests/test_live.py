from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import pytest

from copilotcode_sdk.permissions import build_permission_handler

from .live_support import (
    assert_assistant_contains_token,
    build_live_harness,
    live_session,
    write_live_artifact,
)


pytestmark = pytest.mark.live

LIVE_VALIDATE_E2E_ENV = "COPILOTCODE_RUN_VALIDATE_LIVE_E2E"


def test_live_session_create_and_resume(tmp_path: Path) -> None:
    ready_token = f"ready-{uuid.uuid4().hex[:8]}"
    resumed_token = f"resumed-{uuid.uuid4().hex[:8]}"
    (tmp_path / "ready-token.txt").write_text(ready_token, encoding="utf-8")
    (tmp_path / "resume-token.txt").write_text(resumed_token, encoding="utf-8")

    harness = build_live_harness(tmp_path)
    preflight = harness.client.preflight(require_auth=True)
    assert preflight.ok is True, preflight.to_text()

    session_id = f"copilotcode-live-{uuid.uuid4().hex}"
    ready_prompt = (
        "Read the file `ready-token.txt` in this workspace and reply with only the token it contains."
    )
    resume_prompt = (
        "Read the file `resume-token.txt` in this workspace and reply with only the token it contains."
    )

    with live_session(harness.client, session_id=session_id) as session:
        event = session.send_and_wait(ready_prompt, timeout=180.0)
        assert event is not None
        messages = session.get_messages()
        write_live_artifact(
            harness,
            session_id=session_id,
            label="create",
            prompt=ready_prompt,
            messages=messages,
        )
        assert_assistant_contains_token(messages, ready_token)

    with live_session(harness.client, session_id=session_id, resume=True) as resumed:
        resumed_event = resumed.send_and_wait(resume_prompt, timeout=180.0)
        assert resumed_event is not None
        resumed_messages = resumed.get_messages()
        write_live_artifact(
            harness,
            session_id=session_id,
            label="resume",
            prompt=resume_prompt,
            messages=resumed_messages,
        )
        assert_assistant_contains_token(resumed_messages, resumed_token)


def test_live_instruction_materialization_and_memory_recall(tmp_path: Path) -> None:
    harness = build_live_harness(tmp_path)
    harness.client.materialize_workspace_instructions(overwrite=True)
    token = f"apricot-delta-{uuid.uuid4().hex[:8]}"
    harness.client.memory_store.upsert_memory(
        title="Launch Token",
        description="A unique durable token for live validation.",
        memory_type="reference",
        content=f"The launch token is {token}.",
    )

    prompt = "What is the durable launch token for this workspace? Reply with only the token."
    session_id = f"copilotcode-memory-{uuid.uuid4().hex}"
    with live_session(harness.client, session_id=session_id) as session:
        session.send_and_wait(prompt, timeout=180.0)
        messages = session.get_messages()
        write_live_artifact(
            harness,
            session_id=session_id,
            label="memory-recall",
            prompt=prompt,
            messages=messages,
        )
        assert_assistant_contains_token(messages, token)


def test_live_smoke_test_saves_report_and_transcript_artifacts(tmp_path: Path) -> None:
    smoke_token = f"smoke-{uuid.uuid4().hex[:8]}"
    (tmp_path / "smoke-token.txt").write_text(smoke_token, encoding="utf-8")
    harness = build_live_harness(tmp_path)

    report = harness.client.smoke_test(
        live=True,
        prompt="Read the file `smoke-token.txt` and reply with only the token.",
        timeout=180.0,
        save_report_path=(harness.artifact_root or tmp_path) / "smoke-report.json",
        save_transcript_dir=harness.artifact_root or (tmp_path / ".live-artifacts"),
    )

    assert report.success is True
    assert report.session_id is not None
    assert report.report_path is not None
    assert report.transcript_path is not None
    assert Path(report.report_path).exists()
    assert Path(report.transcript_path).exists()


def test_live_safe_policy_and_skill_prompt(tmp_path: Path) -> None:
    token = f"verify-{uuid.uuid4().hex[:8]}"
    (tmp_path / "CLAUDE.md").write_text(
        f"Verification token: {token}\n",
        encoding="utf-8",
    )

    harness = build_live_harness(tmp_path)
    client = harness.client
    handler = build_permission_handler(
        policy="safe",
        permission_handler=None,
        allowed_roots=(
            client.config.working_path,
            client.config.app_config_home,
            client.memory_store.memory_dir,
        ),
        approved_shell_prefixes=client.config.approved_shell_prefixes,
        brand=client.config.brand,
    )
    denied = handler(
        type(
            "Req",
            (),
            {
                "tool_name": "edit",
                "read_only": False,
                "path": str(tmp_path.parent / "outside.txt"),
                "possible_paths": [],
                "full_command_text": "",
            },
        )(),
        {"toolName": "edit"},
    )
    assert denied.kind != "approved"

    prompt = (
        "/verify inspect CLAUDE.md in this workspace and report the verification token it contains."
    )
    session_id = f"copilotcode-verify-{uuid.uuid4().hex}"
    with live_session(client, session_id=session_id) as session:
        event = session.send_and_wait(prompt, timeout=180.0)
        assert event is not None
        messages = session.get_messages()
        write_live_artifact(
            harness,
            session_id=session_id,
            label="verify-skill",
            prompt=prompt,
            messages=messages,
        )
        assert_assistant_contains_token(messages, token)


def test_live_validate_include_live_preserves_artifacts(tmp_path: Path) -> None:
    if os.getenv(LIVE_VALIDATE_E2E_ENV) != "1":
        pytest.skip(
            f"Set {LIVE_VALIDATE_E2E_ENV}=1 to run the expensive validate --include-live end-to-end check.",
        )

    if os.getenv("COPILOTCODE_VALIDATE_SUBPROCESS") == "1":
        pytest.skip("Avoid recursive validate --include-live execution inside the validation subprocess.")

    artifact_dir = tmp_path / ".live-artifacts"
    env = dict(os.environ)
    env["COPILOTCODE_RUN_LIVE"] = "1"
    env["COPILOTCODE_LIVE_ARTIFACT_DIR"] = str(artifact_dir)
    env["COPILOTCODE_VALIDATE_SUBPROCESS"] = "1"

    result = subprocess.run(
        [sys.executable, "-m", "copilotcode_sdk", "validate", "--include-live", "--json"],
        cwd=str(Path.cwd()),
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    live_phase = next(phase for phase in payload["phases"] if phase["name"] == "live")
    assert live_phase["success"] is True
    assert live_phase["artifact_path"] == str(artifact_dir.resolve())
