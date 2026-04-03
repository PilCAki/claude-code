from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, AsyncIterator, Iterator

import pytest

from copilotcode_sdk import CopilotCodeClient, CopilotCodeConfig


LIVE_ARTIFACT_ENV = "COPILOTCODE_LIVE_ARTIFACT_DIR"


@dataclass(frozen=True, slots=True)
class LiveHarness:
    client: CopilotCodeClient
    artifact_root: Path | None


def build_live_harness(workspace: Path) -> LiveHarness:
    if os.getenv("COPILOTCODE_RUN_LIVE") != "1":
        pytest.skip("Set COPILOTCODE_RUN_LIVE=1 to run live CopilotCode validation.")

    model = os.getenv("COPILOTCODE_TEST_MODEL")
    copilot_config_dir = os.getenv("COPILOTCODE_TEST_COPILOT_CONFIG_DIR")
    github_token = os.getenv("COPILOTCODE_TEST_GITHUB_TOKEN")
    cli_path = os.getenv("COPILOTCODE_TEST_CLI_PATH")
    artifact_root = os.getenv(LIVE_ARTIFACT_ENV)
    resolved_artifact_root = (
        Path(artifact_root).expanduser().resolve(strict=False)
        if artifact_root
        else None
    )
    if resolved_artifact_root is not None:
        resolved_artifact_root.mkdir(parents=True, exist_ok=True)

    config = CopilotCodeConfig(
        working_directory=workspace,
        memory_root=workspace / ".mem",
        model=model,
        copilot_config_dir=copilot_config_dir,
        github_token=github_token,
        cli_path=cli_path,
    )
    return LiveHarness(
        client=CopilotCodeClient(config),
        artifact_root=resolved_artifact_root,
    )


@asynccontextmanager
async def live_session(
    client: CopilotCodeClient,
    *,
    session_id: str | None = None,
    resume: bool = False,
) -> AsyncIterator[Any]:
    session = (
        await client.resume_session(session_id)
        if resume and session_id is not None
        else await client.create_session(session_id=session_id)
    )
    try:
        yield session
    finally:
        await session.disconnect()


def assistant_text(messages: list[Any]) -> str:
    assistant_messages: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(
            message.get("type")
            or message.get("role")
            or message.get("kind")
            or "",
        ).lower()
        if role in {"assistant", "response"}:
            assistant_messages.append(message)

    selected_messages = assistant_messages or messages
    raw = " ".join(_extract_strings(selected_messages))
    return re.sub(r"\s+", " ", raw).strip()


def assert_assistant_contains_token(messages: list[Any], token: str) -> None:
    transcript = assistant_text(messages)
    assert token in transcript, transcript


def write_live_artifact(
    harness: LiveHarness,
    *,
    session_id: str,
    label: str,
    prompt: str,
    messages: list[Any],
) -> Path | None:
    if harness.artifact_root is None:
        return None

    payload = {
        "session_id": session_id,
        "label": label,
        "prompt": prompt,
        "assistant_text": assistant_text(messages),
        "messages": messages,
    }
    path = harness.artifact_root / f"{session_id}-{label}.json"
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def _extract_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        found: list[str] = []
        for key, child in value.items():
            if key in {"type", "role", "kind"}:
                continue
            found.extend(_extract_strings(child))
        return found
    if isinstance(value, (list, tuple)):
        found = []
        for child in value:
            found.extend(_extract_strings(child))
        return found
    return []
