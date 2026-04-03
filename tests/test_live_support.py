from __future__ import annotations

from pathlib import Path

from tests.live_support import LiveHarness, assistant_text, write_live_artifact


def test_assistant_text_prefers_assistant_messages_and_normalizes_whitespace() -> None:
    messages = [
        {"type": "user", "content": "ignore"},
        {"type": "assistant", "content": ["hello", {"nested": "world"}]},
        {"role": "assistant", "text": " spaced   text "},
    ]

    text = assistant_text(messages)

    assert text == "hello world spaced text"


def test_write_live_artifact_persists_payload(tmp_path: Path) -> None:
    harness = LiveHarness(client=None, artifact_root=tmp_path)  # type: ignore[arg-type]

    path = write_live_artifact(
        harness,
        session_id="session-123",
        label="memory",
        prompt="Prompt",
        messages=[{"type": "assistant", "content": "token-123"}],
    )

    assert path is not None and path.exists()
    payload = path.read_text(encoding="utf-8")
    assert "session-123" in payload
    assert "token-123" in payload
