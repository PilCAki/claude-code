"""Subagent context and forking with prompt cache sharing.

When a parent session spawns a child (maintenance pass, agent delegation),
the child can reuse the parent's cacheable system prompt prefix. This avoids
re-sending and re-tokenizing the large static portion of the system prompt,
letting the API cache it across parent and children.

The key mechanism: both parent and child use ``system_message.mode = "append"``
with the *same* cacheable prefix. The non-cacheable suffix differs per session.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Sequence


def _normalize_payload(value: Any) -> Any:
    """Convert SDK event/message objects to JSON-serializable Python values.

    Mirrors ``_normalize_sdk_payload`` from ``client.py`` so subagent code
    can normalize raw SDK messages without importing from the client module.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _normalize_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_payload(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _normalize_payload(to_dict())
    try:
        return _normalize_payload(vars(value))
    except TypeError:
        return str(value)


@dataclass(frozen=True, slots=True)
class SubagentSpec:
    """Specification for spawning a child session."""
    role: str  # e.g. "maintenance", "researcher", "verifier"
    system_prompt_suffix: str  # appended after the shared cacheable prefix
    tools: Sequence[str] = ()
    max_turns: int = 0
    timeout_seconds: float = 3600.0


@dataclass
class SubagentContext:
    """Tracks parent/child relationships and shared prompt state.

    The parent creates this at session start. Each child fork inherits
    ``cacheable_prefix`` so the LLM API can cache it once and reuse
    across all sessions in the tree.
    """
    parent_session_id: str
    cacheable_prefix: str  # The shared static system prompt
    children: list[str] = field(default_factory=list)  # child session IDs

    def build_child_system_message(
        self,
        spec: SubagentSpec,
    ) -> dict[str, Any]:
        """Build a system_message dict for a child session.

        Uses ``mode: "append"`` with the same cacheable prefix as the parent,
        plus the child-specific suffix. This means the API will cache-hit on
        the prefix across parent and all children.
        """
        # The cacheable prefix is the same as the parent's. The suffix is
        # child-specific (role prompt, restricted tool list, etc).
        combined = self.cacheable_prefix
        if spec.system_prompt_suffix:
            combined = combined + "\n\n" + spec.system_prompt_suffix
        return {
            "mode": "append",
            "content": combined,
        }

    def build_maintenance_system_message(
        self,
        task_description: str = "session-memory maintenance agent",
    ) -> dict[str, Any]:
        """Build a minimal system message for maintenance passes.

        Maintenance sessions (extraction, compaction) get the cacheable prefix
        plus a short role description. No tools.
        """
        suffix = (
            f"You are a {task_description}. "
            "Respond only with the requested output. Do not call any tools."
        )
        return {
            "mode": "append",
            "content": self.cacheable_prefix + "\n\n" + suffix,
        }

    def register_child(self, child_session_id: str) -> None:
        self.children.append(child_session_id)


@dataclass
class ChildSession:
    """A forked child session carrying its spec constraints.

    Wraps the raw SDK session with the ``SubagentSpec`` that created it,
    so callers can enforce ``timeout_seconds``, ``max_turns``, and check
    ``tools`` without threading the spec separately.
    """
    session: Any
    spec: SubagentSpec
    session_id: str


class MaxTurnsExceeded(Exception):
    """Raised when an enforced child session exceeds its max_turns limit."""

    def __init__(self, max_turns: int, actual_turns: int) -> None:
        self.max_turns = max_turns
        self.actual_turns = actual_turns
        super().__init__(
            f"Child session exceeded max_turns={max_turns} (attempted turn {actual_turns})"
        )


class EnforcedChildSession:
    """Wraps a :class:`ChildSession` with turn-count and timeout enforcement.

    ``fork_child()`` returns this instead of a bare ``ChildSession`` so that
    ``spec.max_turns`` and ``spec.timeout_seconds`` are actually enforced.
    """

    def __init__(self, child: ChildSession) -> None:
        self._child = child
        self._turn_count = 0

    @property
    def session(self) -> Any:
        return self._child.session

    @property
    def spec(self) -> SubagentSpec:
        return self._child.spec

    @property
    def session_id(self) -> str:
        return self._child.session_id

    @property
    def turn_count(self) -> int:
        return self._turn_count

    async def send_and_wait(self, prompt: str, *, timeout: float | None = None) -> Any:
        """Send a prompt to the child session with enforcement.

        Raises :class:`MaxTurnsExceeded` if the turn limit is hit.
        Raises :class:`asyncio.TimeoutError` if the timeout is exceeded.
        """
        if self.spec.max_turns > 0 and self._turn_count >= self.spec.max_turns:
            raise MaxTurnsExceeded(self.spec.max_turns, self._turn_count)
        self._turn_count += 1
        effective_timeout = timeout or (
            self.spec.timeout_seconds if self.spec.timeout_seconds > 0 else None
        )
        if effective_timeout:
            return await self._child.session.send_and_wait(
                prompt, timeout=effective_timeout,
            )
        return await self._child.session.send_and_wait(prompt)

    async def get_last_response_text(self) -> str:
        """Extract text from the last assistant message in the session.

        The copilot SDK's ``send_and_wait`` returns a ``SessionEvent``, not text.
        Call this after ``send_and_wait`` to get the actual assistant response.
        Uses the same normalization as ``CopilotCodeSession`` to convert SDK
        message objects to plain dicts.
        """
        get_messages = getattr(self._child.session, "get_messages", None)
        if get_messages is None:
            return ""
        raw_messages = await get_messages()
        # Normalize SDK objects to plain dicts (same as _normalize_sdk_messages)
        messages = [_normalize_payload(m) for m in raw_messages]
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role") or msg.get("type") or ""
            if role == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return " ".join(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                return str(content)
        return ""

    async def destroy(self) -> None:
        destroy = getattr(self._child.session, "destroy", None)
        if destroy:
            await destroy()


def build_subagent_context(
    session_id: str,
    cacheable_prefix: str,
) -> SubagentContext:
    """Create a SubagentContext for a parent session."""
    return SubagentContext(
        parent_session_id=session_id,
        cacheable_prefix=cacheable_prefix,
    )
