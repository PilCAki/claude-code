"""Session memory controller — autonomous extraction via maintenance pass."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine, Sequence

from .extraction import build_session_memory_update_prompt
from .memory import MemoryStore


@dataclass
class SessionMemoryState:
    """Mutable per-session state for session memory extraction."""

    initialized: bool = False
    last_summarized_message_id: str | None = None
    tokens_at_last_extraction: int = 0
    extraction_started_at: float | None = None  # time.monotonic()
    tool_calls_since_last_extraction: int = 0


class SessionMemoryController:
    """Manages autonomous session-memory extraction via a maintenance pass.

    Instead of nudging the main model to write memory, this controller runs
    a separate short-lived session (maintenance pass) to update the structured
    session-memory file after each turn when thresholds are met.
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        *,
        min_init_tokens: int = 10_000,
        min_update_tokens: int = 5_000,
        tool_calls_between_updates: int = 3,
        timeout_seconds: float = 300.0,
        promote_on_destroy: bool = True,
    ) -> None:
        self._memory_store = memory_store
        self._min_init_tokens = min_init_tokens
        self._min_update_tokens = min_update_tokens
        self._tool_calls_between_updates = tool_calls_between_updates
        self._timeout_seconds = timeout_seconds
        self._promote_on_destroy = promote_on_destroy
        self._state = SessionMemoryState()

    @property
    def state(self) -> SessionMemoryState:
        return self._state

    def record_tool_call(self) -> None:
        """Increment the tool-call counter (called from hooks)."""
        self._state.tool_calls_since_last_extraction += 1

    def should_extract(
        self,
        context_tokens: int,
        *,
        has_tool_calls_in_last_turn: bool = True,
    ) -> bool:
        """Decide whether to run a maintenance extraction pass.

        Mirrors Claude Code's trigger model:
        1. Init gate: first time context_tokens >= min_init_tokens
        2. Token growth >= min_update_tokens since last extraction
        3. Tool call threshold met OR natural break (no tools in last turn)
        4. Sequential guard: no extraction already in flight
        """
        # Sequential guard
        if self._state.extraction_started_at is not None:
            # Stale guard: treat as cleared if older than the configured timeout
            elapsed = time.monotonic() - self._state.extraction_started_at
            if elapsed < self._timeout_seconds:
                return False
            # Stale — clear it
            self._state.extraction_started_at = None

        # Init gate
        if not self._state.initialized:
            if context_tokens < self._min_init_tokens:
                return False
            self._state.initialized = True

        # Token growth gate (always required)
        token_growth = context_tokens - self._state.tokens_at_last_extraction
        if token_growth < self._min_update_tokens:
            return False

        # Tool call OR natural break gate
        tool_calls_met = (
            self._state.tool_calls_since_last_extraction
            >= self._tool_calls_between_updates
        )
        natural_break = not has_tool_calls_in_last_turn

        return tool_calls_met or natural_break

    def build_update_prompt(self, existing_memory: str = "") -> str:
        """Build the prompt for the maintenance session."""
        return build_session_memory_update_prompt(existing_memory=existing_memory)

    async def run_extraction(
        self,
        create_session_fn: Callable[..., Coroutine[Any, Any, Any]],
        messages: Sequence[dict[str, Any]],
        context_tokens: int = 0,
    ) -> bool:
        """Run the maintenance extraction pass.

        Args:
            create_session_fn: Async callable that creates a short-lived
                Copilot SDK session for the maintenance pass.
            messages: The conversation messages (transcript tail).
            context_tokens: Current context token estimate.

        Returns:
            True if extraction succeeded, False otherwise.
        """
        self._state.extraction_started_at = time.monotonic()
        session = None
        try:
            # Read existing session memory
            existing = self._memory_store.read_session_memory()

            # Build the update prompt
            prompt = self.build_update_prompt(existing)

            # Build a transcript summary for context — slice from last summarized point
            transcript_context = _format_transcript_tail(
                messages,
                last_summarized_message_id=self._state.last_summarized_message_id,
            )
            full_prompt = (
                f"{prompt}\n\n"
                "Recent conversation activity:\n"
                f"<transcript>\n{transcript_context}\n</transcript>"
            )

            # Create and use the maintenance session
            session = await create_session_fn()
            result = await session.send_and_wait(
                full_prompt,
                timeout=self._timeout_seconds,
            )

            # Extract the response text
            response_text = _extract_response_text(result)
            if response_text:
                self._memory_store.write_session_memory(response_text)

            # Update state
            self._state.tokens_at_last_extraction = context_tokens
            self._state.tool_calls_since_last_extraction = 0
            if messages:
                last_msg = messages[-1]
                msg_id = (
                    last_msg.get("id")
                    or last_msg.get("uuid")
                    or last_msg.get("message_id")
                )
                if msg_id:
                    self._state.last_summarized_message_id = str(msg_id)

            return True
        except Exception:
            return False
        finally:
            self._state.extraction_started_at = None
            if session is not None:
                try:
                    await session.destroy()
                except Exception:
                    pass

    async def finalize(self) -> list[Path]:
        """Promote session memory to durable memory on session end.

        Returns the list of promoted memory file paths.
        """
        if not self._promote_on_destroy:
            return []
        return self._memory_store.promote_session_memory()


def _format_transcript_tail(
    messages: Sequence[dict[str, Any]],
    max_chars: int = 30_000,
    *,
    last_summarized_message_id: str | None = None,
) -> str:
    """Format the transcript for the maintenance prompt.

    When *last_summarized_message_id* is provided, only messages **after** that
    ID are included (so the maintenance pass sees only unsummarized content).
    Falls back to the tail-based approach when the ID is not found.
    """
    # Slice from last_summarized_message_id if provided
    sliced = list(messages)
    if last_summarized_message_id is not None:
        for i, msg in enumerate(messages):
            data = msg.get("data") or {}
            msg_id = (
                msg.get("id")
                or msg.get("uuid")
                or msg.get("message_id")
                or (data.get("id") if isinstance(data, dict) else None)
            )
            if msg_id is not None and str(msg_id) == last_summarized_message_id:
                sliced = list(messages[i + 1:])
                break

    lines: list[str] = []
    total = 0
    for msg in reversed(sliced):
        # Support both plain dicts and SDK event dicts
        data = msg.get("data") or {}
        role = msg.get("role") or (data.get("role") if isinstance(data, dict) else None) or msg.get("type", "unknown")
        content = msg.get("content", "")
        if not content and isinstance(data, dict):
            content = data.get("content", "") or data.get("message", "") or ""
        line = f"[{role}]: {str(content)[:2000]}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    lines.reverse()
    return "\n".join(lines)


def _extract_response_text(result: Any) -> str:
    """Extract text from a Copilot SDK response.

    Handles plain strings, dicts, and SessionEvent dataclass objects
    returned by the raw Copilot SDK.
    """
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    # SessionEvent dataclass — content lives at result.data.content
    data = getattr(result, "data", None)
    if data is not None:
        content = getattr(data, "content", None)
        if isinstance(content, str):
            return content
    if isinstance(result, dict):
        # Try common response shapes
        for key in ("content", "text", "message", "result"):
            if key in result and isinstance(result[key], str):
                return result[key]
    return ""
