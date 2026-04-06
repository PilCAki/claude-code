"""Session state machine for tracking session lifecycle."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

class SessionStatus(Enum):
    idle = "idle"
    running = "running"
    requires_action = "requires_action"

@dataclass(slots=True)
class RequiresActionDetails:
    tool_name: str
    action_description: str
    tool_use_id: str = ""
    request_id: str = ""

SessionStateListener = Callable[["SessionState", SessionStatus, "RequiresActionDetails | None"], None]

@dataclass
class SessionState:
    """Tracks session lifecycle state with listener support."""

    status: SessionStatus = SessionStatus.idle
    action_details: RequiresActionDetails | None = None
    turn_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0
    _listeners: list[SessionStateListener] = field(default_factory=list, repr=False)

    def add_listener(self, listener: SessionStateListener) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: SessionStateListener) -> None:
        self._listeners.remove(listener)

    def transition_to(self, status: SessionStatus, details: RequiresActionDetails | None = None) -> None:
        """Transition to a new state, notifying listeners."""
        old = self.status
        self.status = status
        self.action_details = details if status == SessionStatus.requires_action else None
        if old != status:
            self._notify(status, details)

    def start_turn(self) -> None:
        """Mark the beginning of a new turn."""
        self.turn_count += 1
        self.transition_to(SessionStatus.running)

    def end_turn(self) -> None:
        """Mark the end of a turn."""
        self.transition_to(SessionStatus.idle)

    def require_action(self, details: RequiresActionDetails) -> None:
        """Transition to requires_action state."""
        self.transition_to(SessionStatus.requires_action, details)

    def record_usage(self, *, input_tokens: int = 0, output_tokens: int = 0,
                     cache_read_tokens: int = 0, cache_creation_tokens: int = 0) -> None:
        """Accumulate token usage from an API response."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cache_read_tokens += cache_read_tokens
        self.total_cache_creation_tokens += cache_creation_tokens

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "status": self.status.value,
            "turn_count": self.turn_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.action_details:
            d["action_details"] = {
                "tool_name": self.action_details.tool_name,
                "action_description": self.action_details.action_description,
            }
        return d

    def _notify(self, status: SessionStatus, details: RequiresActionDetails | None) -> None:
        for listener in self._listeners:
            try:
                listener(self, status, details)
            except Exception:
                pass  # Never let listener errors crash the session
