"""Structured event system for CopilotCode sessions.

Provides typed event classes and an EventBus that hooks and the client
emit to. The existing ``on_event`` callback becomes one listener on the bus.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class EventType(Enum):
    """All event types emitted by the SDK."""
    session_started = "session.started"
    session_destroyed = "session.destroyed"
    turn_started = "turn.started"
    turn_completed = "turn.completed"
    tool_called = "tool.called"
    tool_result = "tool.result"
    tool_denied = "tool.denied"
    cost_accumulated = "cost.accumulated"
    budget_warning = "budget.warning"
    budget_exhausted = "budget.exhausted"
    context_warning = "context.warning"
    compaction_triggered = "compaction.triggered"
    extraction_started = "extraction.started"
    extraction_completed = "extraction.completed"
    mcp_health_warning = "mcp.health_warning"
    model_switched = "model.switched"
    error_occurred = "error.occurred"
    file_changed = "file.changed"
    memory_saved = "memory.saved"


@dataclass(frozen=True, slots=True)
class Event:
    """Base event emitted by the SDK."""
    type: EventType
    timestamp: float = field(default_factory=time.monotonic)
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "timestamp": self.timestamp,
            **self.data,
        }


# Convenience constructors for common events
def session_started(*, session_id: str = "", source: str = "create") -> Event:
    return Event(EventType.session_started, data={"session_id": session_id, "source": source})


def session_destroyed(*, session_id: str = "") -> Event:
    return Event(EventType.session_destroyed, data={"session_id": session_id})


def turn_started(*, turn_count: int) -> Event:
    return Event(EventType.turn_started, data={"turn_count": turn_count})


def turn_completed(*, turn_count: int, input_tokens: int = 0, output_tokens: int = 0) -> Event:
    return Event(EventType.turn_completed, data={
        "turn_count": turn_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    })


def tool_called(*, tool_name: str, cached: bool = False) -> Event:
    return Event(EventType.tool_called, data={"tool_name": tool_name, "cached": cached})


def tool_result(*, tool_name: str, result_chars: int, persisted: bool = False) -> Event:
    return Event(EventType.tool_result, data={
        "tool_name": tool_name,
        "result_chars": result_chars,
        "persisted": persisted,
    })


def tool_denied(*, tool_name: str, reason: str) -> Event:
    return Event(EventType.tool_denied, data={"tool_name": tool_name, "reason": reason})


def cost_accumulated(*, total_cost: float, turn_cost: float, model: str = "") -> Event:
    return Event(EventType.cost_accumulated, data={
        "total_cost": total_cost,
        "turn_cost": turn_cost,
        "model": model,
    })


def budget_warning(*, consumed: int, total: int, progress: float) -> Event:
    return Event(EventType.budget_warning, data={
        "consumed": consumed,
        "total": total,
        "progress": progress,
    })


def budget_exhausted(*, consumed: int, total: int) -> Event:
    return Event(EventType.budget_exhausted, data={"consumed": consumed, "total": total})


def context_warning(*, ratio: float, level: str) -> Event:
    return Event(EventType.context_warning, data={"ratio": ratio, "level": level})


def compaction_triggered(*, mode: str = "full") -> Event:
    return Event(EventType.compaction_triggered, data={"mode": mode})


def model_switched(*, from_model: str, to_model: str) -> Event:
    return Event(EventType.model_switched, data={"from": from_model, "to": to_model})


def error_occurred(*, error: str, recoverable: bool, context: str = "") -> Event:
    return Event(EventType.error_occurred, data={
        "error": error,
        "recoverable": recoverable,
        "context": context,
    })


def file_changed(*, path: str, change_type: str) -> Event:
    return Event(EventType.file_changed, data={"path": path, "change_type": change_type})


EventListener = Callable[[Event], None]


class EventBus:
    """Central event bus for CopilotCode sessions.

    Listeners can subscribe to all events or filter by type.
    Listener errors are silently caught — they must never crash the session.
    """

    def __init__(self) -> None:
        self._global_listeners: list[EventListener] = []
        self._typed_listeners: dict[EventType, list[EventListener]] = {}
        self._history: list[Event] = []
        self._max_history: int = 500

    def subscribe(
        self,
        listener: EventListener,
        *,
        event_type: EventType | None = None,
    ) -> None:
        """Subscribe a listener. If event_type is None, listens to all events."""
        if event_type is None:
            self._global_listeners.append(listener)
        else:
            self._typed_listeners.setdefault(event_type, []).append(listener)

    def unsubscribe(
        self,
        listener: EventListener,
        *,
        event_type: EventType | None = None,
    ) -> None:
        """Remove a listener."""
        if event_type is None:
            try:
                self._global_listeners.remove(listener)
            except ValueError:
                pass
        else:
            listeners = self._typed_listeners.get(event_type, [])
            try:
                listeners.remove(listener)
            except ValueError:
                pass

    def emit(self, event: Event) -> None:
        """Emit an event to all matching listeners."""
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        for listener in self._global_listeners:
            try:
                listener(event)
            except Exception:
                pass

        for listener in self._typed_listeners.get(event.type, []):
            try:
                listener(event)
            except Exception:
                pass

    @property
    def history(self) -> list[Event]:
        """Recent event history (up to max_history)."""
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()

    def events_of_type(self, event_type: EventType) -> list[Event]:
        """Filter history by event type."""
        return [e for e in self._history if e.type == event_type]
