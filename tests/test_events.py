"""Tests for copilotcode_sdk.events module."""
from __future__ import annotations

from copilotcode_sdk.events import (
    Event,
    EventBus,
    EventType,
    budget_exhausted,
    budget_warning,
    context_warning,
    cost_accumulated,
    error_occurred,
    file_changed,
    model_switched,
    session_destroyed,
    session_started,
    tool_called,
    tool_denied,
    tool_result,
    turn_completed,
    turn_started,
)


class TestEvent:
    def test_to_dict(self) -> None:
        e = session_started(session_id="s1", source="create")
        d = e.to_dict()
        assert d["type"] == "session.started"
        assert d["session_id"] == "s1"
        assert d["source"] == "create"
        assert "timestamp" in d

    def test_all_constructors_return_events(self) -> None:
        events = [
            session_started(session_id="s"),
            session_destroyed(session_id="s"),
            turn_started(turn_count=1),
            turn_completed(turn_count=1, input_tokens=10, output_tokens=5),
            tool_called(tool_name="read"),
            tool_result(tool_name="read", result_chars=100),
            tool_denied(tool_name="bash", reason="dangerous"),
            cost_accumulated(total_cost=0.01, turn_cost=0.005, model="opus"),
            budget_warning(consumed=400, total=500, progress=0.8),
            budget_exhausted(consumed=500, total=500),
            context_warning(ratio=0.85, level="warning"),
            model_switched(from_model="opus", to_model="haiku"),
            error_occurred(error="timeout", recoverable=True, context="model_call"),
            file_changed(path="file.py", change_type="modified"),
        ]
        for e in events:
            assert isinstance(e, Event)
            assert isinstance(e.type, EventType)
            assert e.timestamp > 0


class TestEventBus:
    def test_global_listener_receives_all(self) -> None:
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(received.append)

        bus.emit(session_started(session_id="s"))
        bus.emit(turn_started(turn_count=1))

        assert len(received) == 2
        assert received[0].type == EventType.session_started
        assert received[1].type == EventType.turn_started

    def test_typed_listener_filters(self) -> None:
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(received.append, event_type=EventType.turn_started)

        bus.emit(session_started(session_id="s"))
        bus.emit(turn_started(turn_count=1))
        bus.emit(turn_started(turn_count=2))

        assert len(received) == 2
        assert all(e.type == EventType.turn_started for e in received)

    def test_unsubscribe_global(self) -> None:
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(received.append)
        bus.emit(session_started(session_id="s"))
        assert len(received) == 1

        bus.unsubscribe(received.append)
        bus.emit(session_started(session_id="s2"))
        assert len(received) == 1  # no more events

    def test_unsubscribe_typed(self) -> None:
        bus = EventBus()
        received: list[Event] = []
        bus.subscribe(received.append, event_type=EventType.turn_started)
        bus.emit(turn_started(turn_count=1))
        assert len(received) == 1

        bus.unsubscribe(received.append, event_type=EventType.turn_started)
        bus.emit(turn_started(turn_count=2))
        assert len(received) == 1

    def test_listener_error_doesnt_crash(self) -> None:
        bus = EventBus()

        def bad_listener(e: Event) -> None:
            raise ValueError("boom")

        good_received: list[Event] = []
        bus.subscribe(bad_listener)
        bus.subscribe(good_received.append)

        bus.emit(session_started(session_id="s"))
        assert len(good_received) == 1  # good listener still ran

    def test_history(self) -> None:
        bus = EventBus()
        bus.emit(session_started(session_id="s"))
        bus.emit(turn_started(turn_count=1))

        assert len(bus.history) == 2
        assert bus.history[0].type == EventType.session_started

    def test_history_capped(self) -> None:
        bus = EventBus()
        bus._max_history = 5
        for i in range(10):
            bus.emit(turn_started(turn_count=i))
        assert len(bus.history) == 5

    def test_events_of_type(self) -> None:
        bus = EventBus()
        bus.emit(session_started(session_id="s"))
        bus.emit(turn_started(turn_count=1))
        bus.emit(turn_started(turn_count=2))

        turns = bus.events_of_type(EventType.turn_started)
        assert len(turns) == 2

    def test_clear_history(self) -> None:
        bus = EventBus()
        bus.emit(session_started(session_id="s"))
        bus.clear_history()
        assert len(bus.history) == 0
