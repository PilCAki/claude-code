"""Tests for task v2 tool handlers."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from copilotcode_sdk.tasks import TaskStatus, TaskStore
from copilotcode_sdk.task_tools import build_task_tools


def _make_invocation(arguments: dict | None = None):
    """Create a minimal ToolInvocation-like object."""
    inv = MagicMock()
    inv.session_id = "test-session"
    inv.tool_call_id = "call-1"
    inv.tool_name = "test"
    inv.arguments = arguments
    return inv


@pytest.fixture
def store():
    return TaskStore()


@pytest.fixture
def tools(store):
    """Build tools using mocked copilot.types to avoid requiring the SDK."""
    # Patch the import inside build_task_tools
    import copilotcode_sdk.task_tools as tt
    import types

    # Create minimal fakes
    class FakeToolResult:
        def __init__(self, text_result_for_llm="", result_type="success", **kw):
            self.text_result_for_llm = text_result_for_llm
            self.result_type = result_type

    class FakeTool:
        def __init__(self, name="", description="", handler=None, parameters=None,
                     overrides_built_in_tool=False, skip_permission=False):
            self.name = name
            self.description = description
            self.handler = handler
            self.parameters = parameters

    # Create a fake copilot.types module
    fake_types = types.ModuleType("copilot.types")
    fake_types.Tool = FakeTool
    fake_types.ToolInvocation = MagicMock
    fake_types.ToolResult = FakeToolResult

    import sys
    original = sys.modules.get("copilot.types")
    sys.modules["copilot.types"] = fake_types
    try:
        result = build_task_tools(store)
    finally:
        if original is not None:
            sys.modules["copilot.types"] = original
        else:
            sys.modules.pop("copilot.types", None)
    return {t.name: t for t in result}


def _call(tools, name, args=None):
    handler = tools[name].handler
    inv = _make_invocation(args)
    return handler(inv)


class TestTaskCreate:
    def test_create_basic(self, tools, store):
        result = _call(tools, "TaskCreate", {"subject": "Write tests"})
        assert "Task #1 created" in result.text_result_for_llm
        assert store.get(1) is not None
        assert store.get(1).subject == "Write tests"

    def test_create_with_fields(self, tools, store):
        result = _call(tools, "TaskCreate", {
            "subject": "Implement feature",
            "description": "Build the login page",
            "owner": "alice",
        })
        task = store.get(1)
        assert task.description == "Build the login page"
        assert task.owner == "alice"

    def test_create_missing_subject(self, tools):
        result = _call(tools, "TaskCreate", {})
        assert result.result_type == "error"
        assert "subject" in result.text_result_for_llm.lower()

    def test_create_empty_subject(self, tools):
        result = _call(tools, "TaskCreate", {"subject": "   "})
        assert result.result_type == "error"


class TestTaskUpdate:
    def test_update_status(self, tools, store):
        store.create("X")
        result = _call(tools, "TaskUpdate", {"task_id": 1, "status": "in_progress"})
        assert "updated" in result.text_result_for_llm.lower()
        assert store.get(1).status == TaskStatus.in_progress

    def test_update_completed_shows_remaining(self, tools, store):
        store.create("A")
        store.create("B")
        result = _call(tools, "TaskUpdate", {"task_id": 1, "status": "completed"})
        assert "1 task(s) remaining" in result.text_result_for_llm

    def test_update_last_completed(self, tools, store):
        store.create("A")
        result = _call(tools, "TaskUpdate", {"task_id": 1, "status": "completed"})
        assert "All tasks complete" in result.text_result_for_llm

    def test_update_invalid_status(self, tools, store):
        store.create("X")
        result = _call(tools, "TaskUpdate", {"task_id": 1, "status": "bogus"})
        assert result.result_type == "error"

    def test_update_nonexistent(self, tools):
        result = _call(tools, "TaskUpdate", {"task_id": 99, "status": "completed"})
        assert result.result_type == "error"

    def test_update_missing_id(self, tools):
        result = _call(tools, "TaskUpdate", {"status": "completed"})
        assert result.result_type == "error"

    def test_delete_action(self, tools, store):
        store.create("X")
        result = _call(tools, "TaskUpdate", {"task_id": 1, "action": "deleted"})
        assert "deleted" in result.text_result_for_llm.lower()
        assert store.get(1).status == TaskStatus.deleted

    def test_delete_nonexistent(self, tools):
        result = _call(tools, "TaskUpdate", {"task_id": 99, "action": "deleted"})
        assert result.result_type == "error"


class TestTaskList:
    def test_empty(self, tools):
        result = _call(tools, "TaskList")
        assert "no open tasks" in result.text_result_for_llm.lower()

    def test_list_with_tasks(self, tools, store):
        store.create("A", owner="alice")
        store.create("B")
        store.update(1, status="in_progress")
        result = _call(tools, "TaskList")
        assert "#1 [in progress] (owner: alice): A" in result.text_result_for_llm
        assert "#2 [pending]: B" in result.text_result_for_llm

    def test_excludes_deleted(self, tools, store):
        store.create("A")
        store.create("B")
        store.delete(1)
        result = _call(tools, "TaskList")
        assert "#1" not in result.text_result_for_llm
        assert "#2" in result.text_result_for_llm


class TestTaskGet:
    def test_get_existing(self, tools, store):
        store.create("Test task", description="Details here")
        result = _call(tools, "TaskGet", {"task_id": 1})
        assert "Test task" in result.text_result_for_llm
        assert "Details here" in result.text_result_for_llm

    def test_get_nonexistent(self, tools):
        result = _call(tools, "TaskGet", {"task_id": 99})
        assert result.result_type == "error"

    def test_get_missing_id(self, tools):
        result = _call(tools, "TaskGet", {})
        assert result.result_type == "error"

    def test_get_invalid_id(self, tools):
        result = _call(tools, "TaskGet", {"task_id": "abc"})
        assert result.result_type == "error"
