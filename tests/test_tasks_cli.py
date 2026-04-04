"""Tests for the tasks CLI subcommands."""
from __future__ import annotations

from pathlib import Path

import pytest

from copilotcode_sdk.cli import main
from copilotcode_sdk.tasks import TaskStore


@pytest.fixture
def task_store(tmp_path: Path):
    persist = tmp_path / "tasks" / "tasks.json"
    store = TaskStore(persist_path=persist)
    store.create("Write code", owner="alice")
    store.create("Run tests")
    store.update(1, status="in_progress")
    return store, tmp_path


class TestTasksList:
    def test_list_open(self, task_store, capsys):
        store, root = task_store
        rc = main(["tasks", "list", "--memory-root", str(root)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "#1 [in progress] (owner: alice): Write code" in out
        assert "#2 [pending]: Run tests" in out

    def test_list_json(self, task_store, capsys):
        store, root = task_store
        rc = main(["tasks", "list", "--json", "--memory-root", str(root)])
        assert rc == 0
        import json
        data = json.loads(capsys.readouterr().out)
        assert len(data) == 2
        assert data[0]["id"] == 1

    def test_list_all(self, task_store, capsys):
        store, root = task_store
        store.delete(2)
        rc = main(["tasks", "list", "--all", "--memory-root", str(root)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "#2 [deleted]: Run tests" in out

    def test_list_empty(self, tmp_path, capsys):
        rc = main(["tasks", "list", "--memory-root", str(tmp_path)])
        assert rc == 0
        assert "No tasks found" in capsys.readouterr().out


class TestTasksGet:
    def test_get_existing(self, task_store, capsys):
        store, root = task_store
        rc = main(["tasks", "get", "1", "--memory-root", str(root)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Write code" in out
        assert "in progress" in out

    def test_get_json(self, task_store, capsys):
        store, root = task_store
        rc = main(["tasks", "get", "1", "--json", "--memory-root", str(root)])
        assert rc == 0
        import json
        data = json.loads(capsys.readouterr().out)
        assert data["subject"] == "Write code"

    def test_get_not_found(self, task_store, capsys):
        store, root = task_store
        rc = main(["tasks", "get", "99", "--memory-root", str(root)])
        assert rc == 1
        assert "not found" in capsys.readouterr().out


class TestTasksClear:
    def test_clear(self, task_store, capsys):
        store, root = task_store
        rc = main(["tasks", "clear", "--memory-root", str(root)])
        assert rc == 0
        assert "Deleted" in capsys.readouterr().out
        assert not store.persist_path.exists()

    def test_clear_no_file(self, tmp_path, capsys):
        rc = main(["tasks", "clear", "--memory-root", str(tmp_path)])
        assert rc == 0
        assert "No task store" in capsys.readouterr().out
