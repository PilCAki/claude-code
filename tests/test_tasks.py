"""Tests for the task v2 store."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from copilotcode_sdk.tasks import TaskRecord, TaskStatus, TaskStore


class TestTaskRecord:
    def test_to_dict(self):
        task = TaskRecord(id=1, subject="Do X", status=TaskStatus.in_progress, owner="alice")
        d = task.to_dict()
        assert d["id"] == 1
        assert d["status"] == "in_progress"
        assert d["owner"] == "alice"

    def test_from_dict(self):
        task = TaskRecord.from_dict({"id": 2, "subject": "Fix Y", "status": "completed"})
        assert task.id == 2
        assert task.status == TaskStatus.completed

    def test_from_dict_defaults(self):
        task = TaskRecord.from_dict({"id": 3, "subject": "Z"})
        assert task.status == TaskStatus.pending
        assert task.owner == ""


class TestTaskStore:
    def test_create_and_get(self):
        store = TaskStore()
        t = store.create("Write tests")
        assert t.id == 1
        assert t.subject == "Write tests"
        assert t.status == TaskStatus.pending
        assert store.get(1) is t

    def test_sequential_ids(self):
        store = TaskStore()
        t1 = store.create("A")
        t2 = store.create("B")
        t3 = store.create("C")
        assert t1.id == 1
        assert t2.id == 2
        assert t3.id == 3

    def test_update_status(self):
        store = TaskStore()
        store.create("X")
        updated = store.update(1, status="in_progress")
        assert updated is not None
        assert updated.status == TaskStatus.in_progress

    def test_update_multiple_fields(self):
        store = TaskStore()
        store.create("X")
        updated = store.update(1, subject="Y", owner="bob", metadata={"key": "val"})
        assert updated is not None
        assert updated.subject == "Y"
        assert updated.owner == "bob"
        assert updated.metadata == {"key": "val"}

    def test_update_nonexistent(self):
        store = TaskStore()
        assert store.update(99, status="completed") is None

    def test_delete(self):
        store = TaskStore()
        store.create("To delete")
        assert store.delete(1) is True
        task = store.get(1)
        assert task is not None
        assert task.status == TaskStatus.deleted

    def test_delete_nonexistent(self):
        store = TaskStore()
        assert store.delete(99) is False

    def test_list_open_excludes_deleted(self):
        store = TaskStore()
        store.create("A")
        store.create("B")
        store.create("C")
        store.delete(2)
        open_tasks = store.list_open()
        assert [t.id for t in open_tasks] == [1, 3]

    def test_list_open_ordering(self):
        store = TaskStore()
        store.create("Third")
        store.create("First")
        store.create("Second")
        # Should be sorted by ID regardless of creation order
        ids = [t.id for t in store.list_open()]
        assert ids == [1, 2, 3]

    def test_has_open_tasks(self):
        store = TaskStore()
        assert store.has_open_tasks() is False
        store.create("X")
        assert store.has_open_tasks() is True
        store.update(1, status="completed")
        assert store.has_open_tasks() is False

    def test_summary_text_empty(self):
        store = TaskStore()
        assert store.summary_text() == ""

    def test_summary_text(self):
        store = TaskStore()
        store.create("Write code", owner="alice")
        store.create("Run tests")
        store.update(1, status="in_progress")
        text = store.summary_text()
        assert "**Open tasks:**" in text
        assert "#1 [in progress] (owner: alice): Write code" in text
        assert "#2 [pending]: Run tests" in text

    def test_persistence(self, tmp_path: Path):
        persist = tmp_path / "tasks.json"
        store1 = TaskStore(persist_path=persist)
        store1.create("Persisted task", owner="x")
        store1.update(1, status="in_progress")

        # Load from disk
        store2 = TaskStore(persist_path=persist)
        task = store2.get(1)
        assert task is not None
        assert task.subject == "Persisted task"
        assert task.status == TaskStatus.in_progress
        assert task.owner == "x"

        # Next ID should continue
        t = store2.create("Second")
        assert t.id == 2

    def test_persistence_corrupt_file(self, tmp_path: Path):
        persist = tmp_path / "tasks.json"
        persist.write_text("NOT JSON", encoding="utf-8")
        store = TaskStore(persist_path=persist)
        assert store.list_all() == []

    def test_metadata_merge(self):
        store = TaskStore()
        store.create("X", metadata={"a": 1})
        store.update(1, metadata={"b": 2})
        task = store.get(1)
        assert task is not None
        assert task.metadata == {"a": 1, "b": 2}
