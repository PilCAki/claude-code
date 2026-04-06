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


# ---------------------------------------------------------------------------
# Wave 3: Task dependencies
# ---------------------------------------------------------------------------


class TestTaskDependencies:
    def test_add_dependency(self):
        store = TaskStore()
        store.create("Blocker")
        store.create("Blocked")

        ok = store.add_dependency(2, blocked_by=1)

        assert ok is True
        task2 = store.get(2)
        assert task2 is not None
        assert 1 in task2.blocked_by
        task1 = store.get(1)
        assert task1 is not None
        assert 2 in task1.blocks

    def test_add_dependency_nonexistent(self):
        store = TaskStore()
        store.create("A")
        assert store.add_dependency(1, blocked_by=99) is False

    def test_add_dependency_self_reference(self):
        store = TaskStore()
        store.create("A")
        assert store.add_dependency(1, blocked_by=1) is False

    def test_add_dependency_circular(self):
        store = TaskStore()
        store.create("A")
        store.create("B")
        store.add_dependency(2, blocked_by=1)
        # B is blocked by A, so A cannot be blocked by B
        assert store.add_dependency(1, blocked_by=2) is False

    def test_add_dependency_transitive_circular(self):
        store = TaskStore()
        store.create("A")
        store.create("B")
        store.create("C")
        store.add_dependency(2, blocked_by=1)  # B blocked by A
        store.add_dependency(3, blocked_by=2)  # C blocked by B
        # A cannot be blocked by C (A→B→C→A would be circular)
        assert store.add_dependency(1, blocked_by=3) is False

    def test_blockers_incomplete(self):
        store = TaskStore()
        store.create("Blocker")
        store.create("Blocked")
        store.add_dependency(2, blocked_by=1)

        assert store.blockers_incomplete(2) == [1]

        store.update(1, status="completed")
        assert store.blockers_incomplete(2) == []

    def test_cannot_start_with_incomplete_blockers(self):
        store = TaskStore()
        store.create("Blocker")
        store.create("Blocked")
        store.add_dependency(2, blocked_by=1)

        result = store.update(2, status="in_progress")

        assert result is not None
        # Task stays pending because blocker is incomplete
        assert result.status == TaskStatus.pending
        assert "_blocked_reason" in result.metadata

    def test_can_start_after_blocker_completes(self):
        store = TaskStore()
        store.create("Blocker")
        store.create("Blocked")
        store.add_dependency(2, blocked_by=1)
        store.update(1, status="completed")

        result = store.update(2, status="in_progress")

        assert result is not None
        assert result.status == TaskStatus.in_progress

    def test_dependency_persists(self, tmp_path: Path):
        persist = tmp_path / "tasks.json"
        store1 = TaskStore(persist_path=persist)
        store1.create("A")
        store1.create("B")
        store1.add_dependency(2, blocked_by=1)

        store2 = TaskStore(persist_path=persist)
        task2 = store2.get(2)
        assert task2 is not None
        assert 1 in task2.blocked_by

    def test_duplicate_dependency_is_idempotent(self):
        store = TaskStore()
        store.create("A")
        store.create("B")
        store.add_dependency(2, blocked_by=1)
        store.add_dependency(2, blocked_by=1)
        task2 = store.get(2)
        assert task2 is not None
        assert task2.blocked_by.count(1) == 1


# ---------------------------------------------------------------------------
# Wave 3: Task lifecycle hooks
# ---------------------------------------------------------------------------


class TestTaskLifecycleHooks:
    def test_on_task_completed_fires(self):
        completed_tasks: list[TaskRecord] = []
        store = TaskStore(on_task_completed=lambda t: completed_tasks.append(t))
        store.create("X")
        store.update(1, status="completed")

        assert len(completed_tasks) == 1
        assert completed_tasks[0].id == 1

    def test_on_task_completed_does_not_fire_for_non_completion(self):
        completed_tasks: list[TaskRecord] = []
        store = TaskStore(on_task_completed=lambda t: completed_tasks.append(t))
        store.create("X")
        store.update(1, status="in_progress")

        assert len(completed_tasks) == 0

    def test_on_all_tasks_completed_fires(self):
        all_done_calls: list[bool] = []
        store = TaskStore(on_all_tasks_completed=lambda: all_done_calls.append(True))
        store.create("A")
        store.create("B")
        store.update(1, status="completed")
        assert len(all_done_calls) == 0  # still have open tasks

        store.update(2, status="completed")
        assert len(all_done_calls) == 1

    def test_lifecycle_hooks_dont_crash_on_exception(self):
        def bad_hook(t: TaskRecord) -> None:
            raise RuntimeError("boom")

        store = TaskStore(on_task_completed=bad_hook)
        store.create("X")
        # Should not raise
        store.update(1, status="completed")


# ---------------------------------------------------------------------------
# Wave 3.2: Extended TaskRecord and TaskStore
# ---------------------------------------------------------------------------


class TestTaskRecordExtended:
    def test_new_fields_default(self):
        task = TaskRecord(id=1, subject="X")
        assert task.active_form == ""
        assert task.created_at == ""
        assert task.updated_at == ""

    def test_timestamps_set_on_create(self):
        store = TaskStore()
        task = store.create("Write tests")
        assert task.created_at != ""
        assert task.updated_at != ""
        assert "T" in task.created_at  # ISO 8601

    def test_updated_at_changes_on_update(self):
        store = TaskStore()
        task = store.create("X")
        original = task.updated_at
        import time
        time.sleep(0.01)
        store.update(1, status="in_progress")
        updated = store.get(1)
        assert updated.updated_at >= original

    def test_backward_compat_from_dict(self):
        """Old JSON without new fields should load fine."""
        old_data = {"id": 1, "subject": "Old task", "status": "pending",
                    "description": "", "owner": "", "metadata": {}}
        task = TaskRecord.from_dict(old_data)
        assert task.active_form == ""
        assert task.created_at == ""
        assert task.updated_at == ""

    def test_active_form_round_trip(self):
        task = TaskRecord(id=1, subject="X", active_form="Running tests")
        d = task.to_dict()
        restored = TaskRecord.from_dict(d)
        assert restored.active_form == "Running tests"


class TestTaskOutput:
    def test_write_and_read(self, tmp_path: Path):
        store = TaskStore(task_list_id="session-1", task_root=tmp_path / "tasks")
        store.create("Work")
        path = store.write_task_output(1, "Result content")
        assert path.exists()
        assert store.read_task_output(1) == "Result content"

    def test_read_nonexistent(self, tmp_path: Path):
        store = TaskStore(task_list_id="session-1", task_root=tmp_path / "tasks")
        store.create("Work")
        assert store.read_task_output(1) is None

    def test_read_no_outputs_dir(self):
        store = TaskStore()
        assert store.read_task_output(1) is None

    def test_mark_notified(self, tmp_path: Path):
        store = TaskStore(task_list_id="session-1", task_root=tmp_path / "tasks")
        store.create("Work")
        store.mark_notified(1)
        assert store.get(1).metadata.get("notified") is True


class TestPerTaskListStorage:
    def test_storage_layout(self, tmp_path: Path):
        store = TaskStore(task_list_id="sess-abc", task_root=tmp_path / "tasks")
        store.create("Task A")
        assert store.persist_path == tmp_path / "tasks" / "sess-abc" / "tasks.json"
        assert store.outputs_dir == tmp_path / "tasks" / "sess-abc" / "outputs"
        assert store.persist_path.exists()

    def test_resume_loads_from_disk(self, tmp_path: Path):
        store1 = TaskStore(task_list_id="sess-1", task_root=tmp_path / "tasks")
        store1.create("Persistent task")
        store1.update(1, status="in_progress")

        store2 = TaskStore(task_list_id="sess-1", task_root=tmp_path / "tasks")
        assert store2.get(1) is not None
        assert store2.get(1).subject == "Persistent task"
        assert store2.get(1).status == TaskStatus.in_progress
