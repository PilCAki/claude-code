"""Task v2 store — in-memory with single-file JSON persistence."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class TaskStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    deleted = "deleted"


@dataclass(slots=True)
class TaskRecord:
    id: int
    subject: str
    status: TaskStatus = TaskStatus.pending
    description: str = ""
    owner: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    blocked_by: list[int] = field(default_factory=list)
    blocks: list[int] = field(default_factory=list)
    active_form: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskRecord:
        data = dict(data)
        data["status"] = TaskStatus(data.get("status", "pending"))
        data.setdefault("blocked_by", [])
        data.setdefault("blocks", [])
        data.setdefault("active_form", "")
        data.setdefault("created_at", "")
        data.setdefault("updated_at", "")
        return cls(**{k: v for k, v in data.items() if k in cls.__slots__})


class TaskStore:
    """In-memory task store with optional JSON persistence."""

    def __init__(
        self,
        persist_path: Path | None = None,
        *,
        task_list_id: str | None = None,
        task_root: Path | None = None,
        on_task_completed: Callable[[TaskRecord], None] | None = None,
        on_all_tasks_completed: Callable[[], None] | None = None,
    ) -> None:
        self._tasks: dict[int, TaskRecord] = {}
        self._next_id = 1
        self._task_list_id = task_list_id
        self._outputs_dir: Path | None = None

        # Per-task-list storage: derive paths from task_list_id + task_root
        if task_list_id is not None and task_root is not None:
            list_dir = Path(task_root) / task_list_id
            persist_path = list_dir / "tasks.json"
            self._outputs_dir = list_dir / "outputs"

        self._persist_path = persist_path
        self._on_task_completed = on_task_completed
        self._on_all_tasks_completed = on_all_tasks_completed
        if persist_path and persist_path.exists():
            self._load()

    @property
    def persist_path(self) -> Path | None:
        return self._persist_path

    @property
    def task_list_id(self) -> str | None:
        return self._task_list_id

    @property
    def outputs_dir(self) -> Path | None:
        return self._outputs_dir

    def write_task_output(self, task_id: int, content: str) -> Path:
        """Write output artifact for a task. Returns the output file path."""
        if self._outputs_dir is None:
            raise ValueError("No outputs directory configured (set task_list_id and task_root)")
        self._outputs_dir.mkdir(parents=True, exist_ok=True)
        path = self._outputs_dir / f"{task_id}.txt"
        path.write_text(content, encoding="utf-8")
        return path

    def read_task_output(self, task_id: int) -> str | None:
        """Read output artifact for a task. Returns None if not found."""
        if self._outputs_dir is None:
            return None
        path = self._outputs_dir / f"{task_id}.txt"
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    def mark_notified(self, task_id: int) -> None:
        """Mark a task as having had its output read."""
        task = self._tasks.get(task_id)
        if task is not None:
            task.metadata["notified"] = True
            self._save()

    def create(
        self,
        subject: str,
        *,
        description: str = "",
        owner: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TaskRecord:
        now = datetime.now(timezone.utc).isoformat()
        task = TaskRecord(
            id=self._next_id,
            subject=subject,
            description=description,
            owner=owner,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )
        self._tasks[task.id] = task
        self._next_id += 1
        self._save()
        return task

    def get(self, task_id: int) -> TaskRecord | None:
        return self._tasks.get(task_id)

    def add_dependency(self, task_id: int, blocked_by: int) -> bool:
        """Mark *task_id* as blocked by *blocked_by*.

        Sets up both sides of the relationship.  Returns ``False`` if
        either task doesn't exist or the dependency would be circular.
        """
        task = self._tasks.get(task_id)
        blocker = self._tasks.get(blocked_by)
        if task is None or blocker is None:
            return False
        if task_id == blocked_by:
            return False
        # Circular check: blocker must not be (transitively) blocked by task_id
        if self._is_blocked_by(blocked_by, task_id):
            return False
        if blocked_by not in task.blocked_by:
            task.blocked_by.append(blocked_by)
        if task_id not in blocker.blocks:
            blocker.blocks.append(task_id)
        self._save()
        return True

    def blockers_incomplete(self, task_id: int) -> list[int]:
        """Return IDs of incomplete blockers for *task_id*."""
        task = self._tasks.get(task_id)
        if task is None:
            return []
        return [
            bid for bid in task.blocked_by
            if bid in self._tasks
            and self._tasks[bid].status not in (TaskStatus.completed, TaskStatus.deleted)
        ]

    def _is_blocked_by(self, task_id: int, target_id: int) -> bool:
        """Check if *task_id* is transitively blocked by *target_id*."""
        visited: set[int] = set()
        stack = [task_id]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            t = self._tasks.get(current)
            if t is None:
                continue
            for bid in t.blocked_by:
                if bid == target_id:
                    return True
                stack.append(bid)
        return False

    def update(
        self,
        task_id: int,
        *,
        status: TaskStatus | str | None = None,
        subject: str | None = None,
        description: str | None = None,
        owner: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskRecord | None:
        task = self._tasks.get(task_id)
        if task is None:
            return None

        # Dependency enforcement: cannot start if blockers are incomplete
        if status is not None:
            new_status = TaskStatus(status) if isinstance(status, str) else status
            if new_status == TaskStatus.in_progress:
                incomplete = self.blockers_incomplete(task_id)
                if incomplete:
                    # Store the rejection reason in metadata for callers to inspect
                    task.metadata["_blocked_reason"] = (
                        f"Cannot start: blocked by incomplete tasks {incomplete}"
                    )
                    self._save()
                    return task
            task.status = new_status
        if subject is not None:
            task.subject = subject
        if description is not None:
            task.description = description
        if owner is not None:
            task.owner = owner
        if metadata is not None:
            task.metadata.update(metadata)
        task.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()

        # Fire lifecycle hooks
        if task.status == TaskStatus.completed:
            if self._on_task_completed is not None:
                try:
                    self._on_task_completed(task)
                except Exception:
                    pass
            if not self.has_open_tasks() and self._on_all_tasks_completed is not None:
                try:
                    self._on_all_tasks_completed()
                except Exception:
                    pass

        return task

    def delete(self, task_id: int) -> bool:
        task = self._tasks.get(task_id)
        if task is None:
            return False
        task.status = TaskStatus.deleted
        self._save()
        return True

    def list_open(self) -> list[TaskRecord]:
        """Return non-deleted tasks ordered by ID (lowest first)."""
        return sorted(
            (t for t in self._tasks.values() if t.status != TaskStatus.deleted),
            key=lambda t: t.id,
        )

    def list_all(self) -> list[TaskRecord]:
        return sorted(self._tasks.values(), key=lambda t: t.id)

    def has_open_tasks(self) -> bool:
        return any(
            t.status in (TaskStatus.pending, TaskStatus.in_progress)
            for t in self._tasks.values()
        )

    def summary_text(self) -> str:
        """Compact text summary of open tasks for context injection."""
        open_tasks = self.list_open()
        if not open_tasks:
            return ""
        lines = ["**Open tasks:**"]
        for t in open_tasks:
            status = t.status.value.replace("_", " ")
            owner_part = f" (owner: {t.owner})" if t.owner else ""
            lines.append(f"- #{t.id} [{status}]{owner_part}: {t.subject}")
        return "\n".join(lines)

    def _save(self) -> None:
        if self._persist_path is None:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "next_id": self._next_id,
            "tasks": [t.to_dict() for t in self._tasks.values()],
        }
        self._persist_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load(self) -> None:
        if self._persist_path is None or not self._persist_path.exists():
            return
        try:
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        self._next_id = data.get("next_id", 1)
        for raw in data.get("tasks", []):
            task = TaskRecord.from_dict(raw)
            self._tasks[task.id] = task
            if task.id >= self._next_id:
                self._next_id = task.id + 1
