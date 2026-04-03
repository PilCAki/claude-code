"""Task v2 store — in-memory with single-file JSON persistence."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


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

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskRecord:
        data = dict(data)
        data["status"] = TaskStatus(data.get("status", "pending"))
        return cls(**{k: v for k, v in data.items() if k in cls.__slots__})


class TaskStore:
    """In-memory task store with optional JSON persistence."""

    def __init__(self, persist_path: Path | None = None) -> None:
        self._tasks: dict[int, TaskRecord] = {}
        self._next_id = 1
        self._persist_path = persist_path
        if persist_path and persist_path.exists():
            self._load()

    @property
    def persist_path(self) -> Path | None:
        return self._persist_path

    def create(
        self,
        subject: str,
        *,
        description: str = "",
        owner: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TaskRecord:
        task = TaskRecord(
            id=self._next_id,
            subject=subject,
            description=description,
            owner=owner,
            metadata=metadata or {},
        )
        self._tasks[task.id] = task
        self._next_id += 1
        self._save()
        return task

    def get(self, task_id: int) -> TaskRecord | None:
        return self._tasks.get(task_id)

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
        if status is not None:
            task.status = TaskStatus(status) if isinstance(status, str) else status
        if subject is not None:
            task.subject = subject
        if description is not None:
            task.description = description
        if owner is not None:
            task.owner = owner
        if metadata is not None:
            task.metadata.update(metadata)
        self._save()
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
