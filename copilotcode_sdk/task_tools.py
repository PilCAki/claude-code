"""Task v2 custom tools for the Copilot SDK."""
from __future__ import annotations

import json
from typing import Any

from .tasks import TaskStatus, TaskStore


def build_task_tools(store: TaskStore) -> list[Any]:
    """Build SDK Tool objects for task management.

    Returns a list of ``copilot.types.Tool`` instances ready to pass
    into ``create_session(tools=...)``.
    """
    from copilot.types import Tool, ToolInvocation, ToolResult

    def _task_create(invocation: ToolInvocation) -> ToolResult:
        args = invocation.arguments or {}
        subject = args.get("subject", "").strip()
        if not subject:
            return ToolResult(
                text_result_for_llm="Error: 'subject' is required.",
                result_type="error",
            )
        task = store.create(
            subject=subject,
            description=args.get("description", ""),
            owner=args.get("owner", ""),
            metadata=args.get("metadata") or {},
        )
        return ToolResult(
            text_result_for_llm=(
                f"Task #{task.id} created: {task.subject}\n"
                "Use TaskList to see all open tasks and find the next one to work on."
            ),
        )

    def _task_update(invocation: ToolInvocation) -> ToolResult:
        args = invocation.arguments or {}
        task_id = args.get("task_id")
        if task_id is None:
            return ToolResult(
                text_result_for_llm="Error: 'task_id' is required.",
                result_type="error",
            )
        try:
            task_id = int(task_id)
        except (TypeError, ValueError):
            return ToolResult(
                text_result_for_llm=f"Error: invalid task_id '{task_id}'.",
                result_type="error",
            )

        # Handle deletion
        if args.get("action") == "deleted":
            ok = store.delete(task_id)
            if not ok:
                return ToolResult(
                    text_result_for_llm=f"Error: task #{task_id} not found.",
                    result_type="error",
                )
            return ToolResult(text_result_for_llm=f"Task #{task_id} deleted.")

        status = args.get("status")
        if status is not None:
            try:
                TaskStatus(status)
            except ValueError:
                return ToolResult(
                    text_result_for_llm=f"Error: invalid status '{status}'. Use: pending, in_progress, completed.",
                    result_type="error",
                )

        task = store.update(
            task_id,
            status=status,
            subject=args.get("subject"),
            description=args.get("description"),
            owner=args.get("owner"),
            metadata=args.get("metadata"),
        )
        if task is None:
            return ToolResult(
                text_result_for_llm=f"Error: task #{task_id} not found.",
                result_type="error",
            )

        result = f"Task #{task.id} updated: [{task.status.value}] {task.subject}"
        if task.status == TaskStatus.completed:
            open_tasks = store.list_open()
            pending = [t for t in open_tasks if t.status == TaskStatus.pending]
            if pending:
                result += f"\n\n{len(pending)} task(s) remaining. Use TaskList to find next work."
            else:
                result += "\n\nAll tasks complete!"
        return ToolResult(text_result_for_llm=result)

    def _task_list(invocation: ToolInvocation) -> ToolResult:
        tasks = store.list_open()
        if not tasks:
            return ToolResult(text_result_for_llm="No open tasks.")

        lines = []
        for t in tasks:
            status = t.status.value.replace("_", " ")
            owner = f" (owner: {t.owner})" if t.owner else ""
            lines.append(f"#{t.id} [{status}]{owner}: {t.subject}")
        return ToolResult(text_result_for_llm="\n".join(lines))

    def _task_get(invocation: ToolInvocation) -> ToolResult:
        args = invocation.arguments or {}
        task_id = args.get("task_id")
        if task_id is None:
            return ToolResult(
                text_result_for_llm="Error: 'task_id' is required.",
                result_type="error",
            )
        try:
            task_id = int(task_id)
        except (TypeError, ValueError):
            return ToolResult(
                text_result_for_llm=f"Error: invalid task_id '{task_id}'.",
                result_type="error",
            )
        task = store.get(task_id)
        if task is None:
            return ToolResult(
                text_result_for_llm=f"Error: task #{task_id} not found.",
                result_type="error",
            )
        return ToolResult(
            text_result_for_llm=json.dumps(task.to_dict(), indent=2),
        )

    tools = [
        Tool(
            name="TaskCreate",
            description=(
                "Create a new task to track work. Use this when starting multi-step work "
                "or when the user gives you a task with 3 or more steps."
            ),
            handler=_task_create,
            parameters={
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "Short summary of the task (1-2 sentences).",
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description of what needs to be done.",
                    },
                    "owner": {
                        "type": "string",
                        "description": "Who is responsible for this task.",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Arbitrary key-value metadata.",
                    },
                },
                "required": ["subject"],
            },
            skip_permission=True,
        ),
        Tool(
            name="TaskUpdate",
            description=(
                "Update an existing task's status, details, or owner. "
                "Mark tasks in_progress when starting, completed when done."
            ),
            handler=_task_update,
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "integer",
                        "description": "ID of the task to update.",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed"],
                        "description": "New status for the task.",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Updated subject line.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Updated description.",
                    },
                    "owner": {
                        "type": "string",
                        "description": "Updated owner.",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["deleted"],
                        "description": "Set to 'deleted' to remove this task.",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Metadata to merge into the task.",
                    },
                },
                "required": ["task_id"],
            },
            skip_permission=True,
        ),
        Tool(
            name="TaskList",
            description=(
                "List all open tasks. Shows pending and in-progress tasks ordered by ID. "
                "Use after completing a task to find the next one."
            ),
            handler=_task_list,
            parameters={
                "type": "object",
                "properties": {},
            },
            skip_permission=True,
        ),
        Tool(
            name="TaskGet",
            description="Get full details for a specific task by ID.",
            handler=_task_get,
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "integer",
                        "description": "ID of the task to retrieve.",
                    },
                },
                "required": ["task_id"],
            },
            skip_permission=True,
        ),
    ]
    return tools
