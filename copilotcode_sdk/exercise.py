"""Integration exercise system for copilotcode_sdk.

Builds a self-aware prompt that instructs an LLM agent to systematically
exercise every subsystem of the SDK, observe results, and return a
structured JSON report.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

if TYPE_CHECKING:
    from .client import CopilotCodeClient


SubsystemStatus = Literal["pass", "fail", "skip", "error"]


@dataclass
class SubsystemResult:
    """Result of exercising a single SDK subsystem."""

    name: str
    status: SubsystemStatus
    detail: str
    duration_seconds: float = 0.0
    error: str | None = None


@dataclass
class ExerciseReport:
    """Structured report from a full exercise run."""

    product_name: str
    session_id: str
    timestamp: str
    subsystems: list[SubsystemResult] = field(default_factory=list)
    summary: str = ""
    total_duration_seconds: float = 0.0

    @property
    def passed(self) -> int:
        return sum(1 for s in self.subsystems if s.status == "pass")

    @property
    def failed(self) -> int:
        return sum(1 for s in self.subsystems if s.status == "fail")

    @property
    def errored(self) -> int:
        return sum(1 for s in self.subsystems if s.status == "error")

    @property
    def skipped(self) -> int:
        return sum(1 for s in self.subsystems if s.status == "skip")

    @property
    def ok(self) -> bool:
        return self.failed == 0 and self.errored == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_name": self.product_name,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "ok": self.ok,
            "passed": self.passed,
            "failed": self.failed,
            "errored": self.errored,
            "skipped": self.skipped,
            "total_duration_seconds": self.total_duration_seconds,
            "summary": self.summary,
            "subsystems": [asdict(s) for s in self.subsystems],
        }

    def to_text(self) -> str:
        lines = [
            f"Exercise Report — {self.product_name}",
            f"session: {self.session_id}",
            f"timestamp: {self.timestamp}",
            f"ok: {self.ok}",
            f"passed: {self.passed}  failed: {self.failed}  "
            f"errored: {self.errored}  skipped: {self.skipped}",
            f"duration: {self.total_duration_seconds:.1f}s",
            "",
            "Subsystems:",
        ]
        for s in self.subsystems:
            icon = {"pass": "+", "fail": "!", "error": "X", "skip": "-"}.get(
                s.status, "?"
            )
            lines.append(f"  [{icon}] {s.name}: {s.detail}")
            if s.error:
                lines.append(f"      error: {s.error}")
        if self.summary:
            lines.append("")
            lines.append(f"Summary: {self.summary}")
        return "\n".join(lines)


SUBSYSTEM_CHECKLIST: list[dict[str, str]] = [
    {
        "name": "prompt_compiler",
        "description": (
            "from copilotcode_sdk.prompt_compiler import PromptAssembler, PromptPriority; "
            "a = PromptAssembler(); a.add('test', 'hello', priority=PromptPriority.default); "
            "result = a.render(); assert 'hello' in result"
        ),
    },
    {
        "name": "session_state",
        "description": (
            "from copilotcode_sdk.session_state import SessionState, SessionStatus; "
            "s = SessionState(status=SessionStatus.idle); "
            "s.status = SessionStatus.running; assert s.status == SessionStatus.running"
        ),
    },
    {
        "name": "tasks",
        "description": (
            "import tempfile; from pathlib import Path; "
            "from copilotcode_sdk.tasks import TaskStore, TaskStatus; "
            "store = TaskStore(persist_path=Path(tempfile.mkdtemp()) / 'tasks.json'); "
            "t = store.create('test task'); store.update(t.id, status=TaskStatus.in_progress); "
            "assert len(store.list_open()) >= 1"
        ),
    },
    {
        "name": "memory",
        "description": (
            "import tempfile; from pathlib import Path; "
            "from copilotcode_sdk.memory import MemoryStore; "
            "d = Path(tempfile.mkdtemp()); store = MemoryStore(d, d / '.mem'); "
            "store.upsert_memory(title='test', content='hello world', description='exercise test', memory_type='project'); "
            "records = store.list_records(); assert any(r.name == 'test' for r in records)"
        ),
    },
    {
        "name": "compaction",
        "description": (
            "from copilotcode_sdk.compaction import build_compaction_prompt, parse_compaction_response, "
            "format_transcript_for_compaction; "
            "p = build_compaction_prompt(); assert 'Primary request' in p; "
            "r = parse_compaction_response('<summary>test</summary>'); assert r.summary == 'test'; "
            "t = format_transcript_for_compaction([{'role':'user','content':'hi'}]); assert '[user]' in t"
        ),
    },
    {
        "name": "extraction",
        "description": (
            "import tempfile; from pathlib import Path; "
            "from copilotcode_sdk.extraction import build_extraction_prompt, should_extract; "
            "d = Path(tempfile.mkdtemp()); "
            "p = build_extraction_prompt(memory_dir=str(d), project_root=str(d)); assert len(p) > 0; "
            "assert should_extract(tool_call_count=25, total_chars=60000, last_extraction_turn=0, current_turn=30) == True; "
            "assert should_extract(tool_call_count=1, total_chars=100, last_extraction_turn=0, current_turn=1) == False"
        ),
    },
    {
        "name": "session_memory",
        "description": (
            "import tempfile; from pathlib import Path; "
            "from copilotcode_sdk.memory import MemoryStore; "
            "from copilotcode_sdk.session_memory import SessionMemoryController; "
            "d = Path(tempfile.mkdtemp()); store = MemoryStore(d, d / '.mem'); "
            "ctrl = SessionMemoryController(store); "
            "assert ctrl.state.initialized == False"
        ),
    },
    {
        "name": "events",
        "description": (
            "from copilotcode_sdk.events import EventBus, Event, EventType; "
            "bus = EventBus(); received = []; "
            "bus.subscribe(lambda e: received.append(e), event_type=EventType.tool_called); "
            "bus.emit(Event(type=EventType.tool_called, data={'tool': 'test'})); "
            "assert len(received) == 1"
        ),
    },
    {
        "name": "diff",
        "description": (
            "from copilotcode_sdk.diff import generate_diff; "
            "result = generate_diff('hello', 'hello world'); "
            "assert result.changed"
        ),
    },
    {
        "name": "tokenizer",
        "description": (
            "from copilotcode_sdk.tokenizer import estimate_tokens; "
            "count = estimate_tokens('hello world'); assert count > 0"
        ),
    },
    {
        "name": "retry",
        "description": (
            "from copilotcode_sdk.retry import RetryPolicy, RetryState, build_retry_response; "
            "policy = RetryPolicy(); state = RetryState(policy); "
            "resp = build_retry_response(state, error_context='test error'); "
            "assert resp is not None"
        ),
    },
    {
        "name": "suggestions",
        "description": (
            "from copilotcode_sdk.suggestions import build_prompt_suggestions; "
            "result = build_prompt_suggestions(session_turn=3); "
            "assert isinstance(result, list)"
        ),
    },
    {
        "name": "skill_assets",
        "description": (
            "import tempfile; from pathlib import Path; "
            "from copilotcode_sdk.skill_assets import parse_skill_frontmatter; "
            "d = Path(tempfile.mkdtemp()); p = d / 'SKILL.md'; "
            "p.write_text('---\\nname: test\\ndescription: a test\\n---\\n# Test'); "
            "fm = parse_skill_frontmatter(p); assert fm['name'] == 'test'"
        ),
    },
    {
        "name": "permissions",
        "description": (
            "from copilotcode_sdk.permissions import PermissionPolicy; "
            "assert 'safe' in PermissionPolicy.__args__; "
            "assert 'approve_all' in PermissionPolicy.__args__"
        ),
    },
    {
        "name": "model_cost",
        "description": (
            "from copilotcode_sdk.model_cost import calculate_cost, UsageCost; "
            "cost = calculate_cost(model='claude-sonnet-4-20250514', input_tokens=1000, output_tokens=500); "
            "assert isinstance(cost, UsageCost); assert cost.total > 0"
        ),
    },
    {
        "name": "config",
        "description": (
            "from copilotcode_sdk.config import CopilotCodeConfig; "
            "cfg = CopilotCodeConfig(); "
            "assert cfg.permission_policy == 'safe'"
        ),
    },
    {
        "name": "instructions",
        "description": (
            "import tempfile; from pathlib import Path; "
            "from copilotcode_sdk.instructions import load_workspace_instructions; "
            "d = Path(tempfile.mkdtemp()); (d / 'CLAUDE.md').write_text('# Test instructions'); "
            "bundle = load_workspace_instructions(d); "
            "assert 'Test instructions' in bundle.content"
        ),
    },
]


def build_exercise_prompt(
    checklist: Sequence[dict[str, str]] | None = None,
) -> str:
    """Build the system-aware prompt the exercising agent receives."""
    items = checklist if checklist is not None else SUBSYSTEM_CHECKLIST
    checklist_lines = []
    for i, item in enumerate(items, 1):
        code = item["description"].replace("; ", "\n")
        checklist_lines.append(
            f"### {i}. {item['name']}\n```python\n{code}\n```"
        )

    checklist_block = "\n\n".join(checklist_lines)

    return f"""\
You are a self-aware integration exercise agent for copilotcode_sdk.

Your job is to systematically exercise every subsystem listed below by running the exact Python code provided for each one. Execute each code snippet, observe whether it succeeds, and record the result.

## Subsystem Checklist

{checklist_block}

## Instructions

- Run each code snippet above IN ORDER using Python execution.
- Do NOT just read files or inspect source. Actually EXECUTE the code.
- For each, report: name, status (pass/fail/skip/error), a brief detail of what you observed, and any error message.
- If a snippet raises an exception, record it as "fail" or "error" with the traceback.
- After exercising all subsystems, write a brief summary of overall results.

## Output Format

Return your complete results as JSON inside <exercise-report> tags. The JSON should have this structure:

```json
{{
  "subsystems": [
    {{
      "name": "subsystem_name",
      "status": "pass|fail|skip|error",
      "detail": "what you observed",
      "duration_seconds": 0.0,
      "error": null
    }}
  ],
  "summary": "brief overall summary"
}}
```

<exercise-report>
YOUR JSON HERE
</exercise-report>
"""


def parse_exercise_report(
    raw_text: str,
    *,
    product_name: str = "CopilotCode",
    session_id: str = "",
    timestamp: str = "",
    total_duration_seconds: float = 0.0,
) -> ExerciseReport:
    """Parse the agent's structured response into an ExerciseReport."""
    # Try to extract from <exercise-report> tags first
    match = re.search(
        r"<exercise-report>\s*(.*?)\s*</exercise-report>",
        raw_text,
        re.DOTALL,
    )
    json_text = match.group(1) if match else raw_text.strip()

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        # Try to find any JSON object in the text
        json_match = re.search(r"\{[\s\S]*\}", json_text)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                return ExerciseReport(
                    product_name=product_name,
                    session_id=session_id,
                    timestamp=timestamp,
                    total_duration_seconds=total_duration_seconds,
                    summary=f"Failed to parse exercise report from agent response.",
                    subsystems=[
                        SubsystemResult(
                            name="parse_error",
                            status="error",
                            detail="Could not parse JSON from agent response.",
                            error=raw_text[:500],
                        ),
                    ],
                )
        else:
            return ExerciseReport(
                product_name=product_name,
                session_id=session_id,
                timestamp=timestamp,
                total_duration_seconds=total_duration_seconds,
                summary="Failed to parse exercise report from agent response.",
                subsystems=[
                    SubsystemResult(
                        name="parse_error",
                        status="error",
                        detail="No JSON found in agent response.",
                        error=raw_text[:500],
                    ),
                ],
            )

    subsystems = []
    for item in data.get("subsystems", []):
        subsystems.append(
            SubsystemResult(
                name=item.get("name", "unknown"),
                status=item.get("status", "error"),
                detail=item.get("detail", ""),
                duration_seconds=float(item.get("duration_seconds", 0.0)),
                error=item.get("error"),
            ),
        )

    return ExerciseReport(
        product_name=product_name,
        session_id=session_id,
        timestamp=timestamp,
        total_duration_seconds=total_duration_seconds,
        summary=data.get("summary", ""),
        subsystems=subsystems,
    )


def _extract_last_assistant_text(messages: list[Any]) -> str:
    """Extract text from the last assistant message in a message list."""
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            # Try normalizing SDK objects
            to_dict = getattr(msg, "to_dict", None)
            if callable(to_dict):
                msg = to_dict()
            else:
                try:
                    msg = vars(msg)
                except TypeError:
                    continue
        role = str(msg.get("role") or msg.get("type") or "")
        if "assistant" not in role:
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            text = " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
            if text.strip():
                return text
        data = msg.get("data")
        if isinstance(data, dict):
            nested = data.get("content", "")
            if isinstance(nested, str) and nested.strip():
                return nested
    return ""


async def run_exercise(
    client: "CopilotCodeClient",
    *,
    timeout: float = 600.0,
    subsystems: Sequence[str] | None = None,
    save_report_path: str | Path | None = None,
) -> ExerciseReport:
    """Create a real session, send the exercise prompt, parse the report.

    This is the full orchestration entry point called by the CLI.
    """
    from datetime import datetime, timezone

    session_id = f"{client.config.brand.slug}-exercise-{uuid.uuid4().hex[:12]}"
    timestamp = datetime.now(timezone.utc).isoformat()

    # Filter checklist if specific subsystems requested
    checklist: list[dict[str, str]] | None = None
    if subsystems:
        checklist = [
            item
            for item in SUBSYSTEM_CHECKLIST
            if item["name"] in subsystems
        ]

    prompt = build_exercise_prompt(checklist)

    start = time.monotonic()
    session = None
    try:
        session = await client.create_session(session_id=session_id)
        await session.send_and_wait(prompt, timeout=timeout)
        response_text = _extract_last_assistant_text(
            await session.get_messages()
        )
        elapsed = time.monotonic() - start

        report = parse_exercise_report(
            response_text,
            product_name=client.config.brand.public_name,
            session_id=session_id,
            timestamp=timestamp,
            total_duration_seconds=elapsed,
        )
    except Exception as exc:
        elapsed = time.monotonic() - start
        report = ExerciseReport(
            product_name=client.config.brand.public_name,
            session_id=session_id,
            timestamp=timestamp,
            total_duration_seconds=elapsed,
            summary=f"Exercise failed with error: {exc}",
            subsystems=[
                SubsystemResult(
                    name="session_error",
                    status="error",
                    detail="Exercise session failed.",
                    error=str(exc),
                ),
            ],
        )

    if save_report_path:
        path = Path(save_report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.to_dict(), indent=2))

    return report
