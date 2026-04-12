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
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

if TYPE_CHECKING:
    from .client import CopilotCodeClient
    from .config import CopilotCodeConfig


SubsystemStatus = Literal["pass", "fail", "skip", "error"]


@dataclass
class SubsystemResult:
    """Result of exercising a single SDK subsystem."""

    name: str
    status: SubsystemStatus
    detail: str
    duration_seconds: float = 0.0
    error: str | None = None
    token_usage: dict[str, int] = field(default_factory=dict)
    cost: float = 0.0


@dataclass
class ExerciseReport:
    """Structured report from a full exercise run."""

    product_name: str
    session_id: str
    timestamp: str
    mode: str = "subsystem"
    subsystems: list[SubsystemResult] = field(default_factory=list)
    summary: str = ""
    total_duration_seconds: float = 0.0
    ground_truth: dict[str, Any] | None = None

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
        d: dict[str, Any] = {
            "product_name": self.product_name,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "ok": self.ok,
            "passed": self.passed,
            "failed": self.failed,
            "errored": self.errored,
            "skipped": self.skipped,
            "total_duration_seconds": self.total_duration_seconds,
            "summary": self.summary,
            "subsystems": [asdict(s) for s in self.subsystems],
        }
        if self.ground_truth is not None:
            d["ground_truth"] = self.ground_truth
        return d

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


ExerciseMode = Literal["subsystem", "orchestration", "advanced", "cascade", "micro", "chain", "full"]


ORCHESTRATION_SCENARIOS: list[dict[str, str]] = [
    {
        "name": "event_bus_lifecycle",
        "mission": (
            "Every tool call you make inside this session emits events on the SDK's "
            "internal event bus. Your mission: call a few tools (read a file, run a "
            "harmless bash command like `echo hello`), then observe what happens. "
            "The system should be tracking your activity. Report whether you received "
            "any system-injected context or responses that indicate the session machinery "
            "is running (e.g., context from hooks, system reminders, or any behavior "
            "beyond raw tool results)."
        ),
        "expected": (
            "Tool calls succeed. The session is alive and responsive. Any "
            "additionalContext injections from hooks indicate the machinery is wired."
        ),
    },
    {
        "name": "file_change_tracking",
        "mission": (
            "Write a small temp file (e.g., `/tmp/copilotcode_exercise_test.txt` with "
            "content 'exercise test'). Then read it back. The SDK's hooks track file "
            "changes — writing a file should register as a 'created' change internally. "
            "Report whether both operations succeeded and the content round-tripped correctly."
        ),
        "expected": (
            "File write succeeds. File read returns the same content. Both operations "
            "complete without errors, proving the tool pipeline works end-to-end."
        ),
    },
    {
        "name": "git_context_staleness",
        "mission": (
            "Run `git status` via bash — this is a safe, read-only git command. "
            "Then run `git log --oneline -3` via bash. These are read-only and should "
            "NOT trigger the SDK's git-context-staleness detection (which only fires on "
            "mutating commands like git checkout, commit, merge, rebase, pull, switch). "
            "Report the results of both commands and whether any unexpected context "
            "refresh or staleness warning appeared."
        ),
        "expected": (
            "Both git commands succeed and return normal output. No staleness warning "
            "or context refresh should appear because these are read-only commands."
        ),
    },
    {
        "name": "tool_call_accumulation",
        "mission": (
            "Make at least 5 tool calls in sequence: read this file (exercise.py in "
            "the copilotcode_sdk directory), run `echo 1`, run `echo 2`, run `echo 3`, "
            "and read another small file. After each call, note whether any system "
            "context was injected (extraction reminders, context warnings, suggestions). "
            "The SDK's hooks count tool calls and may inject guidance after thresholds "
            "are crossed. Report the total number of tool calls you made and any "
            "system-injected messages you observed."
        ),
        "expected": (
            "All tool calls succeed. The agent can count them. Any system-injected "
            "context (extraction reminders, warnings) indicates hook thresholds are "
            "being tracked."
        ),
    },
    {
        "name": "multi_turn_conversation",
        "mission": (
            "This scenario tests that the session maintains state across your actions. "
            "First, read the file `copilotcode_sdk/config.py`. Note a specific detail "
            "from its content (e.g., a field name or default value). Then run `echo "
            "'state check'` via bash. Finally, recall the detail you noted from the "
            "file — prove that your conversational context is intact across tool calls. "
            "Report whether the session maintained coherence."
        ),
        "expected": (
            "The agent can recall details from earlier tool results, proving the session "
            "maintains conversational context across multiple tool calls."
        ),
    },
    {
        "name": "error_handling",
        "mission": (
            "Deliberately trigger a tool error: try to read a file that doesn't exist "
            "(e.g., `/nonexistent/path/file.txt`). Observe how the session handles it. "
            "Does the error come back cleanly? Does the session remain functional "
            "afterward? Then read a real file to confirm recovery. Report the error "
            "message and whether the session recovered."
        ),
        "expected": (
            "The nonexistent file read returns a clear error. The session does NOT "
            "crash. The subsequent real file read succeeds, proving error recovery."
        ),
    },
]


ADVANCED_ORCHESTRATION_SCENARIOS: list[dict[str, str]] = [
    {
        "name": "extraction_nudge_trigger",
        "mission": (
            "This session uses LOWERED extraction thresholds (5 tool calls instead of "
            "20). Your mission: make at least 5 tool calls in sequence — read "
            "`copilotcode_sdk/config.py`, run `echo one`, run `echo two`, run `echo three`, "
            "and read `copilotcode_sdk/__init__.py`. After each call, watch carefully for "
            "any system-injected `additionalContext` mentioning 'memory checkpoint', "
            "'extraction', or 'save'. Report whether an extraction nudge appeared and "
            "what it said."
        ),
        "expected": (
            "After 5+ tool calls, the hook should inject an extraction nudge via "
            "additionalContext mentioning memory checkpoint or save. Ground truth: "
            "tool_call_count >= 5."
        ),
    },
    {
        "name": "git_safety_gate",
        "mission": (
            "Attempt a DANGEROUS git command that the SDK should block: run "
            "`git push --force origin main` via bash. Observe the response — it should "
            "be denied or blocked with an error about destructive/force operations. "
            "Then run `git status` via bash to confirm the session is still functional. "
            "Report the denial message and whether recovery succeeded."
        ),
        "expected": (
            "The `git push --force` command is denied/blocked by the safety gate. "
            "The subsequent `git status` succeeds normally. Ground truth: event bus "
            "has a tool_denied event."
        ),
    },
    {
        "name": "tool_result_caching",
        "mission": (
            "This session has tool result caching ENABLED. Your mission: (1) Read the "
            "file `copilotcode_sdk/config.py` using the read tool. Note the content. "
            "(2) Read the EXACT same file again with identical arguments. Watch for any "
            "indication that the result was cached (faster response, 'cached' note, or "
            "identical content). (3) Write a small temp file `/tmp/copilotcode_cache_test.txt` "
            "with content 'invalidate'. (4) Read `copilotcode_sdk/config.py` a third time "
            "— the cache should have been invalidated by the write. Report observations "
            "for each step."
        ),
        "expected": (
            "Second read returns cached result. Third read (after write) returns fresh "
            "result because write invalidates the cache. Ground truth: "
            "tool_result_cache_size > 0 after step 2."
        ),
    },
    {
        "name": "task_reminder_injection",
        "mission": (
            "This session uses LOWERED task reminder thresholds (3 turns instead of 10). "
            "Your mission: (1) Create a task using the TaskCreate tool with subject "
            "'exercise test task' and description 'pending task for reminder test'. "
            "(2) Then make 4 non-task tool calls: run `echo a`, `echo b`, `echo c`, "
            "and read `copilotcode_sdk/__init__.py`. (3) After each non-task call, watch "
            "for a system-injected stale-task reminder in `additionalContext` mentioning "
            "open tasks or task tools. Report whether the reminder appeared and what it said."
        ),
        "expected": (
            "After 3+ non-task tool calls with an open task, the hook should inject a "
            "stale-task reminder via additionalContext. Ground truth: task store has open "
            "tasks; tool_call_count >= 3 past task creation."
        ),
    },
    {
        "name": "context_size_warning",
        "mission": (
            "This session uses a LOWERED context size limit (50,000 chars instead of "
            "800,000). Your mission: read several large files to push the estimated "
            "context size past 80%% of the limit (40,000 chars). Read these files in "
            "order: `copilotcode_sdk/hooks.py` (~30KB), then `copilotcode_sdk/client.py` "
            "(~40KB). After each read, watch for a context/compaction warning in "
            "`additionalContext` mentioning 'capacity', 'approaching', or 'exhausted'. "
            "Report whether the warning appeared and at which point."
        ),
        "expected": (
            "After reading enough content to exceed 80%% of the lowered 50K limit, a "
            "context warning should appear in additionalContext. Ground truth: "
            "estimated_context_chars > 40,000."
        ),
    },
    {
        "name": "read_before_write_enforcement",
        "mission": (
            "The SDK enforces reading a file before writing to it. Your mission: "
            "(1) Attempt to write content 'first write' to the file "
            "`/tmp/copilotcode_rbw_test.txt` using the write/edit tool WITHOUT reading "
            "it first. Observe whether you receive a warning about read-before-write. "
            "(2) Now read the file `/tmp/copilotcode_rbw_test.txt`. "
            "(3) Write to it again with content 'second write'. This time there should "
            "be NO read-before-write warning since you read the file. Report whether "
            "the first write was warned and the second was not."
        ),
        "expected": (
            "First write triggers a read-before-write warning. Second write (after "
            "reading) does not trigger the warning. Ground truth: read_file_state "
            "contains the file path after the read step."
        ),
    },
    {
        "name": "repeat_loop_detection",
        "mission": (
            "The SDK detects when the same shell command is run repeatedly with "
            "identical output. Your mission: run the exact command "
            "`echo repeat_test_sentinel` via bash. Then run the EXACT same command "
            "`echo repeat_test_sentinel` again. Watch for a loop-detection warning "
            "in the response or `additionalContext` about repeating the same command. "
            "Report what happened on each run."
        ),
        "expected": (
            "The second identical shell command triggers a loop-detection warning. "
            "Ground truth: recent_shell contains duplicate command signatures."
        ),
    },
]


CASCADE_ORCHESTRATION_SCENARIOS: list[dict[str, str]] = [
    {
        "name": "context_accounting_desync",
        "mission": (
            "This session has tool result caching ENABLED and a context limit of "
            "20,000 chars. Your mission: test whether cached results are counted "
            "toward context size estimation.\n"
            "(1) Read `copilotcode_sdk/config.py` (~6K chars) — this is tracked AND cached.\n"
            "(2) Read the EXACT same file again with identical arguments — cache hit.\n"
            "(3) Read it a THIRD time — still cached.\n"
            "After each read, watch for any context/compaction warning in "
            "`additionalContext`. With a 20K limit and 80%% threshold (16K), three "
            "reads of a 6K file would trigger a warning IF all three were tracked "
            "(18K > 16K). But if cache hits skip tracking, only the first read "
            "counts (6K < 16K) and NO warning should appear.\n"
            "Report whether a context warning appeared and after which read."
        ),
        "expected": (
            "No context warning fires because cached reads skip context tracking. "
            "Ground truth: estimated_context_chars ~6K (1x), not ~18K (3x). "
            "tool_result_cache_size > 0."
        ),
    },
    {
        "name": "injection_priority_collision",
        "mission": (
            "This session has extraction threshold at 3 tool calls AND context "
            "limit at 20,000 chars (warning at 80%% = 16K). Your mission: cross "
            "BOTH thresholds in the same tool call and observe which injection wins.\n"
            "(1) Run `echo collision_setup_1` — tool call #1, small output.\n"
            "(2) Run `echo collision_setup_2` — tool call #2, small output.\n"
            "(3) Read `copilotcode_sdk/hooks.py` (~30K chars) — tool call #3 "
            "crosses BOTH the extraction threshold (3 calls) AND context threshold "
            "(30K > 16K).\n"
            "After step 3, watch carefully: do you see a CONTEXT WARNING "
            "('capacity', 'approaching', 'exhausted') or an EXTRACTION NUDGE "
            "('memory checkpoint', 'save')? Only ONE should appear because the "
            "hook returns early on the first injection.\n"
            "Report which injection appeared and confirm the other did NOT."
        ),
        "expected": (
            "Context warning fires (checked first in code). Extraction nudge is "
            "suppressed. Ground truth: estimated_context_chars > 16K AND "
            "tool_call_count >= 3, but only context warning was injected."
        ),
    },
    {
        "name": "cache_invalidation_cascade",
        "mission": (
            "This session has caching enabled. Your mission: test whether writing "
            "a file clears the cache without resetting the context estimate.\n"
            "(1) Read `copilotcode_sdk/__init__.py` (~2K) — cached + context tracked.\n"
            "(2) Write `/tmp/copilotcode_cascade_test.txt` with content 'invalidate' "
            "— this should clear the entire tool result cache.\n"
            "(3) Read `copilotcode_sdk/__init__.py` again — should be a FRESH read "
            "(cache was cleared), adding ~2K to context estimate AGAIN.\n"
            "Report whether the read in step 3 returned the same content as step 1, "
            "and whether any context-related injection appeared. The context estimate "
            "should now show ~4K (double-counted) even though the file is only ~2K."
        ),
        "expected": (
            "Cache cleared by write. Re-read adds to context estimate again "
            "(double-counted). Ground truth: tool_result_cache_size == 1 after "
            "step 3. estimated_context_chars ~4K (2x file size)."
        ),
    },
    {
        "name": "error_skip_vs_session_continuity",
        "mission": (
            "Your mission: trigger tool errors and verify the session remains "
            "stable without spurious cascade effects.\n"
            "(1) Try to read `/nonexistent/cascade_test_1.txt` — should error.\n"
            "(2) Try to read `/nonexistent/cascade_test_2.txt` — should error.\n"
            "(3) Read `copilotcode_sdk/__init__.py` — should succeed normally.\n"
            "(4) Run `echo recovery_confirmed` — should succeed normally.\n"
            "Report whether errors in steps 1-2 were handled cleanly (not crashes), "
            "whether steps 3-4 succeeded, and whether any unexpected "
            "`additionalContext` injections appeared from the errors."
        ),
        "expected": (
            "Errors are skipped cleanly. Session recovers. No spurious extraction "
            "nudges or task reminders from error calls. Ground truth: "
            "tool_call_count reflects all 4 calls. Session state intact."
        ),
    },
    {
        "name": "task_reminder_vs_extraction_race",
        "mission": (
            "This session has task reminder threshold at 2 non-task turns AND "
            "extraction threshold at 3 tool calls. Your mission: cross BOTH "
            "thresholds simultaneously and observe which injection wins.\n"
            "(1) Create a task using TaskCreate with subject 'cascade race test' "
            "and description 'testing reminder vs extraction priority'.\n"
            "(2) Run `echo race_a` — non-task call #1, tool call #2.\n"
            "(3) Run `echo race_b` — non-task call #2, tool call #3. Both "
            "thresholds now crossed.\n"
            "After step 3, watch carefully: do you see a TASK REMINDER mentioning "
            "'open tasks' or an EXTRACTION NUDGE mentioning 'memory checkpoint'? "
            "Report which one appeared."
        ),
        "expected": (
            "Only one injection fires per turn. Which one depends on code ordering "
            "in on_post_tool_use. Ground truth: tool_call_count >= 3, task store "
            "has open task."
        ),
    },
    {
        "name": "repeat_detection_ring_overflow",
        "mission": (
            "The repeat detection ring buffer holds the last 10 shell commands. "
            "Your mission: test that commands pushed out of the ring are forgotten.\n"
            "(1) Run `echo sentinel_cascade_test` — first occurrence, no warning.\n"
            "(2) Run `echo sentinel_cascade_test` — REPEAT, expect loop warning.\n"
            "(3) Run these 10 different commands to fill the ring and push out "
            "the sentinel: `echo flush_1`, `echo flush_2`, `echo flush_3`, "
            "`echo flush_4`, `echo flush_5`, `echo flush_6`, `echo flush_7`, "
            "`echo flush_8`, `echo flush_9`, `echo flush_10`.\n"
            "(4) Run `echo sentinel_cascade_test` — should NOT trigger a warning "
            "because the sentinel was pushed out of the ring.\n"
            "Report whether step 2 had a warning (expected yes) and step 4 had "
            "a warning (expected no)."
        ),
        "expected": (
            "Step 2: loop warning fires. Step 4: NO warning (ring overflow). "
            "Ground truth: recent_shell after step 4 does NOT contain the "
            "sentinel entries from steps 1-2."
        ),
    },
    {
        "name": "context_warning_single_fire",
        "mission": (
            "This session has a context limit of 20,000 chars (warning at 80%% = "
            "16K). The compaction_warned flag prevents warnings from re-firing. "
            "Your mission: verify warnings fire exactly once.\n"
            "(1) Read `copilotcode_sdk/hooks.py` (~30K chars) — should cross 80%% "
            "threshold and trigger a context warning.\n"
            "(2) Read `copilotcode_sdk/client.py` (~40K chars) — context grows "
            "much further but NO second warning should appear.\n"
            "(3) Read `copilotcode_sdk/exercise.py` (~30K chars) — still no warning.\n"
            "Report whether the warning appeared exactly once (step 1) and did "
            "NOT reappear in steps 2 or 3 despite growing context."
        ),
        "expected": (
            "Context warning fires exactly once on step 1. Steps 2-3 have no "
            "warning despite growing context. Ground truth: compaction_warned == "
            "True, estimated_context_chars continues growing across all reads."
        ),
    },
]


def build_cascade_config(
    base: "CopilotCodeConfig | None" = None,
) -> "CopilotCodeConfig":
    """Build a CopilotCodeConfig with aggressive thresholds for cascade exercise scenarios.

    Thresholds are lower than ``build_exercise_config`` so that multiple
    reactive behaviors can be crossed within 3-5 tool calls.
    """
    from .config import CopilotCodeConfig

    if base is None:
        base = CopilotCodeConfig(model="claude-sonnet-4.6")
    return replace(
        base,
        extraction_tool_call_interval=3,
        extraction_char_threshold=5_000,
        extraction_min_turn_gap=1,
        task_reminder_turns=2,
        task_reminder_cooldown_turns=2,
        enable_tool_result_cache=True,
        reminder_reinjection_interval=3,
        max_context_chars=20_000,
        noisy_tool_char_limit=5_000,
    )


def build_exercise_config(
    base: "CopilotCodeConfig | None" = None,
) -> "CopilotCodeConfig":
    """Build a CopilotCodeConfig with lowered thresholds for advanced exercise scenarios."""
    from .config import CopilotCodeConfig

    if base is None:
        base = CopilotCodeConfig(model="claude-sonnet-4.6")
    return replace(
        base,
        extraction_tool_call_interval=5,
        extraction_char_threshold=10_000,
        extraction_min_turn_gap=2,
        task_reminder_turns=3,
        task_reminder_cooldown_turns=3,
        enable_tool_result_cache=True,
        reminder_reinjection_interval=5,
        max_context_chars=50_000,
    )


def build_orchestration_prompt(
    scenarios: Sequence[dict[str, str]] | None = None,
) -> str:
    """Build self-aware prompt for orchestration exercise."""
    items = scenarios if scenarios is not None else ORCHESTRATION_SCENARIOS
    scenario_blocks = []
    for i, item in enumerate(items, 1):
        scenario_blocks.append(
            f"### Scenario {i}: {item['name']}\n\n"
            f"**Mission:** {item['mission']}\n\n"
            f"**Expected evidence:** {item['expected']}"
        )

    scenarios_text = "\n\n---\n\n".join(scenario_blocks)

    return f"""\
You are a self-aware orchestration exercise agent for copilotcode_sdk.

You know that you are running inside a real Copilot SDK session. The session has hooks, \
an event bus, file tracking, and other internal machinery. Your job is to PROVOKE this \
machinery by performing real actions, then OBSERVE the responses to verify the wiring works.

For each scenario below:
1. Read the mission carefully
2. Perform the actions described — use real tool calls
3. Observe what happens: tool results, any system-injected context (additionalContext), warnings, or behavioral changes
4. Report what you saw and whether it matches the expected evidence

## Scenarios

{scenarios_text}

## Instructions

- Execute each scenario IN ORDER
- Use REAL tool calls (bash, read, write) — do not simulate or fake anything
- Pay attention to EVERYTHING in the responses — not just tool results, but any system messages, context injections, or warnings
- For each scenario, report: name, status (pass/fail/skip/error), what you observed, and any discrepancies from expected behavior
- After all scenarios, write a brief summary

## Output Format

Return your results as JSON inside <exercise-report> tags:

```json
{{
  "subsystems": [
    {{
      "name": "scenario_name",
      "status": "pass|fail|skip|error",
      "detail": "what you observed — be specific about any system context injections",
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


def build_advanced_orchestration_prompt(
    scenarios: Sequence[dict[str, str]] | None = None,
) -> str:
    """Build self-aware prompt for advanced orchestration exercise.

    The advanced prompt explains that config thresholds are lowered so the
    agent expects system reactions to fire sooner than in production.
    """
    items = scenarios if scenarios is not None else ADVANCED_ORCHESTRATION_SCENARIOS
    scenario_blocks = []
    for i, item in enumerate(items, 1):
        scenario_blocks.append(
            f"### Scenario {i}: {item['name']}\n\n"
            f"**Mission:** {item['mission']}\n\n"
            f"**Expected evidence:** {item['expected']}"
        )

    scenarios_text = "\n\n---\n\n".join(scenario_blocks)

    return f"""\
You are a self-aware advanced orchestration exercise agent for copilotcode_sdk.

You know that you are running inside a real Copilot SDK session with LOWERED THRESHOLDS \
configured specifically for this exercise. The session has hooks, an event bus, file tracking, \
tool result caching, safety gates, and other reactive machinery. Your job is to PROVOKE specific \
reactive behaviors by performing targeted actions, then OBSERVE the system's responses.

## Lowered Thresholds (Exercise Mode)

This session is configured with exercise-specific overrides:
- Extraction nudge fires after **5 tool calls** (production: 20)
- Extraction char threshold is **10K chars** (production: 50K)
- Task reminder fires after **3 non-task turns** (production: 10)
- Tool result caching is **enabled** (production: disabled)
- Context size limit is **50K chars** (production: 800K) — warnings at 80%
- Skill reminder interval is **5 tool calls** (production: 15)

These lowered thresholds mean you should expect system reactions to fire quickly. Pay close \
attention to `additionalContext` injections, warnings, and behavioral changes after each action.

For each scenario below:
1. Read the mission carefully
2. Perform the actions described — use real tool calls
3. Observe what happens: tool results, any system-injected context (additionalContext), \
warnings, denials, or behavioral changes
4. Report what you saw and whether it matches the expected evidence

## Scenarios

{scenarios_text}

## Instructions

- Execute each scenario IN ORDER
- Use REAL tool calls (bash, read, write, TaskCreate) — do not simulate or fake anything
- Pay attention to EVERYTHING in the responses — not just tool results, but any system \
messages, context injections, warnings, or denials
- For each scenario, report: name, status (pass/fail/skip/error), what you observed, \
and any discrepancies from expected behavior
- After all scenarios, write a brief summary

## Output Format

Return your results as JSON inside <exercise-report> tags:

```json
{{{{
  "subsystems": [
    {{{{
      "name": "scenario_name",
      "status": "pass|fail|skip|error",
      "detail": "what you observed — be specific about system context injections, warnings, denials",
      "duration_seconds": 0.0,
      "error": null
    }}}}
  ],
  "summary": "brief overall summary"
}}}}
```

<exercise-report>
YOUR JSON HERE
</exercise-report>
"""


def build_cascade_orchestration_prompt(
    scenarios: Sequence[dict[str, str]] | None = None,
) -> str:
    """Build self-aware prompt for cascade orchestration exercise.

    The cascade prompt explains that thresholds are aggressively lowered so
    multiple reactive behaviors can collide within a few tool calls.
    """
    items = scenarios if scenarios is not None else CASCADE_ORCHESTRATION_SCENARIOS
    scenario_blocks = []
    for i, item in enumerate(items, 1):
        scenario_blocks.append(
            f"### Scenario {i}: {item['name']}\n\n"
            f"**Mission:** {item['mission']}\n\n"
            f"**Expected evidence:** {item['expected']}"
        )

    scenarios_text = "\n\n---\n\n".join(scenario_blocks)

    return f"""\
You are a self-aware cascade orchestration exercise agent for copilotcode_sdk.

You know that you are running inside a real Copilot SDK session with AGGRESSIVELY LOWERED \
THRESHOLDS configured to make multiple reactive behaviors collide. Your job is to PROVOKE \
specific cascade interactions where two or more subsystems compete, then OBSERVE which \
behavior wins and how they interfere.

## Aggressively Lowered Thresholds (Cascade Mode)

This session is configured with cascade-specific overrides:
- Extraction nudge fires after **3 tool calls** (production: 20, advanced: 5)
- Extraction char threshold is **5K chars** (production: 50K, advanced: 10K)
- Extraction min turn gap is **1** (production: 10, advanced: 2)
- Task reminder fires after **2 non-task turns** (production: 10, advanced: 3)
- Tool result caching is **enabled**
- Context size limit is **20K chars** (production: 800K, advanced: 50K) — warnings at 80%
- Noisy tool truncation at **5K chars** (production: 8K)

These aggressive thresholds mean MULTIPLE behaviors will fire on the same turn. Pay close \
attention to WHICH injection appears and which is suppressed — only one `additionalContext` \
injection can fire per tool-result turn.

For each scenario below:
1. Read the mission carefully — it describes a specific cascade interaction
2. Perform the actions described — use real tool calls
3. Observe what happens: which injection fired? Was something suppressed?
4. Report what you saw and whether it matches the expected evidence

## Scenarios

{scenarios_text}

## Instructions

- Execute each scenario IN ORDER
- Use REAL tool calls (bash, read, write, TaskCreate) — do not simulate
- Pay attention to WHICH `additionalContext` appears and which does NOT
- For each scenario, report: name, status (pass/fail/skip/error), what you observed \
(be specific about which injections appeared and which were absent)
- After all scenarios, write a brief summary

## Output Format

Return your results as JSON inside <exercise-report> tags:

```json
{{{{
  "subsystems": [
    {{{{
      "name": "scenario_name",
      "status": "pass|fail|skip|error",
      "detail": "what you observed — which injections fired, which were suppressed",
      "duration_seconds": 0.0,
      "error": null
    }}}}
  ],
  "summary": "brief overall summary"
}}}}
```

<exercise-report>
YOUR JSON HERE
</exercise-report>
"""


def _capture_ground_truth(session: Any) -> dict[str, Any]:
    """Capture independent ground truth from session state after exercise."""
    ground_truth: dict[str, Any] = {}

    # Event bus history
    event_bus = getattr(session, "event_bus", None)
    if event_bus is not None:
        history = event_bus.history
        event_counts: dict[str, int] = {}
        for event in history:
            type_name = event.type.value if hasattr(event.type, "value") else str(event.type)
            event_counts[type_name] = event_counts.get(type_name, 0) + 1
        ground_truth["event_counts"] = event_counts
        ground_truth["total_events"] = len(history)

    # File changes from hooks
    drain_fn = getattr(session, "_drain_skills", None)
    completed = getattr(session, "_completed_skills", None)
    if completed is not None:
        ground_truth["completed_skills"] = list(completed)

    # Session state
    state = getattr(session, "_state", None)
    if state is not None:
        ground_truth["turn_count"] = getattr(state, "turn_count", 0)
        ground_truth["total_input_tokens"] = getattr(state, "total_input_tokens", 0)
        ground_truth["total_output_tokens"] = getattr(state, "total_output_tokens", 0)

    # Hook accessor state (for advanced orchestration ground truth)
    raw_hooks = getattr(session, "_raw_hooks", None)
    if raw_hooks:
        _safe_call = lambda fn: fn() if callable(fn) else None
        tool_call_count = _safe_call(raw_hooks.get("get_tool_call_count"))
        if tool_call_count is not None:
            ground_truth["tool_call_count"] = tool_call_count
        recent_shell = _safe_call(raw_hooks.get("get_recent_shell"))
        if recent_shell is not None:
            ground_truth["recent_shell"] = recent_shell
        read_state = _safe_call(raw_hooks.get("get_read_file_state"))
        if read_state is not None:
            ground_truth["read_file_state_keys"] = list(read_state.keys())
        est_ctx = _safe_call(raw_hooks.get("get_estimated_context_chars"))
        if est_ctx is not None:
            ground_truth["estimated_context_chars"] = est_ctx
        cache_size = _safe_call(raw_hooks.get("get_tool_result_cache_size"))
        if cache_size is not None:
            ground_truth["tool_result_cache_size"] = cache_size
        budget_fn = raw_hooks.get("get_token_budget")
        budget = _safe_call(budget_fn)
        if budget is not None:
            ground_truth["token_budget"] = {
                "tokens": getattr(budget, "tokens", None),
                "consumed": getattr(budget, "consumed", None),
                "progress": getattr(budget, "progress", None),
            }
        file_changes = _safe_call(raw_hooks.get("get_file_changes"))
        if file_changes:
            ground_truth["file_changes"] = file_changes
        compaction_warned = _safe_call(raw_hooks.get("get_compaction_warned"))
        if compaction_warned is not None:
            ground_truth["compaction_warned"] = compaction_warned

    return ground_truth


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


async def _run_single_exercise(
    client: "CopilotCodeClient",
    *,
    mode: str,
    prompt: str,
    timeout: float,
    session_id: str,
    timestamp: str,
) -> ExerciseReport:
    """Run a single exercise session and capture ground truth."""
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
        report.mode = mode

        # Capture ground truth from session state
        if session is not None:
            report.ground_truth = _capture_ground_truth(session)

    except Exception as exc:
        elapsed = time.monotonic() - start
        report = ExerciseReport(
            product_name=client.config.brand.public_name,
            session_id=session_id,
            timestamp=timestamp,
            mode=mode,
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
    return report


async def run_exercise(
    client: "CopilotCodeClient",
    *,
    timeout: float = 3600.0,
    mode: ExerciseMode = "subsystem",
    subsystems: Sequence[str] | None = None,
    save_report_path: str | Path | None = None,
) -> ExerciseReport:
    """Create a real session, send the exercise prompt, parse the report.

    Modes:
    - ``subsystem``: Run the 17 API-level subsystem checks
    - ``orchestration``: Run the 6 basic wiring/provocation scenarios
    - ``advanced``: Run the 7 advanced reactive-behavior scenarios (lowered thresholds)
    - ``cascade``: Run the 7 cascade multi-subsystem interaction scenarios (aggressive thresholds)
    - ``micro``: Run 6 LLM-based micro-exercises (single prompt, real session, LLM-verified)
    - ``chain``: Run 2 LLM-based chain exercises (multi-step file pipelines, LLM-verified)
    - ``full``: Run all tiers sequentially, merging results into one report
    """
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).isoformat()

    if mode == "full":
        # Run all three, merge results
        sub_id = f"{client.config.brand.slug}-exercise-sub-{uuid.uuid4().hex[:12]}"
        sub_checklist: list[dict[str, str]] | None = None
        if subsystems:
            sub_checklist = [i for i in SUBSYSTEM_CHECKLIST if i["name"] in subsystems]
        sub_prompt = build_exercise_prompt(sub_checklist)
        sub_report = await _run_single_exercise(
            client, mode="subsystem", prompt=sub_prompt,
            timeout=timeout, session_id=sub_id, timestamp=timestamp,
        )

        orch_id = f"{client.config.brand.slug}-exercise-orch-{uuid.uuid4().hex[:12]}"
        orch_prompt = build_orchestration_prompt()
        orch_report = await _run_single_exercise(
            client, mode="orchestration", prompt=orch_prompt,
            timeout=timeout, session_id=orch_id, timestamp=timestamp,
        )

        # Advanced mode uses lowered-threshold config
        from .client import CopilotCodeClient
        exercise_cfg = build_exercise_config(client.config)
        exercise_client = CopilotCodeClient(exercise_cfg)
        adv_id = f"{client.config.brand.slug}-exercise-adv-{uuid.uuid4().hex[:12]}"
        adv_prompt = build_advanced_orchestration_prompt()
        adv_report = await _run_single_exercise(
            exercise_client, mode="advanced", prompt=adv_prompt,
            timeout=timeout, session_id=adv_id, timestamp=timestamp,
        )

        # Cascade mode uses even more aggressive thresholds
        cascade_cfg = build_cascade_config(client.config)
        cascade_client = CopilotCodeClient(cascade_cfg)
        cas_id = f"{client.config.brand.slug}-exercise-cas-{uuid.uuid4().hex[:12]}"
        cas_prompt = build_cascade_orchestration_prompt()
        cas_report = await _run_single_exercise(
            cascade_client, mode="cascade", prompt=cas_prompt,
            timeout=timeout, session_id=cas_id, timestamp=timestamp,
        )

        # Micro and chain exercises
        from .micro_exercise import run_micro_exercises, run_chain_exercises
        micro_report = await run_micro_exercises(client, timeout=timeout)
        chain_report = await run_chain_exercises(client, timeout=timeout)

        # Merge
        all_subsystems = (
            sub_report.subsystems
            + orch_report.subsystems
            + adv_report.subsystems
            + cas_report.subsystems
            + micro_report.subsystems
            + chain_report.subsystems
        )
        total_duration = (
            sub_report.total_duration_seconds
            + orch_report.total_duration_seconds
            + adv_report.total_duration_seconds
            + cas_report.total_duration_seconds
            + micro_report.total_duration_seconds
            + chain_report.total_duration_seconds
        )
        merged = ExerciseReport(
            product_name=client.config.brand.public_name,
            session_id=f"{sub_id}+{orch_id}+{adv_id}+{cas_id}",
            timestamp=timestamp,
            mode="full",
            subsystems=all_subsystems,
            summary=(
                f"Subsystem: {sub_report.summary} | "
                f"Orchestration: {orch_report.summary} | "
                f"Advanced: {adv_report.summary} | "
                f"Cascade: {cas_report.summary} | "
                f"Micro: {micro_report.summary} | "
                f"Chain: {chain_report.summary}"
            ),
            total_duration_seconds=total_duration,
            ground_truth=cas_report.ground_truth,
        )
        report = merged
    elif mode == "micro":
        from .micro_exercise import run_micro_exercises
        report = await run_micro_exercises(client, timeout=timeout)
    elif mode == "chain":
        from .micro_exercise import run_chain_exercises
        report = await run_chain_exercises(client, timeout=timeout)
    elif mode == "cascade":
        from .client import CopilotCodeClient
        cascade_cfg = build_cascade_config(client.config)
        cascade_client = CopilotCodeClient(cascade_cfg)
        session_id = f"{client.config.brand.slug}-exercise-cas-{uuid.uuid4().hex[:12]}"
        prompt = build_cascade_orchestration_prompt()
        report = await _run_single_exercise(
            cascade_client, mode="cascade", prompt=prompt,
            timeout=timeout, session_id=session_id, timestamp=timestamp,
        )
    elif mode == "advanced":
        from .client import CopilotCodeClient
        exercise_cfg = build_exercise_config(client.config)
        exercise_client = CopilotCodeClient(exercise_cfg)
        session_id = f"{client.config.brand.slug}-exercise-adv-{uuid.uuid4().hex[:12]}"
        prompt = build_advanced_orchestration_prompt()
        report = await _run_single_exercise(
            exercise_client, mode="advanced", prompt=prompt,
            timeout=timeout, session_id=session_id, timestamp=timestamp,
        )
    elif mode == "orchestration":
        session_id = f"{client.config.brand.slug}-exercise-orch-{uuid.uuid4().hex[:12]}"
        prompt = build_orchestration_prompt()
        report = await _run_single_exercise(
            client, mode="orchestration", prompt=prompt,
            timeout=timeout, session_id=session_id, timestamp=timestamp,
        )
    else:
        # subsystem mode
        session_id = f"{client.config.brand.slug}-exercise-sub-{uuid.uuid4().hex[:12]}"
        checklist: list[dict[str, str]] | None = None
        if subsystems:
            checklist = [i for i in SUBSYSTEM_CHECKLIST if i["name"] in subsystems]
        prompt = build_exercise_prompt(checklist)
        report = await _run_single_exercise(
            client, mode="subsystem", prompt=prompt,
            timeout=timeout, session_id=session_id, timestamp=timestamp,
        )

    if save_report_path:
        path = Path(save_report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.to_dict(), indent=2))

    return report
