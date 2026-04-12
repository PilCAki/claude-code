# Auto-generated exercise runner for copilotcode_sdk subsystems
import time
import traceback
import json
import sys
import tempfile
from pathlib import Path

# List of test snippets to execute
tests = [
    {
        "name": "prompt_compiler",
        "code": """from copilotcode_sdk.prompt_compiler import PromptAssembler, PromptPriority
a = PromptAssembler()
a.add('test', 'hello', priority=PromptPriority.default)
result = a.render()
assert 'hello' in result
""",
    },
    {
        "name": "session_state",
        "code": """from copilotcode_sdk.session_state import SessionState, SessionStatus
s = SessionState(status=SessionStatus.idle)
s.status = SessionStatus.running
assert s.status == SessionStatus.running
""",
    },
    {
        "name": "tasks",
        "code": """import tempfile
from pathlib import Path
from copilotcode_sdk.tasks import TaskStore, TaskStatus
store = TaskStore(persist_path=Path(tempfile.mkdtemp()) / 'tasks.json')
t = store.create('test task')
store.update(t.id, status=TaskStatus.in_progress)
assert len(store.list_open()) >= 1
""",
    },
    {
        "name": "memory",
        "code": """import tempfile
from pathlib import Path
from copilotcode_sdk.memory import MemoryStore
d = Path(tempfile.mkdtemp())
store = MemoryStore(d, d / '.mem')
store.upsert_memory(title='test', content='hello world', description='exercise test', memory_type='project')
records = store.list_records()
assert any(r.name == 'test' for r in records)
""",
    },
    {
        "name": "compaction",
        "code": """from copilotcode_sdk.compaction import build_compaction_prompt, parse_compaction_response, format_transcript_for_compaction
p = build_compaction_prompt()
assert 'Primary request' in p
r = parse_compaction_response('<summary>test</summary>')
assert r.summary == 'test'
t = format_transcript_for_compaction([{'role':'user','content':'hi'}])
assert '[user]' in t
""",
    },
    {
        "name": "extraction",
        "code": """import tempfile
from pathlib import Path
from copilotcode_sdk.extraction import build_extraction_prompt, should_extract
d = Path(tempfile.mkdtemp())
p = build_extraction_prompt(memory_dir=str(d), project_root=str(d))
assert len(p) > 0
assert should_extract(tool_call_count=25, total_chars=60000, last_extraction_turn=0, current_turn=30) == True
assert should_extract(tool_call_count=1, total_chars=100, last_extraction_turn=0, current_turn=1) == False
""",
    },
    {
        "name": "session_memory",
        "code": """import tempfile
from pathlib import Path
from copilotcode_sdk.memory import MemoryStore
from copilotcode_sdk.session_memory import SessionMemoryController
d = Path(tempfile.mkdtemp())
store = MemoryStore(d, d / '.mem')
ctrl = SessionMemoryController(store)
assert ctrl.state.initialized == False
""",
    },
    {
        "name": "events",
        "code": """from copilotcode_sdk.events import EventBus, Event, EventType
bus = EventBus()
received = []
bus.subscribe(lambda e: received.append(e), event_type=EventType.tool_called)
bus.emit(Event(type=EventType.tool_called, data={'tool': 'test'}))
assert len(received) == 1
""",
    },
    {
        "name": "diff",
        "code": """from copilotcode_sdk.diff import generate_diff
result = generate_diff('hello', 'hello world')
assert result.changed
""",
    },
    {
        "name": "tokenizer",
        "code": """from copilotcode_sdk.tokenizer import estimate_tokens
count = estimate_tokens('hello world')
assert count > 0
""",
    },
    {
        "name": "retry",
        "code": """from copilotcode_sdk.retry import RetryPolicy, RetryState, build_retry_response
policy = RetryPolicy()
state = RetryState(policy)
resp = build_retry_response(state, error_context='test error')
assert resp is not None
""",
    },
    {
        "name": "suggestions",
        "code": """from copilotcode_sdk.suggestions import build_prompt_suggestions
result = build_prompt_suggestions(session_turn=3)
assert isinstance(result, list)
""",
    },
    {
        "name": "skill_assets",
        "code": """import tempfile
from pathlib import Path
from copilotcode_sdk.skill_assets import parse_skill_frontmatter
d = Path(tempfile.mkdtemp())
p = d / 'SKILL.md'
p.write_text('---\\nname: test\\ndescription: a test\\n---\\n# Test')
fm = parse_skill_frontmatter(p)
assert fm['name'] == 'test'
""",
    },
    {
        "name": "permissions",
        "code": """from copilotcode_sdk.permissions import PermissionPolicy
assert 'safe' in PermissionPolicy.__args__
assert 'approve_all' in PermissionPolicy.__args__
""",
    },
    {
        "name": "model_cost",
        "code": """from copilotcode_sdk.model_cost import calculate_cost, UsageCost
cost = calculate_cost(model='claude-sonnet-4-20250514', input_tokens=1000, output_tokens=500)
assert isinstance(cost, UsageCost)
assert cost.total > 0
""",
    },
    {
        "name": "config",
        "code": """from copilotcode_sdk.config import CopilotCodeConfig
cfg = CopilotCodeConfig()
assert cfg.permission_policy == 'safe'
""",
    },
    {
        "name": "instructions",
        "code": """import tempfile
from pathlib import Path
from copilotcode_sdk.instructions import load_workspace_instructions
d = Path(tempfile.mkdtemp())
(d / 'CLAUDE.md').write_text('# Test instructions')
bundle = load_workspace_instructions(d)
assert 'Test instructions' in bundle.content
""",
    },
]

results = []

for t in tests:
    name = t['name']
    code = t['code']
    start = time.perf_counter()
    try:
        # Execute in isolated namespace
        ns = {}
        exec(code, ns, ns)
        status = 'pass'
        detail = 'Executed successfully'
        err = None
    except AssertionError as e:
        status = 'fail'
        detail = 'AssertionError: %s' % (str(e),)
        err = traceback.format_exc()
    except Exception as e:
        status = 'error'
        detail = 'Exception: %s' % (str(e),)
        err = traceback.format_exc()
    duration = time.perf_counter() - start
    results.append({
        'name': name,
        'status': status,
        'detail': detail,
        'duration_seconds': duration,
        'error': err,
    })

# Summary
passed = sum(1 for r in results if r['status'] == 'pass')
failed = sum(1 for r in results if r['status'] in ('fail','error'))
summary = f"{passed} passed, {failed} failed/errored out of {len(results)} tests"
output = {
    'subsystems': results,
    'summary': summary,
}
# Print JSON wrapped in tags as requested
print('<exercise-report>')
print(json.dumps(output, ensure_ascii=False, indent=2))
print('</exercise-report>')
