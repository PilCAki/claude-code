#!/usr/bin/env python3
import time
import json
import traceback
import tempfile
from pathlib import Path

results = []

snippets = [
    ("prompt_compiler", r"""
from copilotcode_sdk.prompt_compiler import PromptAssembler, PromptPriority
a = PromptAssembler()
a.add('test', 'hello', priority=PromptPriority.HIGH)
result = a.render()
assert 'hello' in result
"""),

    ("session_state", r"""
from copilotcode_sdk.session_state import SessionState, SessionStatus
s = SessionState(session_id='ex', status=SessionStatus.IDLE)
s.status = SessionStatus.ACTIVE
assert s.status == SessionStatus.ACTIVE
"""),

    ("tasks", r"""
import tempfile
from pathlib import Path
from copilotcode_sdk.tasks import TaskStore, TaskStatus
store = TaskStore(persist_path=Path(tempfile.mkdtemp()) / 'tasks.json')
t = store.create('test task')
store.update_status(t.id, TaskStatus.IN_PROGRESS)
assert len(store.list_open()) >= 1
"""),

    ("memory", r"""
import tempfile
from pathlib import Path
from copilotcode_sdk.memory import MemoryStore
d = Path(tempfile.mkdtemp())
store = MemoryStore(d, d / '.mem')
store.write_record('test', 'hello world', description='exercise test', memory_type='project')
records = store.list_records()
assert any(r.name == 'test' for r in records)
"""),

    ("compaction", r"""
from copilotcode_sdk.compaction import build_compaction_prompt, parse_compaction_response, format_transcript_for_compaction
p = build_compaction_prompt()
assert 'Primary request' in p
r = parse_compaction_response('<summary>test</summary>')
assert r.summary == 'test'
t = format_transcript_for_compaction([{'role':'user','content':'hi'}])
assert '[user]' in t
"""),

    ("extraction", r"""
from copilotcode_sdk.extraction import build_extraction_prompt, should_extract
p = build_extraction_prompt()
assert len(p) > 0
assert should_extract(turn_count=20) == True
assert should_extract(turn_count=1) == False
"""),

    ("session_memory", r"""
from copilotcode_sdk.session_memory import SessionMemoryController, SessionMemoryState
ctrl = SessionMemoryController()
assert ctrl.state == SessionMemoryState.EMPTY
"""),

    ("events", r"""
from copilotcode_sdk.events import EventBus, Event, EventType
bus = EventBus()
received = []
bus.subscribe(lambda e: received.append(e), event_type=EventType.TOOL_USE)
bus.emit(Event(type=EventType.TOOL_USE, data={'tool': 'test'}))
assert len(received) == 1
"""),

    ("diff", r"""
from copilotcode_sdk.diff import generate_diff
result = generate_diff('hello', 'hello world')
assert result.has_changes
"""),

    ("tokenizer", r"""
from copilotcode_sdk.tokenizer import estimate_tokens
count = estimate_tokens('hello world')
assert count > 0
"""),

    ("retry", r"""
from copilotcode_sdk.retry import RetryPolicy, RetryState, build_retry_response
policy = RetryPolicy()
state = RetryState()
resp = build_retry_response(policy, state, error='test error')
assert 'test error' in resp or resp is not None
"""),

    ("suggestions", r"""
from copilotcode_sdk.suggestions import build_prompt_suggestions
result = build_prompt_suggestions(tool_names=['read', 'edit'], turn_count=3)
assert isinstance(result, list)
"""),

    ("skill_assets", r"""
from copilotcode_sdk.skill_assets import parse_skill_frontmatter
fm = parse_skill_frontmatter('---\nname: test\ndescription: a test skill\n---\n# Test')
assert fm['name'] == 'test'
"""),

    ("permissions", r"""
from copilotcode_sdk.permissions import PermissionPolicy
policy = PermissionPolicy(mode='safe')
assert policy.mode == 'safe'
"""),

    ("model_cost", r"""
from copilotcode_sdk.model_cost import calculate_cost, UsageCost
cost = calculate_cost(model='claude-sonnet-4-20250514', input_tokens=1000, output_tokens=500)
assert isinstance(cost, UsageCost)
assert cost.total_cost > 0
"""),

    ("config", r"""
from copilotcode_sdk.config import CopilotCodeConfig
cfg = CopilotCodeConfig()
assert cfg.working_directory == '.'
assert cfg.permission_policy == 'safe'
"""),

    ("instructions", r"""
import tempfile
from pathlib import Path
from copilotcode_sdk.instructions import load_workspace_instructions
d = Path(tempfile.mkdtemp())
(d / 'CLAUDE.md').write_text('# Test instructions')
bundle = load_workspace_instructions(d)
assert 'Test instructions' in bundle.content
"""),
]

for name, code in snippets:
    start = time.perf_counter()
    status = 'pass'
    detail = ''
    error = None
    try:
        # execute each snippet in a fresh globals dict
        gl = {'__name__': '__main__'}
        exec(code, gl)
        detail = 'executed successfully'
    except AssertionError as e:
        status = 'fail'
        detail = 'assertion failed: ' + str(e)
        error = traceback.format_exc()
    except Exception as e:
        status = 'error'
        detail = 'exception: ' + str(e)
        error = traceback.format_exc()
    duration = time.perf_counter() - start
    results.append({
        'name': name,
        'status': status,
        'detail': detail,
        'duration_seconds': round(duration, 4),
        'error': error,
    })

summary = {
    'subsystems': results,
    'summary': f"{sum(1 for r in results if r['status']=='pass')} passed, {sum(1 for r in results if r['status']!='pass')} failed/errored",
}

# Print JSON only
print(json.dumps(summary))
