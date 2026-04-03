# Phase 3: Session Memory & Prompt Suggestions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add session-memory extraction, compaction with context preservation, and prompt suggestions so copilotcode_sdk sessions maintain learnings across compaction boundaries and nudge the agent toward available skills.

**Architecture:** Three new modules (`extraction.py`, `compaction.py`) plus hook enhancements in `hooks.py`. Extraction runs periodically during a session to capture durable learnings into memory. Compaction produces a structured summary when the context window is compressed. Prompt suggestions detect unstarted skills with met prerequisites and nudge the agent.

**Tech Stack:** Python 3.10+, pytest, copilotcode_sdk internals (hooks, config, memory)

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `copilotcode_sdk/extraction.py` | Threshold logic + extraction prompt template |
| Create | `copilotcode_sdk/compaction.py` | Compaction prompt template + handoff context builder |
| Modify | `copilotcode_sdk/hooks.py` | Wire extraction trigger + prompt suggestions |
| Modify | `copilotcode_sdk/config.py` | Add `extraction_interval` and `extraction_char_threshold` fields |
| Modify | `copilotcode_sdk/__init__.py` | Export new public names |
| Create | `tests/test_extraction.py` | Tests for extraction module |
| Create | `tests/test_compaction.py` | Tests for compaction module |
| Modify | `tests/test_hooks.py` | Tests for extraction trigger + prompt suggestions |

---

### Task 1: Extraction threshold logic (`extraction.py`)

**Files:**
- Create: `copilotcode_sdk/extraction.py`
- Create: `tests/test_extraction.py`

- [ ] **Step 1: Write failing tests for `should_extract()`**

```python
# tests/test_extraction.py
from __future__ import annotations

from copilotcode_sdk.extraction import should_extract


def test_should_extract_false_below_tool_call_threshold() -> None:
    assert should_extract(tool_call_count=10, total_chars=0, last_extraction_turn=0, current_turn=15) is False


def test_should_extract_true_at_tool_call_threshold() -> None:
    assert should_extract(tool_call_count=20, total_chars=0, last_extraction_turn=0, current_turn=15) is True


def test_should_extract_true_at_char_threshold() -> None:
    assert should_extract(tool_call_count=5, total_chars=50_000, last_extraction_turn=0, current_turn=15) is True


def test_should_extract_false_if_too_recent() -> None:
    # Even though tool_call_count is high, last extraction was only 5 turns ago
    assert should_extract(tool_call_count=20, total_chars=60_000, last_extraction_turn=10, current_turn=15) is False


def test_should_extract_true_with_custom_thresholds() -> None:
    assert should_extract(
        tool_call_count=5,
        total_chars=0,
        last_extraction_turn=0,
        current_turn=15,
        tool_call_interval=5,
        char_threshold=100,
        min_turn_gap=3,
    ) is True


def test_should_extract_respects_custom_min_turn_gap() -> None:
    assert should_extract(
        tool_call_count=20,
        total_chars=0,
        last_extraction_turn=12,
        current_turn=14,
        min_turn_gap=5,
    ) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /c/Users/16026/Documents/Code/claude-code && python -m pytest tests/test_extraction.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement `should_extract()`**

```python
# copilotcode_sdk/extraction.py
from __future__ import annotations

DEFAULT_TOOL_CALL_INTERVAL = 20
DEFAULT_CHAR_THRESHOLD = 50_000
DEFAULT_MIN_TURN_GAP = 10


def should_extract(
    *,
    tool_call_count: int,
    total_chars: int,
    last_extraction_turn: int,
    current_turn: int,
    tool_call_interval: int = DEFAULT_TOOL_CALL_INTERVAL,
    char_threshold: int = DEFAULT_CHAR_THRESHOLD,
    min_turn_gap: int = DEFAULT_MIN_TURN_GAP,
) -> bool:
    """Decide whether to trigger a memory extraction pass.

    Returns True when either threshold is met AND enough turns have
    elapsed since the last extraction.
    """
    if current_turn - last_extraction_turn < min_turn_gap:
        return False
    if tool_call_count >= tool_call_interval:
        return True
    if total_chars >= char_threshold:
        return True
    return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /c/Users/16026/Documents/Code/claude-code && python -m pytest tests/test_extraction.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
cd /c/Users/16026/Documents/Code/claude-code
git add copilotcode_sdk/extraction.py tests/test_extraction.py
git commit -m "feat: add extraction threshold logic (should_extract)"
```

---

### Task 2: Extraction prompt template (`extraction.py`)

**Files:**
- Modify: `copilotcode_sdk/extraction.py`
- Modify: `tests/test_extraction.py`

- [ ] **Step 1: Write failing test for `build_extraction_prompt()`**

```python
# append to tests/test_extraction.py

from copilotcode_sdk.extraction import build_extraction_prompt


def test_build_extraction_prompt_contains_required_sections() -> None:
    prompt = build_extraction_prompt(memory_dir="/tmp/mem", project_root="/tmp/project")
    assert "durable" in prompt.lower()
    assert "/tmp/mem" in prompt
    assert "user" in prompt  # memory types
    assert "feedback" in prompt
    assert "project" in prompt
    assert "reference" in prompt


def test_build_extraction_prompt_includes_project_root() -> None:
    prompt = build_extraction_prompt(memory_dir="/data/mem", project_root="/data/repo")
    assert "/data/repo" in prompt
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `cd /c/Users/16026/Documents/Code/claude-code && python -m pytest tests/test_extraction.py::test_build_extraction_prompt_contains_required_sections tests/test_extraction.py::test_build_extraction_prompt_includes_project_root -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `build_extraction_prompt()`**

```python
# Add to copilotcode_sdk/extraction.py

def build_extraction_prompt(*, memory_dir: str, project_root: str) -> str:
    """Build the prompt that asks the model to extract durable learnings.

    This prompt is injected as additionalContext when extraction triggers.
    The model should save any durable learnings it has accumulated to
    the memory directory.
    """
    return (
        "**Session memory extraction checkpoint.** "
        "Review what you have learned in this session so far and save any durable "
        "learnings to the memory system. Durable learnings are facts, preferences, "
        "or context that would be valuable in future conversations about this project.\n\n"
        "Memory types to consider:\n"
        "- **user**: Role, preferences, expertise level\n"
        "- **feedback**: Corrections or confirmed approaches\n"
        "- **project**: Ongoing work, goals, decisions, timelines\n"
        "- **reference**: Pointers to external resources\n\n"
        f"Memory directory: `{memory_dir}`\n"
        f"Project root: `{project_root}`\n\n"
        "Only save things that are not already derivable from the code or git history. "
        "Skip ephemeral task details. If you have nothing durable to save, that is fine — "
        "do not force it. Continue with your current work after this checkpoint."
    )
```

- [ ] **Step 4: Run all extraction tests**

Run: `cd /c/Users/16026/Documents/Code/claude-code && python -m pytest tests/test_extraction.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
cd /c/Users/16026/Documents/Code/claude-code
git add copilotcode_sdk/extraction.py tests/test_extraction.py
git commit -m "feat: add extraction prompt template"
```

---

### Task 3: Config fields for extraction

**Files:**
- Modify: `copilotcode_sdk/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing test for new config fields**

```python
# append to tests/test_config.py

def test_extraction_config_defaults() -> None:
    config = CopilotCodeConfig()
    assert config.extraction_tool_call_interval == 20
    assert config.extraction_char_threshold == 50_000
    assert config.extraction_min_turn_gap == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /c/Users/16026/Documents/Code/claude-code && python -m pytest tests/test_config.py::test_extraction_config_defaults -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Add fields to `CopilotCodeConfig`**

Add these three fields to the `CopilotCodeConfig` dataclass in `copilotcode_sdk/config.py`, after `reminder_reinjection_interval`:

```python
    extraction_tool_call_interval: int = 20
    extraction_char_threshold: int = 50_000
    extraction_min_turn_gap: int = 10
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /c/Users/16026/Documents/Code/claude-code && python -m pytest tests/test_config.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
cd /c/Users/16026/Documents/Code/claude-code
git add copilotcode_sdk/config.py tests/test_config.py
git commit -m "feat: add extraction config fields to CopilotCodeConfig"
```

---

### Task 4: Wire extraction into `on_post_tool_use` hook

**Files:**
- Modify: `copilotcode_sdk/hooks.py`
- Modify: `tests/test_hooks.py`

- [ ] **Step 1: Write failing tests for extraction trigger in hooks**

```python
# append to tests/test_hooks.py

def test_post_tool_use_triggers_extraction_at_threshold(tmp_path: Path) -> None:
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        extraction_tool_call_interval=3,
        extraction_min_turn_gap=0,
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)

    # First 2 calls — no extraction
    for _ in range(2):
        hooks["on_post_tool_use"](
            {"toolName": "read", "toolResult": "content"},
            {},
        )

    # 3rd call — should trigger extraction
    result = hooks["on_post_tool_use"](
        {"toolName": "read", "toolResult": "content"},
        {},
    )

    assert result is not None
    assert "memory extraction" in result["additionalContext"].lower()


def test_post_tool_use_extraction_respects_min_turn_gap(tmp_path: Path) -> None:
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        extraction_tool_call_interval=1,
        extraction_min_turn_gap=100,  # Very high — should never trigger
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)

    result = hooks["on_post_tool_use"](
        {"toolName": "read", "toolResult": "content"},
        {},
    )

    # Should not trigger extraction because min_turn_gap is too high
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /c/Users/16026/Documents/Code/claude-code && python -m pytest tests/test_hooks.py::test_post_tool_use_triggers_extraction_at_threshold tests/test_hooks.py::test_post_tool_use_extraction_respects_min_turn_gap -v`
Expected: FAIL (extraction logic not wired yet)

- [ ] **Step 3: Wire extraction into `on_post_tool_use` in `hooks.py`**

Add import at top of `hooks.py`:

```python
from .extraction import should_extract, build_extraction_prompt
```

Add closure state in `build_default_hooks`, alongside `_tool_call_count` and `_completed_skills`:

```python
    _last_extraction_turn = [0]
    _total_result_chars = [0]
```

Add extraction check in `on_post_tool_use`, after the skill-completion detection block and before the reminder reinjection block:

```python
        # Track result size for extraction threshold
        _total_result_chars[0] += len(result_text)

        # Memory extraction check
        if config.enable_hybrid_memory and should_extract(
            tool_call_count=_tool_call_count[0],
            total_chars=_total_result_chars[0],
            last_extraction_turn=_last_extraction_turn[0],
            current_turn=_tool_call_count[0],
            tool_call_interval=config.extraction_tool_call_interval,
            char_threshold=config.extraction_char_threshold,
            min_turn_gap=config.extraction_min_turn_gap,
        ):
            _last_extraction_turn[0] = _tool_call_count[0]
            _total_result_chars[0] = 0
            extraction_prompt = build_extraction_prompt(
                memory_dir=str(memory_store.memory_dir),
                project_root=str(memory_store.project_root),
            )
            return {"additionalContext": extraction_prompt}
```

Note: The extraction check must come after skill-completion (which has higher priority) but before reminder reinjection. The `result_text` variable is already computed later in the function — you need to move the `_stringify_result` call and the `result_text` assignment earlier (before the extraction block), or compute `_total_result_chars` from `tool_result` directly. The simplest approach: move the `result_text = _stringify_result(tool_result)` line to immediately after `_tool_call_count[0] += 1`, before the skill-completion block. Then the early return for noisy tools, shell failures, etc. all still work, and `_total_result_chars` accumulates correctly.

- [ ] **Step 4: Run all hook tests**

Run: `cd /c/Users/16026/Documents/Code/claude-code && python -m pytest tests/test_hooks.py -v`
Expected: All passed (including the 2 new extraction tests)

- [ ] **Step 5: Commit**

```bash
cd /c/Users/16026/Documents/Code/claude-code
git add copilotcode_sdk/hooks.py tests/test_hooks.py
git commit -m "feat: wire memory extraction trigger into post_tool_use hook"
```

---

### Task 5: Compaction prompt and handoff context (`compaction.py`)

**Files:**
- Create: `copilotcode_sdk/compaction.py`
- Create: `tests/test_compaction.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_compaction.py
from __future__ import annotations

from pathlib import Path

from copilotcode_sdk.compaction import build_compaction_prompt, build_handoff_context


def test_build_compaction_prompt_contains_six_summary_points() -> None:
    prompt = build_compaction_prompt()
    # The prompt should instruct the model to produce a 6-point summary
    assert "original goal" in prompt.lower() or "primary request" in prompt.lower()
    assert "key findings" in prompt.lower() or "discoveries" in prompt.lower()
    assert "current state" in prompt.lower() or "progress" in prompt.lower()
    assert "remaining" in prompt.lower() or "next" in prompt.lower()


def test_build_handoff_context_includes_all_sections(tmp_path: Path) -> None:
    context = build_handoff_context(
        compaction_summary="User is analyzing Q1 data.",
        skill_catalog_text="| Skill | Desc |\n|---|---|\n| intake | Ingest |",
        instruction_content="Use DuckDB for queries.",
        memory_index="- [profile](profile.md) - user prefs",
    )
    assert "Q1 data" in context
    assert "intake" in context
    assert "DuckDB" in context
    assert "profile" in context


def test_build_handoff_context_omits_empty_sections(tmp_path: Path) -> None:
    context = build_handoff_context(
        compaction_summary="Analyzing data.",
        skill_catalog_text="",
        instruction_content="",
        memory_index="",
    )
    assert "Analyzing data." in context
    assert "Skill Catalog" not in context
    assert "Workspace Instructions" not in context
    assert "Memory Index" not in context
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /c/Users/16026/Documents/Code/claude-code && python -m pytest tests/test_compaction.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `compaction.py`**

```python
# copilotcode_sdk/compaction.py
from __future__ import annotations


def build_compaction_prompt() -> str:
    """Build the prompt used to summarize the session before context compaction.

    The model should produce a structured summary covering 6 points so
    that a fresh context window can resume the work without loss.
    """
    return (
        "The conversation context is being compacted. Produce a structured summary "
        "that preserves everything a fresh context window would need to continue "
        "this work seamlessly. Cover these points:\n\n"
        "1. **Primary request and intent** — what the user originally asked for and why\n"
        "2. **Key findings and discoveries** — important facts, data patterns, or decisions made\n"
        "3. **Current state of progress** — what has been completed, what files were created or modified\n"
        "4. **Remaining work** — what still needs to be done, in priority order\n"
        "5. **Active assumptions and caveats** — interpretive decisions, data quality issues, open questions\n"
        "6. **Critical context** — anything else that would be lost without this summary "
        "(column mappings, metric values, error patterns, user preferences expressed during the session)\n\n"
        "Be specific and concrete. Include file paths, metric values, and exact names. "
        "A vague summary is worse than no summary."
    )


def build_handoff_context(
    *,
    compaction_summary: str,
    skill_catalog_text: str = "",
    instruction_content: str = "",
    memory_index: str = "",
) -> str:
    """Assemble the full handoff context injected after compaction.

    Combines the compaction summary with durable context (skill catalog,
    workspace instructions, memory index) so the fresh context window
    has everything it needs.
    """
    sections = [
        "## Session Continuation\n",
        "The previous context was compacted. Below is the preserved context.\n",
        "### Compaction Summary\n",
        compaction_summary,
    ]

    if skill_catalog_text:
        sections.append("\n### Skill Catalog\n")
        sections.append(skill_catalog_text)

    if instruction_content:
        sections.append("\n### Workspace Instructions\n")
        sections.append(instruction_content)

    if memory_index:
        sections.append("\n### Memory Index\n")
        sections.append(memory_index)

    return "\n".join(sections)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /c/Users/16026/Documents/Code/claude-code && python -m pytest tests/test_compaction.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
cd /c/Users/16026/Documents/Code/claude-code
git add copilotcode_sdk/compaction.py tests/test_compaction.py
git commit -m "feat: add compaction prompt and handoff context builder"
```

---

### Task 6: Prompt suggestions in hooks

**Files:**
- Modify: `copilotcode_sdk/hooks.py`
- Modify: `tests/test_hooks.py`

- [ ] **Step 1: Write failing tests for prompt suggestions**

```python
# append to tests/test_hooks.py

def test_post_tool_use_suggests_available_skills(tmp_path: Path) -> None:
    """After enough tool calls, suggest skills whose prerequisites are met."""
    intake_dir = tmp_path / "skills" / "intake"
    intake_dir.mkdir(parents=True)
    (intake_dir / "SKILL.md").write_text(
        "---\nname: intake\ndescription: Ingest.\ntype: data-intake\n"
        "outputs: outputs/intake/\nrequires: none\n---\n\n# Intake\n",
        encoding="utf-8",
    )
    analysis_dir = tmp_path / "skills" / "analysis"
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "SKILL.md").write_text(
        "---\nname: analysis\ndescription: Analyze.\ntype: rcm-analysis\n"
        "outputs: outputs/analysis/\nrequires: data-intake\n---\n\n# Analysis\n",
        encoding="utf-8",
    )

    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        extraction_tool_call_interval=999,  # disable extraction
        extraction_min_turn_gap=999,
        reminder_reinjection_interval=0,  # disable reminder
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    _, skill_map = build_skill_catalog([str(tmp_path / "skills")])
    hooks = build_default_hooks(config, store, skill_map=skill_map)

    # Simulate 4 tool calls (above the 3-turn threshold for suggestions)
    for i in range(3):
        hooks["on_post_tool_use"](
            {"toolName": "read", "toolResult": "ok"},
            {},
        )

    # 4th call — should suggest "intake" since it has no prerequisites
    result = hooks["on_post_tool_use"](
        {"toolName": "read", "toolResult": "ok"},
        {},
    )

    assert result is not None
    ctx = result["additionalContext"]
    assert "intake" in ctx.lower()


def test_post_tool_use_no_suggestion_before_threshold(tmp_path: Path) -> None:
    """No suggestion on the very first tool call."""
    skill_dir = tmp_path / "skills" / "intake"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: intake\ndescription: Ingest.\ntype: data-intake\n"
        "outputs: outputs/intake/\nrequires: none\n---\n\n# Intake\n",
        encoding="utf-8",
    )

    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        extraction_tool_call_interval=999,
        extraction_min_turn_gap=999,
        reminder_reinjection_interval=0,
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    _, skill_map = build_skill_catalog([str(tmp_path / "skills")])
    hooks = build_default_hooks(config, store, skill_map=skill_map)

    result = hooks["on_post_tool_use"](
        {"toolName": "read", "toolResult": "ok"},
        {},
    )

    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /c/Users/16026/Documents/Code/claude-code && python -m pytest tests/test_hooks.py::test_post_tool_use_suggests_available_skills tests/test_hooks.py::test_post_tool_use_no_suggestion_before_threshold -v`
Expected: FAIL

- [ ] **Step 3: Implement prompt suggestions in `hooks.py`**

Add a helper function `_find_suggestable_skills` in `hooks.py`:

```python
SUGGESTION_TURN_THRESHOLD = 3
SUGGESTION_INTERVAL = 15  # Don't suggest more often than every 15 calls


def _find_suggestable_skills(
    skill_map: dict[str, dict[str, str]],
    completed_skills: set[str],
) -> list[str]:
    """Find skills that are not started and whose prerequisites are met."""
    completed_types = {
        fm.get("type", "")
        for name, fm in skill_map.items()
        if name in completed_skills and fm.get("type")
    }

    suggestable = []
    for name, fm in skill_map.items():
        if name in completed_skills:
            continue
        requires = fm.get("requires", "").strip().lower()
        if not requires or requires == "none" or requires in completed_types:
            suggestable.append(name)
    return suggestable
```

Add suggestion logic in `on_post_tool_use`, after the extraction block and after the reminder reinjection block (as a low-priority fallback — only fires when nothing else returned):

In `build_default_hooks`, add closure state:
```python
    _last_suggestion_turn = [0]
```

At the end of `on_post_tool_use`, before the final `return None`:

```python
        # Prompt suggestion: nudge toward available skills
        if (
            skill_map
            and _tool_call_count[0] > SUGGESTION_TURN_THRESHOLD
            and _tool_call_count[0] - _last_suggestion_turn[0] >= SUGGESTION_INTERVAL
        ):
            suggestable = _find_suggestable_skills(skill_map, _completed_skills)
            if suggestable:
                _last_suggestion_turn[0] = _tool_call_count[0]
                names = ", ".join(suggestable)
                return {
                    "additionalContext": (
                        f"Reminder: the following skills are available and their prerequisites are met: {names}. "
                        "Check the skill catalog if you haven't started these yet."
                    ),
                }
```

- [ ] **Step 4: Run all hook tests**

Run: `cd /c/Users/16026/Documents/Code/claude-code && python -m pytest tests/test_hooks.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
cd /c/Users/16026/Documents/Code/claude-code
git add copilotcode_sdk/hooks.py tests/test_hooks.py
git commit -m "feat: add prompt suggestions for available skills in hooks"
```

---

### Task 7: Export new public names from `__init__.py`

**Files:**
- Modify: `copilotcode_sdk/__init__.py`

- [ ] **Step 1: Write failing test**

```python
# This can be verified inline — check import works
# python -c "from copilotcode_sdk import should_extract, build_extraction_prompt, build_compaction_prompt, build_handoff_context"
```

- [ ] **Step 2: Add exports to `__init__.py`**

Add imports:

```python
from .extraction import build_extraction_prompt, should_extract
from .compaction import build_compaction_prompt, build_handoff_context
```

Add to `__all__`:

```python
    "build_compaction_prompt",
    "build_extraction_prompt",
    "build_handoff_context",
    "should_extract",
```

- [ ] **Step 3: Verify imports**

Run: `cd /c/Users/16026/Documents/Code/claude-code && python -c "from copilotcode_sdk import should_extract, build_extraction_prompt, build_compaction_prompt, build_handoff_context; print('OK')"` 
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
cd /c/Users/16026/Documents/Code/claude-code
git add copilotcode_sdk/__init__.py
git commit -m "feat: export extraction and compaction public API"
```

---

### Task 8: Full test suite validation

**Files:** None (read-only)

- [ ] **Step 1: Run the full test suite**

Run: `cd /c/Users/16026/Documents/Code/claude-code && python -m pytest tests/ -v --tb=short`
Expected: All tests pass (existing + new). No regressions.

- [ ] **Step 2: Verify test count**

Confirm the new tests are counted:
- `test_extraction.py`: 8 tests
- `test_compaction.py`: 3 tests
- `test_hooks.py`: existing 16 + 4 new = 20 tests
- `test_config.py`: existing + 1 new

Total new tests: 16

- [ ] **Step 3: Commit (if any fixes were needed)**

```bash
cd /c/Users/16026/Documents/Code/claude-code
git add -A
git commit -m "fix: resolve test suite issues from phase 3 integration"
```

---

### Task 9: Back-to-back experiment validation

**Files:** None (integration test, not automated)

- [ ] **Step 1: Run first experiment in `harness_3_experiments`**

Run the experiment with a small dataset. Verify:
- All 3 skills chain (intake -> analysis -> report)
- Memory extraction fires at least once (check for extraction prompt in logs)
- Memory files are written to the memory directory

- [ ] **Step 2: Run second experiment on a different dataset**

Run a second experiment on a different file. Verify:
- The memory from run 1 is injected into the session context
- The agent benefits from prior learnings (e.g., column mapping conventions)

- [ ] **Step 3: Document results**

Write a brief summary of the back-to-back test results — what worked, what didn't, any adjustments needed.
