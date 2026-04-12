# Cascade Orchestration Exercise Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "cascade" exercise tier with 7 scenarios that test multi-subsystem interactions where two or more reactive hook behaviors compete or interfere within the same turn.

**Architecture:** New `CASCADE_ORCHESTRATION_SCENARIOS` list, `build_cascade_config()` factory (more aggressive thresholds than advanced), `build_cascade_orchestration_prompt()`, and `"cascade"` mode in `run_exercise()`. One new hook accessor `get_compaction_warned()`. Ground truth capture already handles the rest.

**Tech Stack:** Python 3.10+, dataclasses, pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `copilotcode_sdk/hooks.py` | Modify | Add `get_compaction_warned()` accessor |
| `copilotcode_sdk/exercise.py` | Modify | Scenarios, config builder, prompt, mode support, ground truth |
| `copilotcode_sdk/cli.py` | Modify | Add `"cascade"` to `--mode` choices |
| `tests/test_exercise.py` | Modify | 7 new tests |

---

### Task 1: Add `get_compaction_warned` hook accessor

**Files:**
- Modify: `copilotcode_sdk/hooks.py:977-996`
- Test: `tests/test_exercise.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_exercise.py`, add at the bottom:

```python
def test_capture_ground_truth_compaction_warned() -> None:
    from copilotcode_sdk.events import EventBus

    class FakeSession:
        event_bus = EventBus()
        _state = None
        _completed_skills = None
        _raw_hooks = {
            "get_tool_call_count": lambda: 0,
            "get_recent_shell": lambda: [],
            "get_read_file_state": lambda: {},
            "get_estimated_context_chars": lambda: 0,
            "get_tool_result_cache_size": lambda: 0,
            "get_compaction_warned": lambda: True,
            "get_token_budget": lambda: None,
            "get_file_changes": lambda: {},
        }

    gt = _capture_ground_truth(FakeSession())
    assert gt["compaction_warned"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_exercise.py::test_capture_ground_truth_compaction_warned -v`
Expected: FAIL — `KeyError: 'compaction_warned'`

- [ ] **Step 3: Add accessor to hooks.py**

In `copilotcode_sdk/hooks.py`, add this closure right after `get_tool_result_cache_size`:

```python
    def get_compaction_warned() -> bool:
        """Accessor for whether the compaction/context warning has already fired."""
        return _compaction_warned[0]
```

And add it to the return dict:

```python
        "get_tool_result_cache_size": get_tool_result_cache_size,
        "get_compaction_warned": get_compaction_warned,
    }
```

- [ ] **Step 4: Add ground truth capture for compaction_warned**

In `copilotcode_sdk/exercise.py`, in `_capture_ground_truth()`, after the `file_changes` block (around line 854), add:

```python
        compaction_warned = _safe_call(raw_hooks.get("get_compaction_warned"))
        if compaction_warned is not None:
            ground_truth["compaction_warned"] = compaction_warned
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_exercise.py::test_capture_ground_truth_compaction_warned -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add copilotcode_sdk/hooks.py copilotcode_sdk/exercise.py tests/test_exercise.py
git commit -m "feat: add get_compaction_warned hook accessor for cascade ground truth"
```

---

### Task 2: Add `CASCADE_ORCHESTRATION_SCENARIOS` and `build_cascade_config`

**Files:**
- Modify: `copilotcode_sdk/exercise.py`
- Test: `tests/test_exercise.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_exercise.py`, update the imports to include the new symbols:

```python
from copilotcode_sdk.exercise import (
    ExerciseReport,
    SubsystemResult,
    SUBSYSTEM_CHECKLIST,
    ORCHESTRATION_SCENARIOS,
    ADVANCED_ORCHESTRATION_SCENARIOS,
    CASCADE_ORCHESTRATION_SCENARIOS,
    build_exercise_prompt,
    build_orchestration_prompt,
    build_advanced_orchestration_prompt,
    build_cascade_orchestration_prompt,
    build_exercise_config,
    build_cascade_config,
    parse_exercise_report,
    _capture_ground_truth,
)
```

Then add tests at the bottom:

```python
def test_cascade_orchestration_scenarios_exist() -> None:
    assert len(CASCADE_ORCHESTRATION_SCENARIOS) == 7
    names = [s["name"] for s in CASCADE_ORCHESTRATION_SCENARIOS]
    assert "context_accounting_desync" in names
    assert "injection_priority_collision" in names
    assert "cache_invalidation_cascade" in names
    assert "error_skip_vs_session_continuity" in names
    assert "task_reminder_vs_extraction_race" in names
    assert "repeat_detection_ring_overflow" in names
    assert "context_warning_single_fire" in names


def test_build_cascade_config_thresholds() -> None:
    cfg = build_cascade_config()
    assert cfg.extraction_tool_call_interval == 3
    assert cfg.extraction_char_threshold == 5_000
    assert cfg.extraction_min_turn_gap == 1
    assert cfg.task_reminder_turns == 2
    assert cfg.task_reminder_cooldown_turns == 2
    assert cfg.enable_tool_result_cache is True
    assert cfg.reminder_reinjection_interval == 3
    assert cfg.max_context_chars == 20_000
    assert cfg.noisy_tool_char_limit == 5_000


def test_build_cascade_config_preserves_base() -> None:
    from copilotcode_sdk.config import CopilotCodeConfig

    base = CopilotCodeConfig(model="claude-opus-4-20250514")
    cfg = build_cascade_config(base)
    assert cfg.model == "claude-opus-4-20250514"
    assert cfg.extraction_tool_call_interval == 3  # overridden
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_exercise.py::test_cascade_orchestration_scenarios_exist tests/test_exercise.py::test_build_cascade_config_thresholds tests/test_exercise.py::test_build_cascade_config_preserves_base -v`
Expected: FAIL — `ImportError: cannot import name 'CASCADE_ORCHESTRATION_SCENARIOS'`

- [ ] **Step 3: Add `CASCADE_ORCHESTRATION_SCENARIOS` to exercise.py**

In `copilotcode_sdk/exercise.py`, after the closing `]` of `ADVANCED_ORCHESTRATION_SCENARIOS` (line 622) and before `build_exercise_config`, add:

```python
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
            "`additionalContext`. With a 20K limit and 80% threshold (16K), three "
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
```

- [ ] **Step 4: Add `build_cascade_config` to exercise.py**

After the closing `]` of `CASCADE_ORCHESTRATION_SCENARIOS` and before `build_exercise_config`, add:

```python
def build_cascade_config(
    base: "CopilotCodeConfig | None" = None,
) -> "CopilotCodeConfig":
    """Build a CopilotCodeConfig with aggressive thresholds for cascade exercise scenarios.

    Thresholds are lower than ``build_exercise_config`` so that multiple
    reactive behaviors can be crossed within 3-5 tool calls.
    """
    from .config import CopilotCodeConfig

    if base is None:
        base = CopilotCodeConfig()
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_exercise.py::test_cascade_orchestration_scenarios_exist tests/test_exercise.py::test_build_cascade_config_thresholds tests/test_exercise.py::test_build_cascade_config_preserves_base -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add copilotcode_sdk/exercise.py tests/test_exercise.py
git commit -m "feat: add CASCADE_ORCHESTRATION_SCENARIOS and build_cascade_config"
```

---

### Task 3: Add `build_cascade_orchestration_prompt`

**Files:**
- Modify: `copilotcode_sdk/exercise.py`
- Test: `tests/test_exercise.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_cascade_orchestration_prompt_self_aware() -> None:
    prompt = build_cascade_orchestration_prompt()
    assert "self-aware" in prompt
    assert "PROVOKE" in prompt
    assert "OBSERVE" in prompt
    assert "cascade" in prompt.lower()
    assert "exercise-report" in prompt
    for scenario in CASCADE_ORCHESTRATION_SCENARIOS:
        assert scenario["name"] in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_exercise.py::test_build_cascade_orchestration_prompt_self_aware -v`
Expected: FAIL — `ImportError: cannot import name 'build_cascade_orchestration_prompt'`

- [ ] **Step 3: Add the prompt builder to exercise.py**

In `copilotcode_sdk/exercise.py`, after `build_advanced_orchestration_prompt` (after line 794), add:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_exercise.py::test_build_cascade_orchestration_prompt_self_aware -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add copilotcode_sdk/exercise.py tests/test_exercise.py
git commit -m "feat: add build_cascade_orchestration_prompt"
```

---

### Task 4: Add `"cascade"` mode to `ExerciseMode` and `run_exercise`

**Files:**
- Modify: `copilotcode_sdk/exercise.py`
- Test: `tests/test_exercise.py`

- [ ] **Step 1: Write the failing test**

```python
def test_cascade_mode_in_exercise_mode() -> None:
    from copilotcode_sdk.exercise import ExerciseMode
    # Verify "cascade" is a valid literal value
    import typing
    args = typing.get_args(ExerciseMode)
    assert "cascade" in args
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_exercise.py::test_cascade_mode_in_exercise_mode -v`
Expected: FAIL — `"cascade"` not in args

- [ ] **Step 3: Update ExerciseMode**

In `copilotcode_sdk/exercise.py`, change the `ExerciseMode` definition:

From:
```python
ExerciseMode = Literal["subsystem", "orchestration", "advanced", "full"]
```

To:
```python
ExerciseMode = Literal["subsystem", "orchestration", "advanced", "cascade", "full"]
```

- [ ] **Step 4: Update `run_exercise` docstring and add cascade branch**

Update the docstring:
```python
    """Create a real session, send the exercise prompt, parse the report.

    Modes:
    - ``subsystem``: Run the 17 API-level subsystem checks
    - ``orchestration``: Run the 6 basic wiring/provocation scenarios
    - ``advanced``: Run the 7 advanced reactive-behavior scenarios (lowered thresholds)
    - ``cascade``: Run the 7 cascade multi-subsystem interaction scenarios (aggressive thresholds)
    - ``full``: Run all four sequentially, merging results into one report
    """
```

Add a new `elif mode == "cascade"` branch right after the `elif mode == "advanced"` block and before `elif mode == "orchestration"`:

```python
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
```

Update the `mode == "full"` branch to include cascade. After the advanced report block, add:

```python
        # Cascade mode uses even more aggressive thresholds
        cascade_cfg = build_cascade_config(client.config)
        cascade_client = CopilotCodeClient(cascade_cfg)
        cas_id = f"{client.config.brand.slug}-exercise-cas-{uuid.uuid4().hex[:12]}"
        cas_prompt = build_cascade_orchestration_prompt()
        cas_report = await _run_single_exercise(
            cascade_client, mode="cascade", prompt=cas_prompt,
            timeout=timeout, session_id=cas_id, timestamp=timestamp,
        )
```

Update the merge to include cascade results:

```python
        all_subsystems = (
            sub_report.subsystems
            + orch_report.subsystems
            + adv_report.subsystems
            + cas_report.subsystems
        )
        total_duration = (
            sub_report.total_duration_seconds
            + orch_report.total_duration_seconds
            + adv_report.total_duration_seconds
            + cas_report.total_duration_seconds
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
                f"Cascade: {cas_report.summary}"
            ),
            total_duration_seconds=total_duration,
            ground_truth=cas_report.ground_truth,
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_exercise.py::test_cascade_mode_in_exercise_mode -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add copilotcode_sdk/exercise.py tests/test_exercise.py
git commit -m "feat: add cascade mode to ExerciseMode and run_exercise"
```

---

### Task 5: Update CLI `--mode` choices

**Files:**
- Modify: `copilotcode_sdk/cli.py:120-125`
- Test: `tests/test_exercise.py`

- [ ] **Step 1: Write the failing test**

```python
def test_cli_exercise_cascade_mode() -> None:
    from copilotcode_sdk.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["exercise", "--mode", "cascade"])
    assert args.mode == "cascade"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_exercise.py::test_cli_exercise_cascade_mode -v`
Expected: FAIL — `error: argument --mode: invalid choice: 'cascade'`

- [ ] **Step 3: Update CLI choices**

In `copilotcode_sdk/cli.py`, change line 122:

From:
```python
        choices=("subsystem", "orchestration", "advanced", "full"),
```

To:
```python
        choices=("subsystem", "orchestration", "advanced", "cascade", "full"),
```

And update the help text on line 124:

From:
```python
        help="Exercise mode: subsystem (API checks), orchestration (basic wiring), advanced (reactive behaviors), or full (all).",
```

To:
```python
        help="Exercise mode: subsystem (API checks), orchestration (basic wiring), advanced (reactive behaviors), cascade (multi-subsystem interactions), or full (all).",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_exercise.py::test_cli_exercise_cascade_mode -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add copilotcode_sdk/cli.py tests/test_exercise.py
git commit -m "feat: add cascade to CLI --mode choices"
```

---

### Task 6: Run full test suite and verify

**Files:**
- Test: `tests/test_exercise.py`, `tests/`

- [ ] **Step 1: Run exercise tests**

Run: `pytest tests/test_exercise.py -x -q`
Expected: All tests pass (26 existing + 7 new = 33)

- [ ] **Step 2: Run affected module tests**

Run: `pytest tests/test_exercise.py tests/test_config.py tests/test_hooks.py tests/test_client.py tests/test_cli.py -x -q`
Expected: All pass, no regressions

- [ ] **Step 3: Final commit if any fixups needed**

```bash
git add -u
git commit -m "fix: test suite fixups for cascade orchestration"
```
