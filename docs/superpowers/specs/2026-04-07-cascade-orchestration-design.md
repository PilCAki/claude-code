# Cascade Orchestration Exercise — Multi-Subsystem Interaction Tests

## Context

The exercise system has three tiers:
- **Subsystem** (17 tests): API-level imports and construction
- **Basic orchestration** (6 tests): Session viability — tool pipeline, error recovery, multi-turn coherence
- **Advanced orchestration** (7 tests): Single reactive behaviors — extraction nudge, git safety gate, caching, task reminders, context warning, read-before-write, repeat detection

None of these test what happens when **two or more reactive behaviors interact in the same turn**. The SDK's hook system processes tool results sequentially and returns early on the first `additionalContext` injection. This means when multiple thresholds are crossed simultaneously, only one behavior fires — and which one wins depends on code ordering, not priority. These interactions represent real integration risks.

This spec adds a **cascade tier** of 7 scenarios that provoke multi-subsystem interactions and observe the results.

## Design

### Cascade Config

A new `build_cascade_config(base)` function applies more aggressive threshold overrides than the advanced tier. The goal is to make it possible to cross **multiple thresholds within 3-5 tool calls**.

| Field | Advanced | Cascade | Why |
|-------|---------|---------|-----|
| `extraction_tool_call_interval` | 5 | 3 | Must fire alongside other thresholds |
| `extraction_char_threshold` | 10,000 | 5,000 | Lower char threshold for faster trigger |
| `extraction_min_turn_gap` | 2 | 1 | Allow back-to-back extraction |
| `task_reminder_turns` | 3 | 2 | Must fire before extraction to test collision |
| `task_reminder_cooldown_turns` | 3 | 2 | Faster cooldown |
| `enable_tool_result_cache` | True | True | Same |
| `reminder_reinjection_interval` | 5 | 3 | Faster reminders |
| `max_context_chars` | 50,000 | 20,000 | One file read crosses 80% |
| `noisy_tool_char_limit` | 8,000 | 5,000 | Lower to make truncation interact |

### New Mode: `"cascade"`

`ExerciseMode` becomes `Literal["subsystem", "orchestration", "advanced", "cascade", "full"]`.

- `"cascade"` — builds cascade config, creates client, runs cascade prompt
- `"full"` — runs all four tiers sequentially, merging results

### New Hook Accessor

Add `get_compaction_warned() -> bool` to `build_default_hooks()` return dict. Returns `_compaction_warned[0]`. Needed for scenario #7 ground truth.

## Scenarios (7)

### 1. context_accounting_desync

**What it tests:** Cached tool results skip `on_post_tool_use`, so context size estimation is not updated on cache hits.

**Provoke:**
1. Read `copilotcode_sdk/config.py` (~6K chars) — tracked + cached
2. Read it again identically — cache hit, `on_post_tool_use` skipped
3. Read it a third time — still cached

**Agent observes:** Whether the session behaves as if context grew by 6K (one read) or 18K (three reads). If context warning fires after just 3 reads of a 6K file with a 20K limit, the cache bypass is NOT happening (context tracked 3x). If it doesn't fire, the cache bypass IS happening (context tracked 1x).

**Ground truth:** `estimated_context_chars` should show ~6K (1x), not ~18K (3x). `tool_result_cache_size > 0`.

### 2. injection_priority_collision

**What it tests:** When context warning threshold AND extraction nudge threshold are both crossed in the same `on_post_tool_use` call, the context warning fires first (line 571) and the extraction nudge (line 743) is never reached.

**Provoke:**
1. Make 2 small tool calls (echo commands) to approach extraction threshold (3 calls)
2. Read a large file (~20K) that pushes estimated context past 80% of 20K limit AND is the 3rd tool call

**Agent observes:** Which `additionalContext` appears — a context/compaction warning mentioning "capacity" or an extraction nudge mentioning "memory checkpoint"? Only one should appear.

**Ground truth:** `estimated_context_chars > 16_000` (80% of 20K) AND `tool_call_count >= 3`. Both thresholds crossed but only context warning injected.

### 3. cache_invalidation_cascade

**What it tests:** Writing a file clears the entire tool result cache (`_tool_result_cache.clear()`) but does NOT reset `_estimated_context_chars`. Subsequent re-reads add to the estimate again, causing double-counting.

**Provoke:**
1. Read `copilotcode_sdk/__init__.py` (~2K) — cached + context tracked
2. Write `/tmp/copilotcode_cascade_test.txt` — cache cleared, context estimate unchanged
3. Read `copilotcode_sdk/__init__.py` again — fresh read, context tracked AGAIN

**Agent observes:** Whether tool results differ between step 1 and step 3 (step 1 was cached, step 3 is fresh). Whether any context injection changes.

**Ground truth:** `tool_result_cache_size == 0` after write (step 2), then `tool_result_cache_size == 1` after re-read (step 3). `estimated_context_chars` shows ~4K (2x the file size, not 1x) because the estimate accumulated both reads.

### 4. error_skip_vs_session_continuity

**What it tests:** Tool execution errors are handled with `errorHandling: "skip"` (not retried), while the session continues normally. Multiple consecutive tool errors should not corrupt session state or trigger unrelated cascade behaviors (e.g., extraction nudge, task reminder).

**Provoke:**
1. Try to read a nonexistent file `/nonexistent/cascade_test_1.txt` — error, skipped
2. Try to read another nonexistent file `/nonexistent/cascade_test_2.txt` — error, skipped
3. Read a real file `copilotcode_sdk/__init__.py` — should succeed normally
4. Run `echo recovery_confirmed` — should succeed normally

**Agent observes:** Both errors return clean error messages (not crashes). Steps 3 and 4 succeed. No unexpected `additionalContext` injections from the errors (errors should not count toward extraction or task reminder thresholds since they were skipped).

**Ground truth:** Event bus has `tool_result` events for all 4 calls. `tool_call_count` reflects all 4 calls (including errors). Session state is intact.

### 5. task_reminder_vs_extraction_race

**What it tests:** When task reminder threshold (2 non-task turns) AND extraction threshold (3 tool calls) are both crossed, which injection wins?

**Provoke:**
1. Create a task via TaskCreate
2. Run `echo a` (non-task call #1, tool call #2)
3. Run `echo b` (non-task call #2, tool call #3 — both thresholds crossed)

**Agent observes:** Which `additionalContext` appears on the 3rd call — a task reminder mentioning "open tasks" or an extraction nudge mentioning "memory checkpoint"?

**Ground truth:** `tool_call_count >= 3`. Task store has open task. Both thresholds crossed but only one injection returned.

### 6. repeat_detection_ring_overflow

**What it tests:** The repeat detection ring buffer holds the last 10 shell commands. After 10 different commands push an earlier command out of the ring, repeating that command is no longer detected.

**Provoke:**
1. Run `echo sentinel_cascade_test` — first occurrence, no warning
2. Run `echo sentinel_cascade_test` — repeat, expect loop warning
3. Run 10 different `echo` commands (`echo flush_1` through `echo flush_10`) to fill the ring
4. Run `echo sentinel_cascade_test` — should NOT trigger warning (pushed out of ring)

**Agent observes:** Warning on step 2 (expected), no warning on step 4 (ring overflow).

**Ground truth:** `recent_shell` after step 4 should NOT contain `sentinel_cascade_test` from steps 1-2 (they were pushed out by the 10 flush commands).

### 7. context_warning_single_fire

**What it tests:** The `_compaction_warned` flag prevents context warnings from firing more than once per session, even if context continues to grow past the threshold.

**Provoke:**
1. Read a large file (~20K) to cross the 80% threshold (16K of 20K limit) — warning fires
2. Read another file — context grows further, but NO second warning
3. Read a third file — still no warning

**Agent observes:** Context warning appears exactly once (step 1). Steps 2 and 3 have no context warning despite growing context.

**Ground truth:** `compaction_warned == True`. `estimated_context_chars` continues to grow across all reads.

## Files Modified

| File | Changes |
|------|---------|
| `copilotcode_sdk/hooks.py` | Add `get_compaction_warned()` accessor to return dict |
| `copilotcode_sdk/exercise.py` | `CASCADE_ORCHESTRATION_SCENARIOS`, `build_cascade_config()`, `build_cascade_orchestration_prompt()`, `"cascade"` in ExerciseMode, updated `run_exercise()` |
| `copilotcode_sdk/cli.py` | Add `"cascade"` to `--mode` choices |
| `tests/test_exercise.py` | ~7 new tests |

## Verification

1. `pytest tests/test_exercise.py -x -q` — all tests pass
2. `pytest tests/ -x -q` — full suite passes
3. `copilotcode exercise --mode cascade --json --permission-policy approve_all` — agent provokes cascades, observes results
4. Review ground truth — multi-threshold states confirmed
5. `copilotcode exercise --mode full` — all four tiers run
