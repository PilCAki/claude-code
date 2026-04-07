# Memory Lifecycle Exercise Design Spec

## Problem

The memory system (MemoryStore, session memory, SessionMemoryController) has unit tests covering the data layer and threshold logic, but nothing tests that memory actually works end-to-end in a real session. Real-world usage — recurring data analysis across sessions — consistently breaks because:
- Memory write calls don't persist to disk
- Session memory extraction never triggers
- Memories exist on disk but aren't loaded into context for the next session
- Agent fabricates prior findings instead of recalling them

We need an LLM-based exercise that proves the system can learn over time across sessions.

## Solution

One new chain exercise (`memory_lifecycle`) added to the existing micro/chain exercise system in `micro_exercise.py`. It runs two real sessions against the same memory root, simulating a recurring data analysis workflow where session 2 must build on session 1's findings.

## Scenario: Recurring Sales Analysis

### Fixtures

**sales_q1.csv** (8 rows):
```
region,product,revenue,units
Northeast,Widget A,4200,84
Northeast,Widget B,3100,62
Southeast,Widget A,5800,116
Southeast,Widget B,890,18
Midwest,Widget A,3600,72
Midwest,Widget B,2200,44
West,Widget A,6100,122
West,Widget B,1500,30
```

Pre-computed Q1 ground truth:
- Region totals: Northeast $7,300, Southeast $6,690, Midwest $5,800, West $7,600
- Top region: West ($7,600)
- Low-revenue product (below $1000): Southeast Widget B ($890)

**sales_q2.csv** (8 rows, same structure, different values):
```
region,product,revenue,units
Northeast,Widget A,4800,96
Northeast,Widget B,3500,70
Southeast,Widget A,6200,124
Southeast,Widget B,1400,28
Midwest,Widget A,3100,62
Midwest,Widget B,1900,38
West,Widget A,5900,118
West,Widget B,1600,32
```

Pre-computed Q2 ground truth:
- Region totals: Northeast $8,300, Southeast $7,600, Midwest $5,000, West $7,500
- Top region: Northeast ($8,300)
- Deltas vs Q1: Northeast +$1,000, Southeast +$910, Midwest -$800, West -$100
- Southeast Widget B recovered from $890 to $1,400 (no longer below $1,000)

### Stakeholder Questions (Injected in Session 1)

These are included in the step 1 prompt but NOT repeated in step 4. The agent must recall them from memory.

1. "The VP of Sales asked: is the Southeast region viable long-term or should we consider reallocating its budget?"
2. "Marketing wants to know: are Widget B sales declining across all regions or just specific ones?"

## Exercise Steps

### Step 1 — Analyze Q1 and Save Findings

**Setup:** Write `sales_q1.csv` to temp dir.

**Prompt:**
> Read sales_q1.csv and analyze the data. Identify the top region by total revenue and any products with revenue below $1000.
>
> Two stakeholder questions came up during this analysis:
> 1. The VP of Sales asked: is the Southeast region viable long-term or should we consider reallocating its budget?
> 2. Marketing wants to know: are Widget B sales declining across all regions or just specific ones?
>
> Answer both questions based on the data. Then save your key findings, analytical conclusions, and these open stakeholder questions to memory so future sessions can build on this work.

**Rubric:** Agent read the file. Identified West as top region ($7,600). Identified Southeast Widget B ($890) as below $1,000. Answered both stakeholder questions. Called a memory-save tool or wrote a memory file. Findings must be based on actual CSV data.

**Ground truth:** Pre-computed Q1 answers.

### Step 2 — Verify Memory Content

**Prompt:**
> What findings have been saved to memory about this project? List them.

**Rubric:** Agent accessed its memory system and reported findings that include: (a) West as the top region by revenue, (b) Southeast Widget B as the low-revenue outlier, (c) the VP's question about Southeast viability, (d) Marketing's question about Widget B trends. The memory must contain derived analytical conclusions and stakeholder questions — NOT a copy of raw CSV data. If the memory is just a data dump of the CSV, this fails.

**Ground truth:** Pre-computed Q1 findings + contents of any `.md` files in the memory directory.

### Step 3 — Trigger Extraction and Remove Q1 Data

**Prompt:**
> Read sales_q1.csv again and calculate the average revenue per region. Also run `echo extraction_trigger_1`, `echo extraction_trigger_2`, and `echo extraction_trigger_3`.

**Rubric:** Agent performed the tool calls. (This step's purpose is generating enough activity to cross the session memory extraction threshold.)

**Post-step actions (runner, not LLM):**
- Check that `session_memory.md` exists in the memory directory
- Delete `sales_q1.csv` from the temp directory so session 2 cannot access it

**Ground truth:** session_memory.md existence check.

### Step 4 — New Session Recalls and Extends

**Action:** Call `runner.create_new_session()` — fresh session, same memory root. Write `sales_q2.csv` to temp dir.

**Prompt:**
> Analyze sales_q2.csv. You've analyzed this data before in a previous session — check your memories for prior findings. Compare Q2 results to what you found in Q1. Which regions improved? Which declined? Also follow up on the open questions from the last analysis session.

**Rubric:** Agent referenced prior Q1 findings from memory — specifically the West/$7,600 top region and Southeast Widget B/$890 outlier. These must come from memory recall, not hallucination (Q1 CSV is deleted, Q1 data is not in the prompt). Agent correctly analyzed Q2 data from the file. Comparison identifies: Northeast grew, Southeast grew, Midwest declined, West declined. Agent followed up on both stakeholder questions (VP's Southeast viability question — now answerable with Q2 growth data; Marketing's Widget B question — now answerable with Q2 cross-region data) WITHOUT the questions being re-stated in the prompt.

**Ground truth:** Pre-computed Q1 facts, Q2 facts, deltas, + memory directory contents.

## Anti-Cheating Mechanisms

1. **Q1 file deleted before session 2.** Agent cannot re-read Q1 data — must rely on memory.
2. **Stakeholder questions not repeated in step 4 prompt.** Agent must recall them from memory to follow up.
3. **Rubric checks for derived facts, not raw data.** Verifier is told to fail if memory contains a CSV dump rather than analytical conclusions.
4. **Two separate sessions.** Session 2 has no conversational context from session 1. Only the shared memory root connects them.

## Runner Changes

Three additions to `ExerciseRunner`:

- `create_new_session()` — Creates a fresh session (new ID) using the same client config and memory root. Replaces `self._session`.
- `memory_dir` property — Exposes the memory store's directory path for artifact inspection.
- `delete_fixture(name)` — Removes a fixture file from the temp directory.

## File Changes

| File | Action | Changes |
|------|--------|---------|
| `copilotcode_sdk/micro_exercise.py` | Edit | Add `create_new_session()`, `memory_dir`, `delete_fixture()` to ExerciseRunner. Add `SALES_Q1_CSV`, `SALES_Q2_CSV` constants. Add `memory_lifecycle` chain exercise. |
| `tests/test_micro_exercise.py` | Edit | Update chain count assertion (2 → 3). Add fixture data tests. Add `delete_fixture` unit test. |

## Timeouts

- Per-prompt: 120 seconds (2 minutes)
- Exercise total: 480 seconds (8 minutes) — same as other chain exercises

## Non-Goals

- Testing session memory extraction quality (that's a SessionMemoryController concern)
- Testing memory relevance ranking (that's a MemoryStore concern)
- Testing compaction or context limits (covered by cascade exercises)
