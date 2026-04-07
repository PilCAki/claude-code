# Micro & Chain Exercise Design Spec

## Problem

The existing exercise tiers (subsystem, orchestration, advanced, cascade) test data structures and prompt construction — they pass trivially and don't catch real bugs. Real-world multi-step workflows consistently fail due to: tools not being called, hooks not firing, code not wired, data hallucinated, timeouts, wrong working directories. We need exercises that run real LLM sessions end-to-end and verify results with an LLM verifier.

## Solution

Two layers of LLM-based exercises:

1. **Micro-exercises** — one session, one prompt, one behavior, ~10-15 seconds each
2. **Chain exercises** — multi-step file pipelines (read -> transform -> report), ~2-4 minutes each

Both use a real `CopilotCodeClient` session and an LLM verifier (Haiku) for pass/fail judgment.

## Architecture

### ExerciseRunner

The test harness each exercise receives. Responsibilities:

- **Temp directory**: auto-created, auto-cleaned, used as session working directory
- **Real session**: `CopilotCodeClient` with exercise config (`build_exercise_config()`), pointed at temp dir
- **Fixtures**: `write_fixture(name, content)` creates files in temp dir; `read_fixture(name)` reads them back
- **Prompt dispatch**: `prompt(text, timeout=120)` sends a message to the live session, returns full response including tool calls
- **Ground truth**: after each prompt, snapshots `_raw_hooks` state (tool call count, files read, shell commands, context size)
- **Verification**: `verify(result, rubric, ground_truth=None)` calls the Haiku verifier

### Verifier

A single Haiku LLM call per verification. Receives:

```
You are verifying whether an AI agent completed a task correctly.

## Task Prompt
{the prompt sent to the agent}

## Agent Response
{full text response}

## Tool Calls Made
{list of tool names, arguments, and results}

## Ground Truth
{dict from _raw_hooks + fixture content}

## Rubric
{human-written rubric string}

Judge whether the agent's response satisfies the rubric.
Return JSON: {"passed": true/false, "reasoning": "one paragraph explaining why"}
```

The verifier sees tool calls and their results, not just text — so it can distinguish "actually read the file" from "claimed to read the file."

If the verifier call itself fails (timeout, parse error), the exercise result is `"error"` not `"fail"`.

### Status Distinction

- `"pass"` — verifier confirmed rubric satisfied
- `"fail"` — verifier said agent didn't satisfy rubric (real bug to investigate)
- `"error"` — infrastructure problem (timeout, verifier parse error, session creation failed)

## Micro-Exercises (6)

### 1. read_file
- **Setup**: Write a CSV fixture with known data (names, ages)
- **Prompt**: "Read data.txt and tell me what's in it."
- **Rubric**: "A Read/file-read tool call was made targeting data.txt. Response references the actual data (Alice, Bob, their ages) — not hallucinated content."

### 2. write_file
- **Setup**: None
- **Prompt**: "Create a file called output.txt containing exactly three lines: 'alpha', 'beta', 'gamma' — one per line, no extra whitespace."
- **Rubric**: "A Write/file-write tool call was made. output.txt exists in the working directory. File content is exactly three lines: alpha, beta, gamma."
- **Ground truth**: `read_fixture("output.txt")` after prompt

### 3. shell_command
- **Setup**: Write two fixture files (a.txt, b.txt) so the directory isn't empty
- **Prompt**: "Run ls in the current directory and tell me what files are here."
- **Rubric**: "A Bash/shell tool call was made running ls or equivalent. Response mentions both a.txt and b.txt."

### 4. multi_tool
- **Setup**: Write two fixtures — prices.csv (items + prices) and inventory.csv (items + quantities)
- **Prompt**: "Read both prices.csv and inventory.csv, then tell me which item has the highest total value (price * quantity)."
- **Rubric**: "Two Read/file-read tool calls were made (one per file). Response identifies the correct item based on actual multiplication of real values from the files — not hallucinated."
- **Ground truth**: pre-computed correct answer

### 5. error_recovery
- **Setup**: Write only good.txt (no bad.txt)
- **Prompt**: "Try to read bad.txt, then read good.txt. Tell me what happened with each."
- **Rubric**: "Agent attempted to read bad.txt and encountered an error/failure. Agent then successfully read good.txt using a Read tool call. Response acknowledges the first file failed and reports content of the second file accurately."

### 6. follow_instructions
- **Setup**: None
- **Prompt**: "Create a file called report.csv with a header row 'name,score,grade' followed by exactly 5 data rows. Use realistic student names and scores between 0-100. Assign grades: A for 90+, B for 80-89, C for 70-79, D for 60-69, F below 60."
- **Rubric**: "A Write tool call was made. report.csv exists. File has exactly 6 lines (1 header + 5 data). Header is 'name,score,grade'. Each data row has 3 comma-separated fields. Scores are between 0-100. Grades match the scoring rubric (A/B/C/D/F thresholds)."
- **Ground truth**: `read_fixture("report.csv")` after prompt

## Chain Exercises (2)

### 1. read_transform_report

**Step 1 — Read**:
- Setup: Write `sales.csv` with 10 rows of (product, units_sold, price_per_unit)
- Prompt: "Read sales.csv and summarize the columns and row count."
- Rubric: "Read tool was called. Response correctly identifies the three columns and states there are 10 data rows."

**Step 2 — Transform**:
- Prompt: "Calculate the total revenue (units_sold * price_per_unit) for each product. Write the results to summary.txt with one line per product in the format 'product: $revenue'."
- Rubric: "Write tool was called. summary.txt exists. Each line matches 'product: $N' format. Revenue values are correct based on the actual CSV data."
- Ground truth: pre-computed revenues + `read_fixture("summary.txt")`

**Step 3 — Report**:
- Prompt: "Read summary.txt and write a one-paragraph report identifying the top-selling product by revenue."
- Rubric: "Read tool was called targeting summary.txt. Response identifies the correct top product matching the actual data in summary.txt — not hallucinated."
- Ground truth: `read_fixture("summary.txt")` + correct top product

### 2. multi_file_synthesis

**Step 1 — Read**:
- Setup: Write `employees.csv` (name, department, salary) and `departments.csv` (department, budget, location)
- Prompt: "Read both employees.csv and departments.csv. Tell me how many employees and departments there are."
- Rubric: "Two Read tool calls made. Response states correct employee count and department count."

**Step 2 — Merge**:
- Prompt: "Create merged.csv that joins employees with their department info. Columns: name, department, salary, budget, location. Write one row per employee."
- Rubric: "Write tool was called. merged.csv exists with correct header. Row count matches employee count. Each employee's department info (budget, location) matches departments.csv."
- Ground truth: pre-computed merged data + `read_fixture("merged.csv")`

**Step 3 — Analyze**:
- Prompt: "Read merged.csv and tell me which department has the highest average salary. Include the average salary amount."
- Rubric: "Read tool was called targeting merged.csv. Response identifies the correct department with highest average salary and the amount is within $1 of the actual computed average."
- Ground truth: correct department + average

## File Structure

**New file: `copilotcode_sdk/micro_exercise.py`**

Contains:
- `VerifyResult` dataclass (passed: bool, reasoning: str)
- `ExerciseFailure` exception
- `ExerciseRunner` class
- `run_micro_exercises()` async function
- `run_chain_exercises()` async function
- All 8 exercise definitions as decorated async functions
- `MICRO_EXERCISES` and `CHAIN_EXERCISES` registries

**Modified files:**
- `copilotcode_sdk/exercise.py` — add `"micro"` and `"chain"` to ExerciseMode, wire `run_exercise()` to call `micro_exercise.py`
- `copilotcode_sdk/cli.py` — add `"micro"` and `"chain"` to mode choices
- `tests/test_micro_exercise.py` — unit tests for runner, verifier prompt, exercise registration, report integration

## Timeouts

- Per-prompt timeout: 120 seconds (2 minutes)
- Per micro-exercise timeout: 300 seconds (5 minutes)
- Per chain exercise timeout: 480 seconds (8 minutes)
- Total budget: under 10 minutes sequential

## Integration

Results use existing `SubsystemResult` and `ExerciseReport`. New modes:
- `copilotcode exercise --mode micro` — runs 6 micro-exercises
- `copilotcode exercise --mode chain` — runs 2 chain exercises
- `copilotcode exercise --mode full` — runs all tiers (subsystem, orchestration, advanced, cascade, micro, chain)

## Non-Goals

- Parallel execution (rate limits)
- Deterministic expected outputs (LLM responses vary)
- Testing reactive hooks (covered by advanced/cascade tiers)
- Mocking anything — these are real end-to-end exercises
