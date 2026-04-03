---
name: batch
description: Break a large mechanical change into parallelizable units with a shared verification recipe.
allowed-tools:
  - read
  - edit
  - search
  - execute
  - agent
when_to_use: Use for sweeping changes that can be split into independent work units. Examples: "migrate all call sites", "bulk rename this API", "apply this refactor across the repo".
argument-hint: "<large-scale change request>"
context: fork
---

# Batch

## Goal
Turn a large change into a coordinated multi-agent plan that can be executed safely and checked consistently.

## Steps

### 1. Research the scope
Find every directory, pattern, or call site touched by the requested change.

Success criteria:
- You understand the breadth of the change.
- You know which files or modules can move independently.

### 2. Decompose the work
Split the change into independent units with clear ownership and minimal overlap.

Success criteria:
- Each unit has a title, scope, and success condition.
- Parallel workers will not fight over the same files.

### 3. Define one verification recipe
Specify the exact tests, commands, or interactions every worker should use to prove their slice works.

Success criteria:
- Every unit has a concrete verification path.
- The recipe is specific enough to run without guessing.

### 4. Launch and track workers
Delegate bounded units to sub-agents, then keep an explicit checklist of status and blockers.

Success criteria:
- Work progresses in parallel without losing coordination.
- The final report clearly states which units completed and which did not.

## Rules
- Do not parallelize tightly coupled edits that require constant merge resolution.
- Keep the coordination plan visible to the main session.
- Require each worker to report verification, not just code changes.
