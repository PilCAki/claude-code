---
name: simplify
description: Review recent changes for duplication, unnecessary complexity, and cleanup opportunities.
allowed-tools:
  - read
  - edit
  - search
  - execute
  - agent
when_to_use: Use after a code change when you want a cleanup pass grounded in existing patterns. Examples: "simplify this diff", "clean up the implementation", "look for unnecessary complexity".
argument-hint: "[extra focus, if any]"
---

# Simplify

## Goal
Make the changed code smaller, clearer, and more idiomatic without drifting beyond the task.

## Steps

### 1. Inspect the delta
Review the current diff or the files most recently touched.

Success criteria:
- You know exactly what changed.
- You can point to the highest-risk files for cleanup.

### 2. Look for reuse opportunities
Search the surrounding code for existing helpers, patterns, or abstractions that should be reused instead of duplicating logic.

Success criteria:
- You identified any existing utilities worth reusing.
- You avoided introducing new abstractions unless the duplication is real and repeated.

### 3. Remove avoidable complexity
Trim speculative flexibility, redundant state, unnecessary comments, or one-off helpers.

Success criteria:
- The code remains correct and easier to follow.
- You did not widen scope into unrelated refactors.

### 4. Re-run focused verification
Run the narrowest useful tests or commands that confirm the cleanup did not break behavior.

Success criteria:
- You verified the simplified code still works.

## Rules
- Stay inside the user's requested behavior.
- Prefer deleting complexity over relocating it.
- Do not create documentation unless the user asked for it.
