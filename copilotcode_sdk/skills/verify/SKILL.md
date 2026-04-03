---
name: verify
description: Verify that a code change works by running the system and trying to break it.
allowed-tools:
  - read
  - search
  - execute
when_to_use: Use when the user wants evidence that a change works, especially after non-trivial implementation work. Examples: "verify this fix", "test the new flow", "try to break it".
argument-hint: "[what changed and what success should look like]"
context: fork
---

# Verify

## Goal
Prove whether the current implementation actually works. Do not stop at code review or a passing happy path.

## Inputs
- The original user task.
- The files or features that changed.
- Any explicit success criteria the user gave.

## Steps

### 1. Understand the target behavior
Read the relevant files, specs, and repo instructions before you run checks.

Success criteria:
- You can state what should happen on success.
- You know which commands or interactions exercise the change directly.

### 2. Run the standard checks
Run the build, tests, linters, or type checks that normally protect this project.

Success criteria:
- You captured the exact commands.
- You know which checks passed and which failed.

### 3. Exercise the change directly
Run the app, script, endpoint, or workflow that changed. Prefer real execution over static reasoning.

Success criteria:
- You verified the changed behavior with a direct command or interaction.
- You captured the output that supports the result.

### 4. Try at least one adversarial probe
Probe the edge that the implementer was most likely to miss: boundary inputs, malformed input, repeated actions, missing data, concurrency, or persistence.

Success criteria:
- You ran at least one realistic break attempt.
- You recorded whether the system handled it correctly.

### 5. Report a verdict
End with a concise verdict and the evidence behind it.

Success criteria:
- The report clearly says PASS, FAIL, or PARTIAL.
- The report distinguishes confirmed behavior from anything you could not verify.

## Rules
- Reading code is useful context, not verification.
- Never claim a check passed if you did not run it.
- Prefer exact commands and observed output over paraphrase.
- If the environment blocks verification, say what you could verify and what was blocked.
