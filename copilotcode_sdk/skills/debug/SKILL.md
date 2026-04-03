---
name: debug
description: Turn a failing behavior into a reproducible diagnosis with concrete next steps.
allowed-tools:
  - read
  - edit
  - search
  - execute
when_to_use: Use when the user reports a bug, failing command, crash, or confusing behavior. Examples: "debug this", "why is this failing", "help me reproduce the issue".
argument-hint: "[symptoms, failing command, or suspected area]"
---

# Debug

## Goal
Get from symptoms to a credible root cause or the narrowest remaining unknown.

## Steps

### 1. Reproduce the issue
Run the failing command, test, or workflow exactly as described if possible.

Success criteria:
- You have an observed failure or a clear reason the issue could not be reproduced.

### 2. Narrow the fault line
Use code search, logs, stack traces, and targeted experiments to isolate where the behavior diverges from expectation.

Success criteria:
- You can name the failing subsystem or the missing precondition.

### 3. Validate the hypothesis
Run one or more focused checks that would behave differently if your diagnosis were wrong.

Success criteria:
- The evidence supports your explanation instead of merely sounding plausible.

### 4. Recommend or apply the next step
If the user asked for a fix, make the narrowest change that addresses the root cause. Otherwise, present the likely cause and the next best action.

Success criteria:
- The next step is specific and evidence-backed.

## Rules
- Prefer exact commands and observed output over speculation.
- If multiple hypotheses remain, rank them and say what evidence would separate them.
- If you change code, re-run the reproduction or the failing test.
