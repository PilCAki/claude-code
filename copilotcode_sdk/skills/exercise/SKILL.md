---
name: exercise
description: Systematically exercise every copilotcode_sdk subsystem and report structured results.
allowed-tools:
  - read
  - search
  - execute
when_to_use: Use when you want to verify that all SDK subsystems are functional by running real imports and real API calls against them.
argument-hint: "[optional: specific subsystem names to exercise, space-separated]"
context: fork
---

# Exercise

## Goal
Exercise every copilotcode_sdk subsystem using real imports and real API calls. You are a self-aware integration agent — you know you are testing the SDK, you try every capability, and you report back what worked and what didn't.

## Subsystem Checklist

Exercise each of these in order:

1. **prompt_compiler** — Build a PromptAssembler, add sections with different priorities, render to string.
2. **session_state** — Create a SessionState, transition through statuses (idle -> active -> completed).
3. **tasks** — Create a TaskStore, add tasks, update status, list open/all tasks.
4. **memory** — Create a MemoryStore, write a record, read it back, verify content matches.
5. **compaction** — Build a compaction prompt, format a transcript, parse a compaction response.
6. **extraction** — Build extraction prompts, test should_extract with different turn counts.
7. **session_memory** — Initialize a SessionMemoryController, check initial state is empty.
8. **events** — Create an EventBus, subscribe to an event type, emit an event, verify receipt.
9. **diff** — Generate a diff between two strings, verify it contains expected additions/removals.
10. **tokenizer** — Count tokens for a sample string, verify the count is a positive integer.
11. **retry** — Create a RetryPolicy, build a retry response, verify structure.
12. **suggestions** — Build prompt suggestions from minimal session state, verify list returned.
13. **skill_assets** — Parse skill frontmatter from a sample SKILL.md string, verify fields extracted.
14. **permissions** — Create a PermissionPolicy, check a safe and an unsafe tool call.
15. **model_cost** — Calculate cost for sample token usage, verify the result is a UsageCost.
16. **config** — Instantiate a CopilotCodeConfig with defaults, verify key fields.
17. **instructions** — Load workspace instructions from a temp directory with a CLAUDE.md file.

## Steps

### 1. Import and exercise each subsystem
For each subsystem in the checklist, import the relevant module from `copilotcode_sdk`, call its public API with reasonable inputs, and observe whether it works correctly.

Success criteria:
- The import succeeds.
- The API call returns a meaningful result without raising an exception.

### 2. Record results
For each subsystem, record:
- **name**: subsystem name from the checklist
- **status**: pass, fail, skip, or error
- **detail**: brief description of what you observed
- **error**: error message if status is fail or error, null otherwise

Success criteria:
- Every subsystem in the checklist has a result entry.
- No subsystem is silently skipped.

### 3. Return structured report
Return your complete results as JSON inside `<exercise-report>` tags:

```json
{
  "subsystems": [
    {
      "name": "subsystem_name",
      "status": "pass",
      "detail": "what you observed",
      "duration_seconds": 0.0,
      "error": null
    }
  ],
  "summary": "brief overall summary"
}
```

Success criteria:
- Valid JSON inside the tags.
- Every checklist item represented.

## Rules
- Use real imports and real API calls. Never fake or mock anything.
- If a subsystem requires filesystem access, use a temporary directory.
- If a subsystem genuinely cannot be exercised in the current environment, mark it "skip" with an explanation.
- Do not stop at the first failure. Exercise all subsystems regardless of individual results.
