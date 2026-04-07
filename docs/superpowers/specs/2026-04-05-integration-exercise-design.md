# Integration Exercise System — Design Spec

**Date:** 2026-04-05
**Status:** Approved

## Problem

copilotcode_sdk has 17+ subsystems (prompt compiler, tasks, memory, compaction, events, diff, tokenizer, etc.) and 742+ unit tests, but no way to verify that everything works together inside a real LLM session. Unit tests exercise individual functions in isolation; the exercise system verifies that a real agent can import, call, and observe every subsystem's public API.

## Design

### Architecture: Two Layers

**Layer 1 — CLI Command (`copilotcode exercise`)**
The parent orchestrator. Creates a real Copilot SDK session, sends the exercise prompt, waits for the agent to complete, parses the structured JSON report, and outputs results.

**Layer 2 — Exercise Skill (`skills/exercise/SKILL.md`)**
The child-reachable skill that defines the subsystem checklist. When the agent runs inside a session, it follows this skill's instructions to systematically exercise each subsystem.

### Key Principle: No Mocks, No Fakes

The agent uses real imports, real API calls, and real responses. If a subsystem requires filesystem access (memory, instructions), the agent uses a temporary directory. If something genuinely cannot be exercised in the current environment, it's marked "skip" with an explanation — never silently omitted.

### Subsystem Checklist (17 subsystems)

1. prompt_compiler — Build/render PromptAssembler
2. session_state — Transition SessionState through statuses
3. tasks — TaskStore CRUD operations
4. memory — MemoryStore write/read round-trip
5. compaction — Build prompt, format transcript, parse response
6. extraction — Build prompts, test should_extract logic
7. session_memory — Initialize SessionMemoryController
8. events — EventBus subscribe/emit/verify
9. diff — Generate diff between two strings
10. tokenizer — Count tokens for sample string
11. retry — Build retry response from RetryPolicy
12. suggestions — Build prompt suggestions
13. skill_assets — Parse skill frontmatter
14. permissions — Check PermissionPolicy decisions
15. model_cost — Calculate cost for sample usage
16. config — Instantiate CopilotCodeConfig, verify defaults
17. instructions — Load workspace instructions bundle

### Report Format

The agent returns structured JSON inside `<exercise-report>` tags:

```json
{
  "subsystems": [
    {
      "name": "subsystem_name",
      "status": "pass|fail|skip|error",
      "detail": "what was observed",
      "duration_seconds": 0.0,
      "error": null
    }
  ],
  "summary": "brief overall summary"
}
```

The CLI parses this into an `ExerciseReport` dataclass with `.to_dict()` and `.to_text()` methods, following the same pattern as `SmokeTestReport`.

### CLI Interface

```
copilotcode exercise [--json] [--timeout 600] [--save-report path.json] [--subsystems name1 name2 ...]
```

- `--json`: Emit JSON output instead of human-readable text
- `--timeout`: Max seconds to wait (default 600)
- `--save-report`: Write JSON report to file
- `--subsystems`: Filter to specific subsystems (default: all 17)

## Files

| File | Purpose |
|------|---------|
| `copilotcode_sdk/exercise.py` | Report dataclasses, prompt builder, parser, orchestrator |
| `copilotcode_sdk/skills/exercise/SKILL.md` | Child-reachable exercise skill |
| `copilotcode_sdk/cli.py` | Exercise subcommand |
| `copilotcode_sdk/__init__.py` | Exports |
| `tests/test_exercise.py` | Unit tests |
