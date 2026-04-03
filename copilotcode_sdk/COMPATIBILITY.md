# CopilotCode Compatibility Matrix

CopilotCode targets behavioral parity with the portable parts of Claude Code, not protocol-level equivalence.

| Claude Code area | Status | CopilotCode approach |
| --- | --- | --- |
| Modular system prompt | Ported | Python prompt compiler emits one Copilot append-mode system message. |
| Durable memory (`MEMORY.md` plus topic files) | Ported | Python `MemoryStore` manages a file index plus topic files under `~/.copilotcode/projects/.../memory/`. |
| Query-time memory recall | Shimmed | Deterministic header-and-keyword scoring replaces model-assisted memory selection. |
| Built-in read-only exploration agent | Ported | `researcher` custom agent scoped to read, search, and execute tools. |
| Built-in planning agent | Ported | `planner` custom agent scoped to read, search, and execute tools. |
| Main implementation agent | Ported | `implementer` custom agent scoped to read, edit, search, execute, and agent tools. |
| Verification agent | Ported | `verifier` custom agent with `infer: false` and adversarial verification guidance. |
| Bundled slash workflows | Ported | Copilot `SKILL.md` assets for `verify`, `remember`, `simplify`, `debug`, `batch`, and `skillify`. |
| Session persistence and workspace state | Ported | Uses Copilot infinite sessions plus CopilotCode durable memory files. |
| Hook-based guardrails | Ported | Session hooks add memory context, path policy, shell timeout defaults, and result normalization. |
| Prompt section overrides | Shimmed | CopilotCode compiles sections itself rather than relying on Claude Code internals. |
| Dream and Kairos background automation | Not portable | Explicitly out of scope for v1. No hidden autonomous background agents. |
| Buddy UI, bridge transports, undercover mode, internal telemetry | Not portable | Excluded from the Python SDK port. |
