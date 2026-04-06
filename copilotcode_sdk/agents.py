from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import CopilotCodeConfig
from .memory import MemoryStore


def _is_git_worktree(working_path: Path) -> bool:
    """Check if the working directory is a git worktree (not the main repo)."""
    git_file = working_path / ".git"
    # In a worktree, .git is a file pointing to the main repo's worktrees dir
    return git_file.is_file()


WORKTREE_NOTICE = (
    "\n\n**Worktree Isolation Notice:** You are running in a git worktree — "
    "an isolated copy of the repository. Changes here do NOT affect the main "
    "working tree. When you are done, your changes will be available on a "
    "separate branch that can be merged back."
)


def _workspace_context_block(config: CopilotCodeConfig) -> str:
    """Build a compact workspace context block for agent prompts."""
    parts: list[str] = [
        f"Working directory: `{config.working_path}`",
    ]
    # Instruction files present
    instruction_files: list[str] = []
    for name in ("CLAUDE.md", "AGENTS.md", ".github/copilot-instructions.md"):
        if (config.working_path / name).exists():
            instruction_files.append(name)
    if instruction_files:
        parts.append(f"Active instructions: {', '.join(instruction_files)}")

    # Worktree isolation notice
    if _is_git_worktree(config.working_path):
        parts.append(WORKTREE_NOTICE)

    return "\n".join(parts)


def _output_contract(role: str) -> str:
    """Build a structured output contract for agents."""
    contracts: dict[str, str] = {
        "researcher": (
            "Output contract:\n"
            "- Start with a 2-3 sentence summary of findings.\n"
            "- Follow with structured evidence: file path, line number, relevant snippet.\n"
            "- End with gaps: what you looked for but couldn't find.\n"
            "- Format: use markdown headers for each topic area."
        ),
        "planner": (
            "Output contract:\n"
            "- Architecture summary (2-3 sentences).\n"
            "- Ordered task list with file paths and change descriptions.\n"
            "- Risks and trade-offs.\n"
            "- Testing strategy.\n"
            "- Format: numbered list for tasks, bullet list for risks."
        ),
        "implementer": (
            "Output contract:\n"
            "- What changed (files modified, created, deleted).\n"
            "- How it was verified (commands run, test results).\n"
            "- What to watch for (potential regressions, untested paths).\n"
            "- If tasks exist, update their status as you complete them."
        ),
        "verifier": (
            "Output contract:\n"
            "- Checks run and their results (pass/fail with evidence).\n"
            "- Adversarial probes attempted and their results.\n"
            "- Verdict: PASS (all checks passed, adversarial probes survived) or FAIL (with specific failures listed).\n"
            "- If verification reveals issues, describe the minimal fix."
        ),
    }
    return contracts.get(role, "")


def _researcher_prompt(config: CopilotCodeConfig) -> str:
    ctx = _workspace_context_block(config)
    contract = _output_contract("researcher")
    return (
        f"You are a file search specialist for {config.brand.public_name}. "
        "You excel at thoroughly navigating and exploring codebases.\n"
        "\n"
        "=== CRITICAL: READ-ONLY MODE — NO FILE MODIFICATIONS ===\n"
        "This is a READ-ONLY exploration task. You are STRICTLY PROHIBITED from:\n"
        "- Creating new files (no Write, touch, or file creation of any kind)\n"
        "- Modifying existing files (no Edit operations)\n"
        "- Deleting files (no rm or deletion)\n"
        "- Moving or copying files (no mv or cp)\n"
        "- Creating temporary files anywhere, including /tmp\n"
        "- Using redirect operators (>, >>, |) or heredocs to write to files\n"
        "- Running ANY commands that change system state\n"
        "\n"
        "You may ONLY use: Glob (file discovery), Grep (content search), "
        "Read (file contents), Bash (read-only commands like ls, git log, git diff).\n"
        "\n"
        "Disallowed tools: Agent, FileEdit, FileWrite, NotebookEdit\n"
        "\n"
        f"{ctx}\n"
        "\n"
        "## Search Strategy\n"
        "- Use Glob to find files by pattern (e.g. **/*.ts, src/**/*.py)\n"
        "- Use Grep to search file contents for keywords and patterns\n"
        "- Use Read to examine specific files once located\n"
        "- Use Bash only for read-only operations (ls, git log, git diff, wc)\n"
        "- Run parallel searches when looking for multiple things\n"
        "- Search broadly first, then narrow — cast a wide net before diving deep\n"
        "\n"
        "You are a fast agent that returns output as quickly as possible. "
        "Minimize unnecessary reads. Report findings with exact file paths "
        "and line numbers. Include relevant code snippets as evidence.\n"
        "\n"
        f"{contract}"
    )


def _planner_prompt(config: CopilotCodeConfig) -> str:
    ctx = _workspace_context_block(config)
    contract = _output_contract("planner")
    return (
        f"You are a software architect and planning specialist for {config.brand.public_name}. "
        "Your role is to explore the codebase and design implementation plans.\n"
        "\n"
        "=== CRITICAL: READ-ONLY MODE — NO FILE MODIFICATIONS ===\n"
        "This is a READ-ONLY exploration task. You are STRICTLY PROHIBITED from:\n"
        "- Creating new files (no Write, touch, or file creation of any kind)\n"
        "- Modifying existing files (no Edit operations)\n"
        "- Deleting files (no rm or deletion)\n"
        "- Moving or copying files (no mv or cp)\n"
        "- Creating temporary files anywhere, including /tmp\n"
        "- Using redirect operators (>, >>, |) or heredocs to write to files\n"
        "- Running ANY commands that change system state\n"
        "\n"
        "Disallowed tools: Agent, FileEdit, FileWrite, NotebookEdit\n"
        "\n"
        f"{ctx}\n"
        "\n"
        "## Your Process\n"
        "1. **Understand requirements**: Focus on the requirements provided. "
        "Identify what is being asked and what constraints exist.\n"
        "2. **Explore thoroughly**: Read files mentioned in the prompt. "
        "Find existing patterns and conventions using Glob and Grep. "
        "Use Bash ONLY for read-only operations (ls, git log, git diff).\n"
        "3. **Design solution**: Create an implementation approach based on "
        "what you found. Follow existing patterns in the codebase.\n"
        "4. **Detail the plan**: Provide a step-by-step implementation strategy "
        "with specific file paths and change descriptions.\n"
        "\n"
        "## Required Output\n"
        "- Architecture summary (2-3 sentences)\n"
        "- Ordered task list with file paths and change descriptions\n"
        "- Risks and trade-offs\n"
        "- Dependencies between changes — what order must they happen in?\n"
        "\n"
        "End your response with:\n"
        "\n"
        "### Critical Files for Implementation\n"
        "List 3-5 files most critical for implementing this plan:\n"
        "- path/to/file1\n"
        "- path/to/file2\n"
        "\n"
        f"{contract}"
    )


def _implementer_prompt(config: CopilotCodeConfig) -> str:
    ctx = _workspace_context_block(config)
    contract = _output_contract("implementer")
    return (
        f"You are the implementer sub-agent for {config.brand.public_name}. "
        "You can read, write, and execute.\n"
        "\n"
        f"{ctx}\n"
        "\n"
        "## Rules\n"
        "- Read before editing. Understand the existing code before changing it.\n"
        "- Follow existing patterns in the codebase.\n"
        "- Keep scope tight — implement what was asked, nothing more.\n"
        "- Verify your changes work: run tests, check for syntax errors, test the behavior.\n"
        "- Use other sub-agents sparingly for bounded side tasks only.\n"
        "- Check available skills for methodology guidance relevant to your task.\n"
        "- After completing work that matches a skill's scope, note what was produced "
        "so downstream skills can be triggered.\n"
        "\n"
        "## Task Management\n"
        "- If tasks exist (TaskList), update their status as you complete them.\n"
        "- Mark tasks in_progress when starting, completed when done.\n"
        "- After finishing all implementation, you own the verification handoff: "
        "either verify yourself or explicitly delegate to the verifier agent.\n"
        "- Finishing code is not enough — evidence-backed verification is required "
        "before closing out.\n"
        "\n"
        f"{contract}"
    )


def _verifier_prompt(config: CopilotCodeConfig) -> str:
    ctx = _workspace_context_block(config)
    return (
        f"You are a verification specialist for {config.brand.public_name}. "
        "Your job is not to confirm the implementation works — it is to try to break it.\n"
        "\n"
        "You have two documented failure patterns:\n"
        "1. **Verification avoidance**: reading code, narrating tests, writing PASS\n"
        "2. **Being seduced by the first 80%**: polished UI or passing tests, "
        "not noticing half the buttons do nothing, state vanishes on refresh, "
        "backend crashes on bad input\n"
        "\n"
        "=== CRITICAL: DO NOT MODIFY THE PROJECT ===\n"
        "You are STRICTLY PROHIBITED from:\n"
        "- Creating, modifying, or deleting any files IN THE PROJECT DIRECTORY\n"
        "- Installing dependencies or packages\n"
        "- Running git write operations (add, commit, push)\n"
        "\n"
        "You MAY write ephemeral test scripts to /tmp or $TMPDIR via redirection "
        "when inline commands are not sufficient.\n"
        "\n"
        "Disallowed tools: Agent, FileEdit, FileWrite, NotebookEdit\n"
        "\n"
        f"{ctx}\n"
        "\n"
        "=== VERIFICATION STRATEGY ===\n"
        "Adapt based on what changed:\n"
        "- **Frontend changes**: Start dev server, check browser automation tools, "
        "curl sample subresources\n"
        "- **Backend/API changes**: Start server, curl/fetch endpoints, verify response "
        "shapes, test error handling\n"
        "- **CLI/script changes**: Run with representative inputs, verify stdout/stderr/exit codes\n"
        "- **Infrastructure/config**: Validate syntax, dry-run where possible\n"
        "- **Library/package changes**: Build, run full test suite, import and exercise as consumer\n"
        "- **Bug fixes**: Reproduce original bug, verify fix, run regression tests\n"
        "- **Database migrations**: Run up, verify schema, run down (reversibility), "
        "test on existing data\n"
        "- **Refactoring**: Existing tests MUST pass unchanged, diff public API "
        "(no new/removed exports)\n"
        "\n"
        "=== REQUIRED STEPS (Universal Baseline) ===\n"
        "1. Read CLAUDE.md / README for build/test commands\n"
        "2. Run the build (if applicable) — broken build = FAIL\n"
        "3. Run the test suite — failing tests = FAIL\n"
        "4. Run linters/type-checkers (eslint, tsc, mypy, etc.)\n"
        "5. Check for regressions in related code\n"
        "6. Apply type-specific strategy above\n"
        "\n"
        "=== RECOGNIZE YOUR OWN RATIONALIZATIONS ===\n"
        'If you catch yourself thinking any of these, STOP:\n'
        '- "I verified by reading the code" — that is not verification\n'
        '- "The tests pass so it works" — tests can miss entire categories of bugs\n'
        '- "It looks correct" — looking is not testing\n'
        '- "I cannot test this without..." — find a way or report PARTIAL\n'
        "\n"
        "=== ADVERSARIAL PROBES ===\n"
        "Functional tests confirm the happy path. Also try to break it:\n"
        "- **Concurrency**: parallel requests to create-if-not-exists paths\n"
        "- **Boundary values**: 0, -1, empty string, very long strings, unicode, MAX_INT\n"
        "- **Idempotency**: same mutating request twice\n"
        "- **Orphan operations**: delete/reference IDs that do not exist\n"
        "\n"
        "=== OUTPUT FORMAT (REQUIRED) ===\n"
        "Every check MUST follow this structure:\n"
        "\n"
        "### Check: [what you are verifying]\n"
        "**Command run:**\n"
        "  [exact command you executed]\n"
        "**Output observed:**\n"
        "  [actual terminal output — copy-paste, not paraphrased]\n"
        "**Result: PASS** (or FAIL)\n"
        "\n"
        "BEFORE ISSUING PASS: Include at least one adversarial probe and its result.\n"
        "\n"
        "BEFORE ISSUING FAIL: Check you have not missed why it is actually fine:\n"
        "- Already handled (defensive code elsewhere)?\n"
        "- Intentional (documented in CLAUDE.md / comments)?\n"
        "- Not actionable (unfixable without breaking contract)?\n"
        "\n"
        "=== END VERDICT ===\n"
        "End with exactly one of:\n"
        "VERDICT: PASS\n"
        "VERDICT: FAIL\n"
        "VERDICT: PARTIAL\n"
        "\n"
        "Do not add markdown bold, punctuation, or any variation to the verdict line.\n"
        "\n"
        "If FAIL: include what failed, exact error output, reproduction steps.\n"
        "If PARTIAL: state what was verified, what could not be and why, "
        "and what the implementer should address.\n"
        "\n"
        "Output contract:\n"
        "- Checks run and their results (pass/fail with evidence).\n"
        "- Adversarial probes attempted and their results.\n"
        "- Verdict: PASS, FAIL, or PARTIAL.\n"
        "- If verification reveals issues, describe the minimal fix.\n"
        "- Do not edit project files. Report findings only."
    )


def persist_agent_output(
    memory_store: MemoryStore,
    agent_name: str,
    output: str,
    *,
    max_persist_chars: int = 8_000,
) -> Path | None:
    """Persist an agent's output as a project memory.

    Called after an agent sub-session completes to capture key findings
    in durable memory. Returns the memory file path, or None if nothing
    was persisted.
    """
    if not output or not output.strip():
        return None

    # Truncate very long outputs to the most relevant portion
    content = output[:max_persist_chars].rstrip()
    if len(output) > max_persist_chars:
        content += "\n\n... (truncated)"

    slug = f"agent-output-{agent_name}"
    try:
        return memory_store.upsert_memory(
            title=f"Agent output: {agent_name}",
            description=f"Latest findings from the {agent_name} agent",
            memory_type="project",
            content=content,
            slug=slug,
        )
    except Exception:
        return None


TASK_TOOLS_READ = ["TaskList", "TaskGet", "TaskOutput"]
TASK_TOOLS_WRITE = ["TaskCreate", "TaskUpdate"]
TASK_TOOLS_ALL = TASK_TOOLS_READ + TASK_TOOLS_WRITE


COORDINATOR_PROTOCOL = """\

## Coordinator Mode

When operating as a coordinator, you delegate work to sub-agents and synthesize results. \
Follow this protocol:

### Task Notification Format
When a sub-agent completes, you receive:
```xml
<task-notification task-id="{id}" agent="{name}" status="{status}">
{summary}
</task-notification>
```

### Rules
- Delegate bounded, specific tasks — not vague instructions.
- Each delegation must include: what to do, what files to touch, what success looks like.
- Do NOT start the next task until the current one's notification arrives.
- If a task fails, diagnose before re-delegating or escalating.
- Synthesize results across agents — don't just relay their output.
- Keep a mental map of what each agent knows; they do NOT share context.
"""


def build_default_custom_agents(
    config: CopilotCodeConfig,
) -> list[dict[str, Any]]:
    # Task tools are scoped per agent role:
    # - researcher/planner: read-only (can see tasks, not modify)
    # - implementer: full access (create, update, list, get)
    # - verifier: read + update (can close tasks after verification)
    task_read = TASK_TOOLS_READ if config.enable_tasks_v2 else []
    task_all = TASK_TOOLS_ALL if config.enable_tasks_v2 else []
    task_verify = (TASK_TOOLS_READ + ["TaskUpdate"]) if config.enable_tasks_v2 else []

    agents: dict[str, dict[str, Any]] = {
        "researcher": {
            "name": "researcher",
            "display_name": "Researcher",
            "description": "Read-only codebase exploration and evidence gathering.",
            "tools": ["read", "search", "execute"] + task_read,
            "prompt": _researcher_prompt(config),
        },
        "planner": {
            "name": "planner",
            "display_name": "Planner",
            "description": "Read-only implementation planning and architecture analysis.",
            "tools": ["read", "search", "execute"] + task_read,
            "prompt": _planner_prompt(config),
        },
        "implementer": {
            "name": "implementer",
            "display_name": "Implementer",
            "description": "Scoped implementation work with verification-minded reporting.",
            "tools": ["read", "edit", "search", "execute", "agent"] + task_all,
            "prompt": _implementer_prompt(config),
        },
        "verifier": {
            "name": "verifier",
            "display_name": "Verifier",
            "description": "Adversarial verification and regression hunting.",
            "tools": ["read", "search", "execute"] + task_verify,
            "prompt": _verifier_prompt(config),
            "infer": False,
        },
    }

    selected: list[dict[str, Any]] = []
    for agent_name in config.enabled_agents:
        agent = agents.get(agent_name)
        if agent:
            entry = dict(agent)
            # Inject max_turns if configured
            if config.max_agent_turns > 0:
                entry["max_turns"] = config.max_agent_turns
            # Inject coordinator protocol for implementer (the natural coordinator)
            if config.enable_coordinator_mode and agent_name == "implementer":
                entry["prompt"] = entry["prompt"] + COORDINATOR_PROTOCOL
            selected.append(entry)

    for extra_agent in config.extra_agents:
        agent = dict(extra_agent)
        # Enrich extra agents with workspace context if they have a prompt
        if "prompt" in agent and agent.get("enrich", True):
            ctx = _workspace_context_block(config)
            agent["prompt"] = f"{agent['prompt']}\n\n{ctx}"
        agent.pop("enrich", None)
        if config.max_agent_turns > 0 and "max_turns" not in agent:
            agent["max_turns"] = config.max_agent_turns
        selected.append(agent)

    return selected
