from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from .brand import BrandSpec, DEFAULT_BRAND


def _verify_skill() -> str:
    return """---
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
"""


def _remember_skill(brand: BrandSpec) -> str:
    return f"""---
name: remember
description: Review and maintain durable project or user memory in the {brand.public_name} memory store.
allowed-tools:
  - read
  - edit
  - search
when_to_use: Use when the user wants {brand.public_name} to remember, forget, or clean up durable context. Examples: "remember this preference", "forget that note", "organize project memory".
argument-hint: "[memory request or cleanup goal]"
---

# Remember

## Goal
Keep durable memory clean, useful, and limited to information that will matter in future sessions.

## Steps

### 1. Read the memory index
Inspect `MEMORY.md` first so you understand what is already stored and can avoid duplicates.

Success criteria:
- You know whether an existing memory should be updated instead of creating a new file.

### 2. Classify the information
Decide whether the request belongs in one of these durable buckets:
- `user`: long-lived user preferences or collaboration patterns
- `feedback`: do or do not repeat specific behaviors
- `project`: stable repo conventions, commands, or architecture notes
- `reference`: durable supporting information worth resurfacing later

Success criteria:
- The memory type is explicit.
- Ephemeral task-specific notes are rejected instead of stored.

### 3. Apply the change
Create, update, or remove the relevant memory file, then update the index so it remains a concise pointer list.

Success criteria:
- The memory file and index agree.
- The index entry is one concise line.

### 4. Confirm the durable value
Briefly explain what was stored or removed and why it will matter in future work.

Success criteria:
- The user can tell what changed without rereading every file.

## Rules
- Do not store transient TODOs, one-off debugging facts, or information that can be re-derived cheaply from the repo.
- Prefer updating the best existing memory over creating near-duplicates.
- If the user asks to forget something, remove both the topic file and the index reference.
"""


def _simplify_skill() -> str:
    return """---
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
"""


def _debug_skill() -> str:
    return """---
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
"""


def _batch_skill() -> str:
    return """---
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
"""


def _skillify_skill() -> str:
    return """---
name: skillify
description: Convert a successful repeatable workflow into a reusable Copilot skill.
allowed-tools:
  - read
  - edit
  - search
  - execute
  - ask_user
when_to_use: Use near the end of a successful workflow when the same process is likely to repeat. Examples: "turn this into a skill", "capture this workflow", "make a reusable skill from this session".
argument-hint: "[optional description of the workflow]"
---

# Skillify

## Goal
Capture a repeatable workflow as a clear `SKILL.md` that another session can execute reliably.

## Steps

### 1. Analyze the workflow
Review what was done, what inputs were required, what artifacts were produced, and where the user corrected the process.

Success criteria:
- You can describe the workflow as ordered steps with explicit success criteria.

### 2. Interview for missing details
Use `ask_user` only for the details that are still ambiguous: naming, storage location, trigger phrases, required arguments, and hard constraints.

Success criteria:
- You asked only the questions needed to make the skill reusable.

### 3. Draft the skill
Write a `SKILL.md` with frontmatter, goal, inputs, steps, and success criteria for each major step.

Success criteria:
- The draft is self-contained.
- Someone new could run the process without access to the original conversation.

### 4. Confirm and save
Show the proposed skill, get confirmation if appropriate, then save it in the chosen skill directory.

Success criteria:
- The user knows where the skill lives and how to invoke it.

## Rules
- Keep simple workflows simple.
- Encode the user's corrections and preferences as explicit rules.
- Prefer clear triggers and concrete artifacts over vague motivational language.
"""


def get_skill_document(skill_name: str, brand: BrandSpec = DEFAULT_BRAND) -> str:
    documents = _render_documents(brand)
    return documents[skill_name]


def iter_skill_documents(
    brand: BrandSpec = DEFAULT_BRAND,
) -> list[tuple[str, str]]:
    return list(_render_documents(brand).items())


def _render_documents(brand: BrandSpec) -> dict[str, str]:
    return {
        "verify": _verify_skill(),
        "remember": _remember_skill(brand),
        "simplify": _simplify_skill(),
        "debug": _debug_skill(),
        "batch": _batch_skill(),
        "skillify": _skillify_skill(),
    }


def parse_skill_frontmatter(skill_md_path: Path) -> dict[str, str]:
    """Parse YAML-ish frontmatter from a SKILL.md file.

    Returns a dict of key-value pairs from the ``---`` delimited block.
    """
    text = skill_md_path.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return {"name": skill_md_path.parent.name}
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        colon = line.find(":")
        if colon < 1:
            continue
        key = line[:colon].strip()
        value = line[colon + 1:].strip()
        if value:
            fields[key] = value
    if "name" not in fields:
        fields["name"] = skill_md_path.parent.name
    return fields


def build_skill_catalog(
    skill_directories: Sequence[str],
    *,
    disabled_skills: Sequence[str] = (),
) -> tuple[str, dict[str, dict[str, str]]]:
    """Scan skill directories and build a formatted catalog with dependency info.

    Returns ``(catalog_text, skill_map)`` where *skill_map* maps skill name to
    parsed frontmatter dict.
    """
    disabled_set = set(disabled_skills)
    skills: list[dict[str, str]] = []

    for dir_path in skill_directories:
        root = Path(dir_path)
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            skill_md = child / "SKILL.md"
            if not skill_md.is_file():
                continue
            fm = parse_skill_frontmatter(skill_md)
            if fm["name"] in disabled_set:
                continue
            skills.append(fm)

    if not skills:
        return "", {}

    skill_map = {s["name"]: s for s in skills}

    # Build dependency chain via topological sort
    ordered = _topo_sort(skills)

    # Format table
    lines = [
        "## Available Skills",
        "",
        "The following skills are available in this workspace. Each skill has a SKILL.md",
        "with full methodology, expected outputs, and quality rubrics. Read the SKILL.md",
        "before starting work that matches a skill's scope.",
        "",
        "| Skill | Type | Description | Requires |",
        "|-------|------|-------------|----------|",
    ]
    for s in ordered:
        name = s["name"]
        stype = s.get("type", "-")
        desc = s.get("description", "-")
        if len(desc) > 80:
            desc = desc[:77] + "..."
        requires = s.get("requires", "none")
        lines.append(f"| {name} | {stype} | {desc} | {requires} |")

    # Build dependency chain text
    chains = _build_dependency_chains(ordered)
    if chains:
        lines.append("")
        lines.append("### Skill Dependencies")
        lines.append("")
        lines.append("Skills declare prerequisites via `requires`. Before starting a skill, check")
        lines.append("whether its prerequisites have been completed. If not, complete them first.")
        lines.append("")
        for chain in chains:
            lines.append(f"  {chain}")

    return "\n".join(lines), skill_map


def _topo_sort(skills: list[dict[str, str]]) -> list[dict[str, str]]:
    """Sort skills so that dependencies come before dependents."""
    type_to_skill: dict[str, str] = {}
    for s in skills:
        stype = s.get("type")
        if stype:
            type_to_skill[stype] = s["name"]

    name_to_skill = {s["name"]: s for s in skills}
    visited: set[str] = set()
    result: list[dict[str, str]] = []

    def visit(name: str) -> None:
        if name in visited or name not in name_to_skill:
            return
        visited.add(name)
        s = name_to_skill[name]
        req = s.get("requires")
        if req and req != "none":
            dep_name = type_to_skill.get(req)
            if dep_name:
                visit(dep_name)
        result.append(s)

    for s in skills:
        visit(s["name"])
    return result


def _build_dependency_chains(ordered: list[dict[str, str]]) -> list[str]:
    """Build human-readable dependency chain strings."""
    type_to_name: dict[str, str] = {}
    for s in ordered:
        stype = s.get("type")
        if stype:
            type_to_name[stype] = s["name"]

    chains: list[list[str]] = []
    in_chain: set[str] = set()

    for s in ordered:
        req = s.get("requires")
        if not req or req == "none":
            continue
        dep_name = type_to_name.get(req)
        if not dep_name:
            continue
        placed = False
        for chain in chains:
            if chain[-1] == dep_name:
                chain.append(s["name"])
                in_chain.add(s["name"])
                placed = True
                break
        if not placed:
            chains.append([dep_name, s["name"]])
            in_chain.add(dep_name)
            in_chain.add(s["name"])

    return [" → ".join(chain) for chain in chains if len(chain) > 1]
