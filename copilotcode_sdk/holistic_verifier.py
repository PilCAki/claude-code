"""Holistic verification — cross-skill quality gate after all skills complete.

Spawns a read-only sub-agent that checks the combined pipeline outputs
for cross-skill consistency, completeness, and quality.

The system prompt is DOMAIN-AGNOSTIC. All domain-specific checks are derived
by the verifier from the skill definitions it receives as input.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from .subagent import SubagentSpec
from .verifier import (
    VerificationResult,
    _has_command_blocks,
    _log_verification_attempt,
    compare_output_hashes,
    extract_failed_checks,
    parse_verdict,
    snapshot_output_hashes,
)

logger = logging.getLogger("copilotcode.holistic_verifier")

HOLISTIC_VERIFIER_MAX_TURNS = 30
HOLISTIC_VERIFIER_TIMEOUT = 3600.0

HOLISTIC_VERIFIER_SYSTEM_PROMPT = """\
You are a holistic pipeline verifier. Your job is to verify cross-skill \
consistency and overall pipeline quality after ALL skills have completed \
their individual verification.

You will receive:
1. All skill definitions (SKILL.md content for every skill in the pipeline)
2. The output directory for each skill
3. Carry-forward artifacts (if available)

=== FIRST: DERIVE YOUR CHECK PLAN ===
Before running any commands, read all skill definitions carefully and derive \
specific checks for each category below. Your checks MUST come from the skill \
definitions — do NOT bring domain knowledge of your own. The skill definitions \
tell you what each skill should produce, what quality rubrics apply, what \
cross-skill dependencies exist, and what carry-forward artifacts are expected.

Write out your check plan explicitly before executing anything.

=== CHECK CATEGORIES ===

1. CROSS-SKILL CONSISTENCY
Where one skill's output is consumed by a downstream skill, verify the values \
match. Specifically:
- If a reporting/summary skill references values from an analysis skill, \
spot-check at least 3 numbers between them.
- If column names or schema are mapped in one skill and used in another, \
verify alignment.
- If an analysis skill reads from an intake skill's exports, verify the \
row counts and column names are consistent.

2. PIPELINE COMPLETENESS
- Every skill's expected output directory (from its frontmatter `outputs:` \
field) exists with non-trivial content.
- All self_score.json files are present and well-formed.
- No skill scored below its stated minimum threshold (read thresholds from \
each skill definition's rubric section).

3. CARRY-FORWARD INTEGRITY
If the workspace contains carry-forward artifacts (prior run outputs, memory \
files, learned mappings, accumulated learnings, or any other artifacts \
described by skill definitions as cross-session inputs):
- Check whether the skills that reference them actually consumed and used them.
- If a skill definition describes using prior artifacts, verify the usage \
produced real output (not placeholder, not ignored, not all N/A).
- This covers ANY form of cross-session continuity — not just metrics.

4. AGGREGATE QUALITY
- Collect all self_score.json files. If multiple skills score low on similar \
dimensions, flag that as a systemic issue.
- Pick 3 data points and trace them end-to-end across the pipeline (from \
earliest skill output through to final skill output).

5. ANTI-GAMING
- If content appears padded (repeated filler, HTML comments with bulk text, \
lorem ipsum, placeholder paragraphs), that is a FAIL.
- If a metric passes nominally but the underlying content is hollow, FAIL it.
- If self_score.json claims high scores but actual artifacts don't support \
them, FAIL with specific evidence of inflation.

=== RULES ===
- You MUST run commands for every check. Reading code/files and reasoning \
about correctness is NOT verification. Execute queries, count rows, \
check schemas, compare values.
- A check without a command run is a SKIP, not a PASS.
- Do NOT create, modify, or fix any output files.
- Do NOT write to the workspace or output directories.
- Do NOT attempt to redo any skill's work.
- You may write ephemeral test scripts to a temp directory only.

=== RECOGNIZE YOUR OWN RATIONALIZATIONS ===
- "The code looks correct based on my reading" — reading is not verification.
- "This is probably fine" — probably is not verified. Run it.
- "This would take too long" — not your call.
If you catch yourself writing an explanation instead of a command, stop. \
Run the command.

=== OUTPUT FORMAT ===
Every check MUST follow this structure:

### Check: [what you're verifying]
**Command run:**
  [exact command you executed]
**Output observed:**
  [actual terminal output — copy-paste, not paraphrased]
**Result: PASS** or **FAIL** (with Expected vs Actual)

End with exactly one of:
VERDICT: PASS
VERDICT: FAIL
VERDICT: PARTIAL

Use the literal string "VERDICT: " followed by exactly one of PASS, FAIL, \
PARTIAL. No markdown bold, no punctuation, no variation. PARTIAL is for \
environmental limitations only — not for "I'm unsure."
"""


def build_holistic_verifier_prompt(
    skill_definitions: dict[str, str],
    output_dirs: dict[str, str],
    workspace: str,
    carry_forward_artifacts: list[str] | None = None,
) -> str:
    """Build the user prompt for the holistic verification sub-agent.

    Parameters
    ----------
    skill_definitions : dict
        Mapping of skill_name -> full SKILL.md content.
    output_dirs : dict
        Mapping of skill_name -> absolute path to that skill's output directory.
    workspace : str
        Absolute path to the workspace root.
    carry_forward_artifacts : list, optional
        Paths to carry-forward files found in the workspace (prior metrics,
        column mappings, memory files, etc.).
    """
    parts = [
        "=== IMPORTANT: HOLISTIC VERIFICATION ONLY ===",
        "You are verifying the COMBINED outputs of all skills below.",
        "Do NOT implement any skill. Do NOT redo any work.",
        "Only check cross-skill consistency, completeness, and quality.",
        "",
        f"=== WORKSPACE ===",
        workspace,
        "",
    ]

    # Skill definitions
    for name, content in skill_definitions.items():
        output_dir = output_dirs.get(name, "(unknown)")
        parts.extend([
            f"=== SKILL DEFINITION: {name} ===",
            f"Output directory: {output_dir}",
            "",
            content,
            "",
        ])

    # Carry-forward artifacts
    if carry_forward_artifacts:
        parts.extend([
            "=== CARRY-FORWARD ARTIFACTS FOUND ===",
            "These files were present in the workspace before/during the run.",
            "Check whether skills that reference carry-forward data actually used them.",
            "",
        ])
        for path in carry_forward_artifacts:
            parts.append(f"- {path}")
        parts.append("")
    else:
        parts.extend([
            "=== CARRY-FORWARD ARTIFACTS ===",
            "None found — skip carry-forward integrity checks.",
            "",
        ])

    return "\n".join(parts)


def _find_carry_forward_artifacts(workspace: Path) -> list[str]:
    """Discover carry-forward artifacts in the workspace.

    Looks for common carry-forward file patterns:
    - prior_run_metrics.json
    - column_mapping.json
    - memory.md / MEMORY.md
    - Any file with 'prior' or 'carry_forward' in its name
    """
    artifacts: list[str] = []
    outputs_dir = workspace / "outputs"

    # Known carry-forward file names
    known_names = [
        "prior_run_metrics.json",
        "column_mapping.json",
    ]
    for name in known_names:
        path = outputs_dir / name
        if path.is_file():
            artifacts.append(str(path))

    # Memory files at workspace root
    for name in ("memory.md", "MEMORY.md"):
        path = workspace / name
        if path.is_file():
            artifacts.append(str(path))

    # Files with 'prior' or 'carry_forward' in name
    if outputs_dir.exists():
        for f in outputs_dir.rglob("*"):
            if f.is_file() and str(f) not in artifacts:
                lower = f.name.lower()
                if "prior" in lower or "carry_forward" in lower or "carryforward" in lower:
                    artifacts.append(str(f))

    return sorted(set(artifacts))


async def run_holistic_verification(
    skill_definitions: dict[str, str],
    output_dirs: dict[str, str],
    workspace: str,
    fork_child: Any,
    carry_forward_artifacts: list[str] | None = None,
    on_event: Any | None = None,
) -> VerificationResult:
    """Spawn a holistic verifier sub-agent and return the result.

    Unlike the skill-level verifier (which checks one skill), this checks
    the combined pipeline outputs for cross-skill consistency.

    Parameters
    ----------
    skill_definitions : dict
        Mapping of skill_name -> full SKILL.md content.
    output_dirs : dict
        Mapping of skill_name -> absolute path to output directory.
    workspace : str
        Absolute path to the workspace root.
    fork_child : callable
        Async callable to fork a child session (from CopilotCodeSession).
    carry_forward_artifacts : list, optional
        Paths to carry-forward files. If None, auto-discovered from workspace.
    """
    workspace_path = Path(workspace)

    if carry_forward_artifacts is None:
        carry_forward_artifacts = _find_carry_forward_artifacts(workspace_path)

    logger.info(
        "Starting holistic verification for %d skills...",
        len(skill_definitions),
    )

    # Snapshot all output dirs before verification
    all_output_hashes: dict[str, dict[str, str]] = {}
    for name, output_dir in output_dirs.items():
        od = Path(output_dir)
        if od.exists():
            all_output_hashes[name] = snapshot_output_hashes(od)

    # Build the verifier spec and prompt
    spec = SubagentSpec(
        role="holistic-verifier",
        system_prompt_suffix=HOLISTIC_VERIFIER_SYSTEM_PROMPT,
        tools=(),
        max_turns=HOLISTIC_VERIFIER_MAX_TURNS,
        timeout_seconds=HOLISTIC_VERIFIER_TIMEOUT,
    )

    user_prompt = build_holistic_verifier_prompt(
        skill_definitions=skill_definitions,
        output_dirs=output_dirs,
        workspace=workspace,
        carry_forward_artifacts=carry_forward_artifacts,
    )

    # Spawn and run the verifier, tracking token usage
    holistic_tokens: dict[str, int] = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    holistic_cost: list[float] = [0.0]

    def _track_holistic_event(evt: Any) -> None:
        try:
            if isinstance(evt, dict):
                etype = evt.get("type", "")
            else:
                etype = getattr(evt, "type", "")
                if hasattr(etype, "value"):
                    etype = etype.value
            if etype == "cost.accumulated":
                _g = (lambda k, d=0: evt.get(k, d)) if isinstance(evt, dict) else (lambda k, d=0: getattr(evt, k, d))
                holistic_tokens["input"] += int(_g("input_tokens", 0) or 0)
                holistic_tokens["output"] += int(_g("output_tokens", 0) or 0)
                holistic_tokens["cache_read"] += int(_g("cache_read_tokens", 0) or 0)
                holistic_tokens["cache_write"] += int(_g("cache_write_tokens", 0) or 0)
                holistic_cost[0] += float(_g("turn_cost", 0) or 0)
        except Exception:
            pass
        if on_event is not None:
            try:
                on_event(evt)
            except Exception:
                pass

    try:
        child = await fork_child(spec, on_event=_track_holistic_event)
        try:
            await child.send_and_wait(user_prompt, timeout=HOLISTIC_VERIFIER_TIMEOUT)
            raw_output = await child.get_last_response_text()
            if not raw_output:
                raw_output = "(holistic verifier produced no text response)"
        finally:
            await child.destroy()
    except Exception as exc:
        result = VerificationResult(
            verdict="MALFUNCTION",
            raw_output=f"Holistic verifier crashed: {exc}",
            token_usage=dict(holistic_tokens),
            cost=holistic_cost[0],
        )
        _log_verification_attempt(
            workspace_path, "holistic", 1, result,
        )
        return result

    # Check for output tampering across all output dirs
    # Added files (verifier scripts) are cleaned up silently.
    # Only modified existing artifacts trigger MALFUNCTION.
    for name, output_dir in output_dirs.items():
        od = Path(output_dir)
        if name in all_output_hashes:
            changed = compare_output_hashes(od, all_output_hashes[name])
            if changed:
                added = [p for p in changed if p not in all_output_hashes[name]]
                modified = [p for p in changed if p in all_output_hashes[name]]

                for rel_path in added:
                    victim = od / rel_path
                    if victim.exists():
                        victim.unlink()
                        logger.info(
                            "VERIFY cleanup: removed verifier script %s/%s",
                            name, rel_path,
                        )

                if modified:
                    logger.warning(
                        "TAMPER: holistic verifier modified existing files "
                        "in %s: %s",
                        name, modified,
                    )
                    result = VerificationResult(
                        verdict="MALFUNCTION",
                        raw_output=(
                            f"Holistic verifier modified existing output "
                            f"artifacts in '{name}': {', '.join(modified)}. "
                            "Verification invalidated."
                        ),
                        token_usage=dict(holistic_tokens),
                        cost=holistic_cost[0],
                    )
                    _log_verification_attempt(
                        workspace_path, "holistic", 1, result,
                    )
                    return result

    # Parse verdict
    verdict = parse_verdict(raw_output)
    if verdict is None:
        result = VerificationResult(
            verdict="MALFUNCTION",
            raw_output=(
                "Holistic verification failed: verifier did not produce a verdict "
                "in the required format.\n\n"
                "--- RAW VERIFIER OUTPUT ---\n" + raw_output
            ),
            token_usage=dict(holistic_tokens),
            cost=holistic_cost[0],
        )
        _log_verification_attempt(workspace_path, "holistic", 1, result)
        return result

    # Check that commands were actually run
    if not _has_command_blocks(raw_output):
        result = VerificationResult(
            verdict="MALFUNCTION",
            raw_output=(
                "Holistic verification failed: verifier did not run any commands.\n\n"
                "--- RAW VERIFIER OUTPUT ---\n" + raw_output
            ),
            token_usage=dict(holistic_tokens),
            cost=holistic_cost[0],
        )
        _log_verification_attempt(workspace_path, "holistic", 1, result)
        return result

    # Extract failed checks
    failed_checks = extract_failed_checks(raw_output)

    # Override PASS if individual checks failed
    if verdict == "PASS" and failed_checks:
        logger.warning(
            "Holistic verifier stated PASS but %d check(s) reported FAIL — "
            "overriding verdict to FAIL",
            len(failed_checks),
        )
        verdict = "FAIL"

    result = VerificationResult(
        verdict=verdict,
        raw_output=raw_output,
        failed_checks=failed_checks,
        token_usage=dict(holistic_tokens),
        cost=holistic_cost[0],
    )
    _log_verification_attempt(workspace_path, "holistic", 1, result)
    return result


def format_holistic_feedback(result: VerificationResult) -> str:
    """Format holistic verification failure as feedback for the orchestrator."""
    parts = [
        "Holistic pipeline verification FAILED.",
        "The following cross-skill issues were found:",
        "",
    ]
    if not result.failed_checks:
        parts.append(
            "(Verifier reported FAIL but no specific checks were extracted. "
            "Review the full verification log.)"
        )
    for check in result.failed_checks:
        parts.append(f"FAIL: {check['check']}")
        parts.append(f"  Command: {check['command']}")
        parts.append(f"  Output: {check['observed']}")
        parts.append("")

    parts.extend([
        "Re-invoke the affected skill(s) with force=true to fix these issues.",
        "Then verify cross-skill consistency again.",
    ])
    return "\n".join(parts)


def build_holistic_checker(
    skill_map: dict[str, dict[str, str]],
    workspace: Path,
    fork_child: Any,
    read_skill_content: Callable[[dict[str, dict[str, str]], str], str | None] | None = None,
    on_event: Any | None = None,
) -> Callable[[], Awaitable[VerificationResult]]:
    """Build a holistic check callable for ``send_until_complete``.

    Reads all skill definitions, gathers output dirs, discovers carry-forward
    artifacts, and returns an async callable that spawns a holistic verifier.

    Parameters
    ----------
    skill_map : dict
        Mapping of skill_name -> frontmatter dict (must include ``_path``
        and ``outputs`` keys).
    workspace : Path
        Workspace root path.
    fork_child : callable
        Async callable to fork a child session.
    read_skill_content : callable, optional
        Function to read a skill's SKILL.md content. If None, reads directly
        from the ``_path`` in skill_map.
    """
    async def _check() -> VerificationResult:
        skill_definitions: dict[str, str] = {}
        output_dirs: dict[str, str] = {}

        for name, fm in skill_map.items():
            # Read skill content
            if read_skill_content is not None:
                content = read_skill_content(skill_map, name)
            else:
                skill_path = fm.get("_path")
                if skill_path and Path(skill_path).is_file():
                    content = Path(skill_path).read_text(encoding="utf-8")
                else:
                    content = None
            if content:
                skill_definitions[name] = content

            # Gather output dir
            outputs_rel = fm.get("outputs", "").strip()
            if outputs_rel:
                output_dirs[name] = str(workspace / outputs_rel)

        carry_forward = _find_carry_forward_artifacts(workspace)

        return await run_holistic_verification(
            skill_definitions=skill_definitions,
            output_dirs=output_dirs,
            workspace=str(workspace),
            fork_child=fork_child,
            carry_forward_artifacts=carry_forward,
            on_event=on_event,
        )

    return _check
