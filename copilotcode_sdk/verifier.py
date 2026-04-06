"""Verification gate for CompleteSkill.

Spawns an adversarial read-only sub-agent to check skill outputs before
allowing skill completion. The verifier must run commands and produce
evidence — reading code and reasoning is not verification.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .subagent import SubagentSpec

logger = logging.getLogger("copilotcode.verifier")


MAX_VERIFICATION_ATTEMPTS = 5
MAX_VERIFIER_MALFUNCTIONS = 2

VERIFIER_TOOLS = ("Read", "Bash", "Glob", "Grep")
VERIFIER_MAX_TURNS = 20
VERIFIER_TIMEOUT = 300.0


class VerificationExhaustedError(Exception):
    """Raised when a skill exhausts all verification attempts."""

    def __init__(self, skill_name: str, trace_path: str) -> None:
        self.skill_name = skill_name
        self.trace_path = trace_path
        super().__init__(
            f"Skill '{skill_name}' failed verification after "
            f"{MAX_VERIFICATION_ATTEMPTS} attempts. "
            f"Trace: {trace_path}"
        )


@dataclass
class VerificationResult:
    """Result from a single verification run."""
    verdict: str  # "PASS", "FAIL", "PARTIAL", "MALFUNCTION"
    raw_output: str
    failed_checks: list[dict[str, str]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"

    @property
    def failed(self) -> bool:
        return self.verdict in ("FAIL", "PARTIAL")

    @property
    def is_malfunction(self) -> bool:
        return self.verdict == "MALFUNCTION"


def snapshot_output_hashes(output_dir: Path) -> dict[str, str]:
    """Compute SHA-256 hashes for all files in the output directory.

    Returns {relative_path: hex_digest} dict. Uses forward slashes
    for consistency across platforms.
    """
    hashes: dict[str, str] = {}
    if not output_dir.exists():
        return hashes
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            rel = f.relative_to(output_dir).as_posix()
            hashes[rel] = hashlib.sha256(f.read_bytes()).hexdigest()
    return hashes


def compare_output_hashes(
    output_dir: Path,
    before: dict[str, str],
) -> list[str]:
    """Compare current file hashes against a previous snapshot.

    Returns list of relative paths that were modified, added, or deleted.
    """
    after = snapshot_output_hashes(output_dir)
    changed: list[str] = []

    for path, old_hash in before.items():
        new_hash = after.get(path)
        if new_hash is None:
            changed.append(path)
        elif new_hash != old_hash:
            changed.append(path)

    for path in after:
        if path not in before:
            changed.append(path)

    return sorted(changed)


VERIFIER_SYSTEM_PROMPT = """You are a verification specialist. Your job is to verify that a skill's \
outputs are correct — not to do the skill's work yourself.

You will receive:
1. The skill's definition (what it should produce, quality rubric, success criteria)
2. The path to the output directory
3. Historical metrics from prior runs (if available)

=== RULES ===
- You MUST run commands for every check. Reading code/files and reasoning \
about correctness is NOT verification. Execute queries, count rows, \
check schemas, compare values.
- A check without a command run is a SKIP, not a PASS.
- Do NOT create, modify, or fix any output files.
- Do NOT write to the workspace or output directories.
- Do NOT attempt to redo the skill's work.
- You may write ephemeral test scripts to a temp directory only.

=== RECOGNIZE YOUR OWN RATIONALIZATIONS ===
- "The code looks correct based on my reading" — reading is not verification. Run it.
- "This is probably fine" — probably is not verified. Run it.
- "This would take too long" — not your call.
If you catch yourself writing an explanation instead of a command, stop. Run the command.

=== WHAT TO CHECK ===
- Output files exist in expected formats
- Schema correctness (column names, types)
- Row counts are reasonable (non-zero, within tolerance of source/prior runs)
- Data integrity (no all-null columns, amounts are numeric, dates parse)
- Cross-reference with prior_run_metrics.json if provided (flag large deviations)
- Quality rubric items from the skill definition
- Cross-skill consistency where applicable

=== OUTPUT FORMAT ===
Every check MUST follow this structure. A check without a Command run block \
is not a PASS — it's a skip.

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
environmental limitations only (tool unavailable, server can't start) — not \
for "I'm unsure."
"""


def parse_verdict(output: str) -> str | None:
    """Extract the verdict from verifier output."""
    for line in reversed(output.strip().splitlines()):
        line = line.strip()
        m = re.match(r"^VERDICT:\s*(PASS|FAIL|PARTIAL)\s*$", line)
        if m:
            return m.group(1)
    return None


def _has_command_blocks(output: str) -> bool:
    """Check that the verifier output contains at least one Command run block."""
    return "**Command run:**" in output


def extract_failed_checks(output: str) -> list[dict[str, str]]:
    """Extract FAIL check blocks from verifier output."""
    checks: list[dict[str, str]] = []
    blocks = re.split(r"(?=### Check:)", output)
    for block in blocks:
        if "**Result: FAIL**" not in block:
            continue
        check_match = re.search(r"### Check:\s*(.+)", block)
        cmd_match = re.search(
            r"\*\*Command run:\*\*\s*\n\s*(.+?)(?=\n\*\*|\Z)",
            block,
            re.DOTALL,
        )
        obs_match = re.search(
            r"\*\*Output observed:\*\*\s*\n\s*(.+?)(?=\n\*\*|\Z)",
            block,
            re.DOTALL,
        )
        checks.append({
            "check": check_match.group(1).strip() if check_match else "(unknown)",
            "command": cmd_match.group(1).strip() if cmd_match else "(none)",
            "observed": obs_match.group(1).strip() if obs_match else "(none)",
        })
    return checks


def build_verifier_prompt(
    skill_content: str,
    output_dir: str,
    prior_metrics: str | None,
) -> str:
    """Build the user prompt for the verification sub-agent."""
    metrics_section = prior_metrics if prior_metrics else "None available — skip historical comparison"
    return (
        "=== IMPORTANT: VERIFICATION ONLY ===\n"
        "You are verifying the outputs of the skill below. Do NOT implement the\n"
        "skill. Do NOT redo any work. Only check that the outputs are correct.\n"
        "\n"
        "=== SKILL DEFINITION ===\n"
        f"{skill_content}\n"
        "\n"
        "=== OUTPUT DIRECTORY ===\n"
        f"{output_dir}\n"
        "\n"
        "=== PRIOR RUN METRICS ===\n"
        f"{metrics_section}\n"
    )


def format_fail_feedback(
    failed_checks: list[dict[str, str]],
    attempt: int,
    max_attempts: int,
) -> str:
    """Format verification failure feedback for the implementer."""
    parts = [
        f"Verification FAILED (attempt {attempt} of {max_attempts}). "
        "Fix the following issues and call CompleteSkill again:",
        "",
    ]
    if not failed_checks:
        parts.append("(Verifier reported FAIL but no specific checks were extracted.)")
    for check in failed_checks:
        parts.append(f"FAIL: {check['check']}")
        parts.append(f"  Command: {check['command']}")
        parts.append(f"  Output: {check['observed']}")
        parts.append("")
    return "\n".join(parts)


def write_failure_trace(
    skill_name: str,
    history: list[dict[str, Any]],
    output_dir: Path,
    workspace: Path,
) -> str:
    """Write a failure trace JSON file and return its path."""
    trace_dir = workspace / "outputs" / "verification_failures"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"{skill_name}_failure_trace.json"

    files: list[str] = []
    total_bytes = 0
    if output_dir.exists():
        for f in sorted(output_dir.rglob("*")):
            if f.is_file():
                files.append(f.relative_to(output_dir).as_posix())
                total_bytes += f.stat().st_size

    trace = {
        "skill": skill_name,
        "attempts": len(history),
        "final_verdict": "FAIL",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "history": history,
        "output_snapshot": {
            "files": files,
            "total_bytes": total_bytes,
        },
    }

    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    return str(trace_path)


def _log_verification_attempt(
    workspace: Path,
    skill_name: str,
    attempt_num: int,
    result: VerificationResult,
) -> Path:
    """Write a verification attempt log to disk and log a summary to console.

    Creates one file per attempt:
      outputs/verification_logs/{skill_name}_attempt_{N}.md

    Returns the path to the log file.
    """
    log_dir = workspace / "outputs" / "verification_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{skill_name}_attempt_{attempt_num:02d}.md"

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    header = (
        f"# Verification: {skill_name} — Attempt {attempt_num}\n"
        f"**Timestamp:** {ts}\n"
        f"**Verdict:** {result.verdict}\n"
    )

    if result.failed_checks:
        header += f"**Failed checks:** {len(result.failed_checks)}\n"
        for fc in result.failed_checks:
            header += f"  - {fc.get('check', '?')}\n"

    header += f"\n---\n\n## Full Verifier Output\n\n"

    log_path.write_text(header + result.raw_output, encoding="utf-8")

    # Console logging — visible in run.py output
    verdict_str = result.verdict
    if result.failed_checks:
        checks_summary = "; ".join(fc.get("check", "?") for fc in result.failed_checks)
        logger.info(
            "VERIFY %s attempt %d → %s | Failed: %s | Log: %s",
            skill_name, attempt_num, verdict_str, checks_summary, log_path.name,
        )
    else:
        logger.info(
            "VERIFY %s attempt %d → %s | Log: %s",
            skill_name, attempt_num, verdict_str, log_path.name,
        )

    return log_path


async def run_verification(
    skill_name: str,
    skill_content: str,
    output_dir: Path,
    workspace: Path,
    fork_child: Any,
    prior_metrics: str | None,
    attempt_num: int = 0,
) -> VerificationResult:
    """Spawn a verifier sub-agent and return the result.

    This is a single verification attempt. The caller (CompleteSkill handler)
    manages retry loops and attempt counting.

    *attempt_num* is used for log file naming only — the caller tracks the
    authoritative count.
    """
    logger.info("Starting verification for '%s' (attempt %d)...", skill_name, attempt_num)

    # 1. Snapshot output hashes before verification
    before_hashes = snapshot_output_hashes(output_dir)

    # 2. Build the verifier spec and prompt
    spec = SubagentSpec(
        role="skill-verifier",
        system_prompt_suffix=VERIFIER_SYSTEM_PROMPT,
        tools=VERIFIER_TOOLS,
        max_turns=VERIFIER_MAX_TURNS,
        timeout_seconds=VERIFIER_TIMEOUT,
    )

    user_prompt = build_verifier_prompt(
        skill_content=skill_content,
        output_dir=str(output_dir),
        prior_metrics=prior_metrics,
    )

    # 3. Spawn and run the verifier
    try:
        child = await fork_child(spec)
        try:
            raw_output = await child.send_and_wait(user_prompt, timeout=VERIFIER_TIMEOUT)
            if not isinstance(raw_output, str):
                raw_output = str(raw_output)
        finally:
            await child.destroy()
    except Exception as exc:
        result = VerificationResult(
            verdict="MALFUNCTION",
            raw_output=f"Verifier crashed: {exc}",
        )
        _log_verification_attempt(workspace, skill_name, attempt_num, result)
        return result

    # 4. Check for output tampering
    changed = compare_output_hashes(output_dir, before_hashes)
    if changed:
        result = VerificationResult(
            verdict="MALFUNCTION",
            raw_output=(
                f"Verifier tampered with output files: {', '.join(changed)}. "
                "Verification invalidated."
            ),
        )
        _log_verification_attempt(workspace, skill_name, attempt_num, result)
        return result

    # 5. Parse verdict
    verdict = parse_verdict(raw_output)
    if verdict is None:
        result = VerificationResult(
            verdict="MALFUNCTION",
            raw_output=(
                "Verification failed: verifier did not produce a verdict in the "
                "required format. Expected final line: VERDICT: PASS, VERDICT: FAIL, "
                "or VERDICT: PARTIAL (exact string, no markdown, no punctuation).\n\n"
                "--- RAW VERIFIER OUTPUT ---\n" + raw_output
            ),
        )
        _log_verification_attempt(workspace, skill_name, attempt_num, result)
        return result

    # 6. Check that commands were actually run
    if not _has_command_blocks(raw_output):
        result = VerificationResult(
            verdict="MALFUNCTION",
            raw_output=(
                "Verification failed: verifier did not run any commands. Every "
                "check must include a 'Command run:' block with actual terminal "
                "output. A check without a command is a skip, not a PASS.\n\n"
                "--- RAW VERIFIER OUTPUT ---\n" + raw_output
            ),
        )
        _log_verification_attempt(workspace, skill_name, attempt_num, result)
        return result

    # 7. Extract failed checks and return
    failed_checks = extract_failed_checks(raw_output) if verdict != "PASS" else []
    result = VerificationResult(
        verdict=verdict,
        raw_output=raw_output,
        failed_checks=failed_checks,
    )
    _log_verification_attempt(workspace, skill_name, attempt_num, result)
    return result
