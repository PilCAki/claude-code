"""Verification gate for CompleteSkill.

Spawns an adversarial read-only sub-agent to check skill outputs before
allowing skill completion. The verifier must run commands and produce
evidence — reading code and reasoning is not verification.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


MAX_VERIFICATION_ATTEMPTS = 5
MAX_VERIFIER_MALFUNCTIONS = 2


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
