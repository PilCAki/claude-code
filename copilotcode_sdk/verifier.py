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

# Don't restrict verifier tools — the available_tools filter may not match
# the raw CLI's internal tool names (e.g. "Bash" vs "powershell").
# The verifier prompt already instructs it not to modify files.
VERIFIER_TOOLS = ()
VERIFIER_MAX_TURNS = 20
VERIFIER_TIMEOUT = 3600.0  # 1 hour — verifier may need to inspect large output dirs


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
    is_tamper: bool = False  # True when verifier modified existing output files
    token_usage: dict[str, int] = field(default_factory=dict)
    cost: float = 0.0

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

=== DETECT GAMING AND MANUFACTURED COMPLIANCE ===
- If content appears padded (repeated filler, HTML comments with bulk text, \
lorem ipsum, placeholder paragraphs), that is a FAIL — not a PASS.
- If a metric passes nominally but the underlying content is hollow \
(e.g., file is large but sections are empty or repetitive), FAIL it.
- If self_score.json claims high scores but the actual artifacts don't \
support them, FAIL with specific evidence of inflation.
- A check that can be gamed by adding junk is a bad check. Evaluate \
substance, not proxies.
- CRITICAL: No mock datasets or mock functionality. Open at least 2-3 detail \
files and verify the values come from real source data, not invented samples. \
Row counts in output files must be proportional to the source data.

=== WHAT TO CHECK ===
- Output files exist in expected formats
- Schema correctness (column names, types)
- Row counts are reasonable (non-zero, within tolerance of source/prior runs)
- Data integrity (no all-null columns, amounts are numeric, dates parse)
- Cross-reference with prior_run_metrics.json if provided (flag large deviations)
- Quality rubric items from the skill definition
- Cross-skill consistency where applicable

=== RUBRIC ENFORCEMENT ===
If the skill definition contains a quality rubric with scored dimensions:
1. Check for self_score.json in the output directory. If missing, FAIL with \
"self_score.json not found — child did not self-score."
2. Verify the JSON is well-formed and covers ALL rubric dimensions.
3. For each empirically checkable dimension, independently verify the claimed score:
   - Sheet coverage: count exported files vs claimed
   - Profile completeness: check profiles have non-blank purpose and quality notes
   - Row counts: query parquet files and compare to metadata
   - Section completeness: check HTML sections exist with real content
   - Narrative substance: verify sections contain multi-sentence text, not filler
4. If your independent score is lower than self-score on 2+ dimensions, \
FAIL with "Self-score inflation detected" and list discrepancies.
5. If total < skill's stated minimum threshold, FAIL.
6. self_score.json that reports all 2s with vague justifications is suspicious — \
verify at least 3 dimensions empirically.

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
        m = re.match(r"^(?:FINAL\s+)?VERDICT:\s*(PASS|FAIL|PARTIAL)\s*$", line)
        if m:
            return m.group(1)
    return None


def _has_command_blocks(output: str) -> bool:
    """Check that the verifier output contains at least one Command run block."""
    # Accept with or without markdown bold markers
    return "Command run:" in output or "**Command run:**" in output


def extract_failed_checks(output: str) -> list[dict[str, str]]:
    """Extract FAIL check blocks from verifier output.

    Matches both bold (``**Result: FAIL**``) and plain (``Result: FAIL``)
    markers so the check works regardless of the verifier's markdown style.
    """
    checks: list[dict[str, str]] = []
    blocks = re.split(r"(?=### Check:)", output)
    for block in blocks:
        # Match bold or plain "Result: FAIL" (with optional trailing text)
        if not re.search(r"(?:\*\*)?Result:\s*FAIL(?:\*\*)?", block):
            continue
        check_match = re.search(r"### Check:\s*(.+)", block)
        # Try bold then plain for command/output blocks
        cmd_match = re.search(
            r"(?:\*\*)?Command run:(?:\*\*)?\s*\n\s*(.+?)(?=\n(?:\*\*)?|\Z)",
            block,
            re.DOTALL,
        )
        obs_match = re.search(
            r"(?:\*\*)?Output observed:(?:\*\*)?\s*\n\s*(.+?)(?=\n(?:\*\*)?|\Z)",
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
    prior_attempts: list[dict[str, Any]] | None = None,
) -> str:
    """Build the user prompt for the verification sub-agent.

    Parameters
    ----------
    prior_attempts : list, optional
        History from earlier verification attempts on the same skill invocation.
        Each entry has ``verdict``, ``failed_checks``, and optionally
        ``raw_output``.  Passed so the fresh verifier knows what was tried
        before and can avoid repeating the same mistakes (e.g. tampering).
    """
    metrics_section = prior_metrics if prior_metrics else "None available — skip historical comparison"

    parts = [
        "=== IMPORTANT: VERIFICATION ONLY ===",
        "You are verifying the outputs of the skill below. Do NOT implement the",
        "skill. Do NOT redo any work. Only check that the outputs are correct.",
        "",
    ]

    # Prior attempt history — helps fresh verifiers avoid past mistakes
    if prior_attempts:
        parts.append("=== PRIOR VERIFICATION ATTEMPTS ===")
        parts.append(
            "This is NOT your first verification of this skill invocation. "
            "Previous verifier instances ran and either failed or malfunctioned. "
            "Learn from their mistakes — do NOT repeat them."
        )
        parts.append("")
        for i, attempt in enumerate(prior_attempts, 1):
            verdict = attempt.get("verdict", "UNKNOWN")
            parts.append(f"--- Attempt {i}: {verdict} ---")
            if verdict == "MALFUNCTION":
                raw = attempt.get("raw_output", "")
                if "tamper" in raw.lower():
                    parts.append(
                        "!! CRITICAL: This verifier was TERMINATED because it "
                        "wrote files into the output directory. This is "
                        "STRICTLY FORBIDDEN. You must NEVER create, modify, or "
                        "write ANY files in the output directory or workspace. "
                        "If you need to run a script, write it to a temp "
                        "directory ONLY (e.g. $env:TEMP or /tmp)."
                    )
                else:
                    parts.append(f"Reason: {raw[:500]}")
            else:
                failed = attempt.get("failed_checks", [])
                if failed:
                    parts.append("Failed checks (the child has since attempted fixes):")
                    for fc in failed:
                        check = fc.get("check", "?")
                        parts.append(f"  - {check}")
                else:
                    parts.append("(No specific failed checks extracted)")
            parts.append("")

        parts.append(
            "The child agent received feedback after each failure and attempted "
            "to fix the issues. Your job is to verify the CURRENT state of "
            "outputs — not to assume prior issues still exist. Re-check "
            "everything from scratch."
        )
        parts.append("")

    parts.extend([
        "=== SKILL DEFINITION ===",
        skill_content,
        "",
        "=== OUTPUT DIRECTORY ===",
        output_dir,
        "",
        "=== PRIOR RUN METRICS ===",
        metrics_section,
    ])

    return "\n".join(parts)


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
    # For MALFUNCTIONs, use a separate counter to avoid overwriting prior logs
    if result.verdict == "MALFUNCTION":
        # Find next available malfunction suffix for this attempt
        existing = list(log_dir.glob(
            f"{skill_name}_attempt_{attempt_num:02d}_malfunction_*.md"
        ))
        mal_idx = len(existing) + 1
        log_path = log_dir / (
            f"{skill_name}_attempt_{attempt_num:02d}_malfunction_{mal_idx:02d}.md"
        )
    else:
        log_path = log_dir / f"{skill_name}_attempt_{attempt_num:02d}.md"

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    tok = result.token_usage
    header = (
        f"# Verification: {skill_name} — Attempt {attempt_num}\n"
        f"**Timestamp:** {ts}\n"
        f"**Verdict:** {result.verdict}\n"
        f"**Cost:** ${result.cost:.6f}\n"
        f"**Tokens:** in={tok.get('input', 0)} out={tok.get('output', 0)} "
        f"cache_r={tok.get('cache_read', 0)} cache_w={tok.get('cache_write', 0)}\n"
    )

    if result.failed_checks:
        header += f"**Failed checks:** {len(result.failed_checks)}\n"
        for fc in result.failed_checks:
            header += f"  - {fc.get('check', '?')}\n"

    # Classify MALFUNCTION reason for logging
    malfunction_reason = ""
    if result.verdict == "MALFUNCTION":
        raw = result.raw_output
        if "tampered with output files" in raw:
            malfunction_reason = "TAMPER: " + raw.split("Verification invalidated")[0].strip()
        elif "Verifier crashed" in raw:
            malfunction_reason = raw[:200]
        elif "did not produce a verdict" in raw:
            malfunction_reason = "NO_VERDICT"
        elif "did not run any commands" in raw:
            malfunction_reason = "NO_COMMANDS"
        else:
            malfunction_reason = raw[:200]
        header += f"**Malfunction reason:** {malfunction_reason}\n"

    header += f"\n---\n\n## Full Verifier Output\n\n"

    log_path.write_text(header + result.raw_output, encoding="utf-8")

    # Console logging — visible in run.py output
    verdict_str = result.verdict
    cost_str = f"${result.cost:.4f}" if result.cost > 0 else "$0"
    tok_str = (
        f"in={tok.get('input', 0)} out={tok.get('output', 0)}"
        if tok else "no-tokens"
    )
    if result.verdict == "MALFUNCTION":
        logger.warning(
            "VERIFY %s attempt %d → MALFUNCTION | %s %s | Reason: %s | Log: %s",
            skill_name, attempt_num, cost_str, tok_str, malfunction_reason, log_path.name,
        )
    elif result.failed_checks:
        checks_summary = "; ".join(fc.get("check", "?") for fc in result.failed_checks)
        logger.info(
            "VERIFY %s attempt %d → %s | %s %s | Failed: %s | Log: %s",
            skill_name, attempt_num, verdict_str, cost_str, tok_str, checks_summary, log_path.name,
        )
    else:
        logger.info(
            "VERIFY %s attempt %d → %s | %s %s | Log: %s",
            skill_name, attempt_num, verdict_str, cost_str, tok_str, log_path.name,
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
    on_event: Any | None = None,
    prior_attempts: list[dict[str, Any]] | None = None,
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
        prior_attempts=prior_attempts,
    )

    # 3. Spawn and run the verifier, tracking token usage
    verifier_tokens: dict[str, int] = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    verifier_cost: list[float] = [0.0]

    def _track_verifier_event(evt: Any) -> None:
        """Capture token counts from the verifier child for logging."""
        try:
            if isinstance(evt, dict):
                etype = evt.get("type", "")
            else:
                etype = getattr(evt, "type", "")
                if hasattr(etype, "value"):
                    etype = etype.value
            if etype == "cost.accumulated":
                _g = (lambda k, d=0: evt.get(k, d)) if isinstance(evt, dict) else (lambda k, d=0: getattr(evt, k, d))
                verifier_tokens["input"] += int(_g("input_tokens", 0) or 0)
                verifier_tokens["output"] += int(_g("output_tokens", 0) or 0)
                verifier_tokens["cache_read"] += int(_g("cache_read_tokens", 0) or 0)
                verifier_tokens["cache_write"] += int(_g("cache_write_tokens", 0) or 0)
                verifier_cost[0] += float(_g("turn_cost", 0) or 0)
        except Exception:
            pass
        if on_event is not None:
            try:
                on_event(evt)
            except Exception:
                pass

    try:
        child = await fork_child(spec, on_event=_track_verifier_event)
        try:
            await child.send_and_wait(user_prompt, timeout=VERIFIER_TIMEOUT)
            raw_output = await child.get_last_response_text()
            if not raw_output:
                raw_output = "(verifier produced no text response)"
        finally:
            await child.destroy()
    except Exception as exc:
        result = VerificationResult(
            verdict="MALFUNCTION",
            raw_output=f"Verifier crashed: {exc}",
        )
        result.token_usage = dict(verifier_tokens)
        result.cost = verifier_cost[0]
        _log_verification_attempt(workspace, skill_name, attempt_num, result)
        return result

    # 4. Check for output tampering
    #    Added files (scripts the verifier wrote) are cleaned up but NOT treated
    #    as MALFUNCTION — the verifier legitimately needs to create temp scripts.
    #    Only *modified* existing artifacts are real tampering.
    changed = compare_output_hashes(output_dir, before_hashes)
    if changed:
        added = [p for p in changed if p not in before_hashes]
        modified = [p for p in changed if p in before_hashes]

        # Clean up added files silently
        for rel_path in added:
            victim = output_dir / rel_path
            if victim.exists():
                victim.unlink()
                logger.info("VERIFY cleanup: removed verifier script %s", rel_path)

        # Only MALFUNCTION if existing artifacts were modified
        if modified:
            logger.warning(
                "TAMPER: verifier modified existing output files %s — "
                "cannot auto-restore",
                modified,
            )
            result = VerificationResult(
                verdict="MALFUNCTION",
                raw_output=(
                    f"Verifier tampered with output files — modified existing "
                    f"output artifacts: {', '.join(modified)}. "
                    f"This invalidates the verification. "
                    f"(Also cleaned up {len(added)} added script(s).)"
                ),
                is_tamper=True,
                token_usage=dict(verifier_tokens),
                cost=verifier_cost[0],
            )
            _log_verification_attempt(workspace, skill_name, attempt_num, result)
            return result

        if added:
            logger.info(
                "VERIFY cleanup: removed %d script(s) the verifier wrote to "
                "output dir: %s",
                len(added), added,
            )

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
            token_usage=dict(verifier_tokens),
            cost=verifier_cost[0],
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
            token_usage=dict(verifier_tokens),
            cost=verifier_cost[0],
        )
        _log_verification_attempt(workspace, skill_name, attempt_num, result)
        return result

    # 7. Extract failed checks — always, regardless of stated verdict
    failed_checks = extract_failed_checks(raw_output)

    # 8. Enforce verdict consistency: if any check reported FAIL, the
    #    overall verdict cannot be PASS.  The verifier model sometimes
    #    rationalises a PASS despite individual failures — override it.
    if verdict == "PASS" and failed_checks:
        logger.warning(
            "Verifier stated PASS but %d check(s) reported FAIL — "
            "overriding verdict to FAIL",
            len(failed_checks),
        )
        verdict = "FAIL"

    result = VerificationResult(
        verdict=verdict,
        raw_output=raw_output,
        failed_checks=failed_checks,
        token_usage=dict(verifier_tokens),
        cost=verifier_cost[0],
    )
    _log_verification_attempt(workspace, skill_name, attempt_num, result)
    return result
