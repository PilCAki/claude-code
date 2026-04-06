"""Verification gate for CompleteSkill.

Spawns an adversarial read-only sub-agent to check skill outputs before
allowing skill completion. The verifier must run commands and produce
evidence — reading code and reasoning is not verification.
"""
from __future__ import annotations

import hashlib
import json
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
