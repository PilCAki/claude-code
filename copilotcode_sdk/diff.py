"""Unified diff generation and change summary utilities."""
from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True, slots=True)
class DiffResult:
    """Result of a diff operation."""
    original_path: str
    modified_path: str
    unified_diff: str
    added_lines: int
    removed_lines: int
    changed: bool

    @property
    def summary(self) -> str:
        if not self.changed:
            return f"{self.original_path}: no changes"
        parts = []
        if self.added_lines:
            parts.append(f"+{self.added_lines}")
        if self.removed_lines:
            parts.append(f"-{self.removed_lines}")
        return f"{self.original_path}: {', '.join(parts)}"


def generate_diff(
    original: str,
    modified: str,
    *,
    original_path: str = "a/file",
    modified_path: str = "b/file",
    context_lines: int = 3,
) -> DiffResult:
    """Generate a unified diff between two strings.

    Args:
        original: The original file content.
        modified: The modified file content.
        original_path: Label for the original file in the diff header.
        modified_path: Label for the modified file in the diff header.
        context_lines: Number of context lines around each change.

    Returns:
        A DiffResult with the unified diff and line change counts.
    """
    original_lines = original.splitlines(keepends=True)
    modified_lines = modified.splitlines(keepends=True)

    diff_lines = list(difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile=original_path,
        tofile=modified_path,
        n=context_lines,
    ))

    added = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))

    return DiffResult(
        original_path=original_path,
        modified_path=modified_path,
        unified_diff="".join(diff_lines),
        added_lines=added,
        removed_lines=removed,
        changed=bool(diff_lines),
    )


def generate_file_diff(
    original_path: Path,
    modified_content: str,
    *,
    context_lines: int = 3,
) -> DiffResult:
    """Generate a diff between a file on disk and new content.

    If the file doesn't exist, treats the original as empty (new file).
    """
    if original_path.exists():
        original = original_path.read_text(encoding="utf-8")
    else:
        original = ""

    return generate_diff(
        original,
        modified_content,
        original_path=str(original_path),
        modified_path=str(original_path),
        context_lines=context_lines,
    )


def summarize_changes(diffs: Sequence[DiffResult]) -> str:
    """Produce a compact summary of multiple diffs.

    Returns a markdown-formatted change summary.
    """
    if not diffs:
        return "No changes."

    changed = [d for d in diffs if d.changed]
    if not changed:
        return "No changes."

    total_added = sum(d.added_lines for d in changed)
    total_removed = sum(d.removed_lines for d in changed)

    lines = [
        f"**{len(changed)} file(s) changed** (+{total_added}, -{total_removed})",
        "",
    ]
    for d in changed:
        lines.append(f"- {d.summary}")

    return "\n".join(lines)


def apply_patch(original: str, unified_diff: str) -> str | None:
    """Apply a unified diff patch to original content.

    Returns the patched content, or None if the patch cannot be applied.
    This is a best-effort implementation using difflib — it handles
    simple patches but may fail on complex ones.
    """
    # Parse the unified diff to extract hunks
    hunks: list[tuple[int, list[str], list[str]]] = []
    current_start = 0
    remove_lines: list[str] = []
    add_lines: list[str] = []
    in_hunk = False

    for line in unified_diff.splitlines(keepends=True):
        if line.startswith("@@"):
            if in_hunk and (remove_lines or add_lines):
                hunks.append((current_start, remove_lines, add_lines))
            # Parse hunk header: @@ -start,count +start,count @@
            parts = line.split()
            if len(parts) >= 3:
                try:
                    old_range = parts[1]  # -start,count
                    current_start = abs(int(old_range.split(",")[0])) - 1
                except (ValueError, IndexError):
                    current_start = 0
            remove_lines = []
            add_lines = []
            in_hunk = True
        elif in_hunk:
            if line.startswith("-") and not line.startswith("---"):
                remove_lines.append(line[1:])
            elif line.startswith("+") and not line.startswith("+++"):
                add_lines.append(line[1:])
            elif line.startswith(" "):
                # Context line — flush current changes
                if remove_lines or add_lines:
                    hunks.append((current_start, remove_lines, add_lines))
                    current_start += len(remove_lines)
                    remove_lines = []
                    add_lines = []
                current_start += 1

    if in_hunk and (remove_lines or add_lines):
        hunks.append((current_start, remove_lines, add_lines))

    if not hunks:
        return None

    # Apply hunks in reverse order to preserve line numbers
    result_lines = original.splitlines(keepends=True)
    for start, removes, adds in reversed(hunks):
        end = start + len(removes)
        result_lines[start:end] = adds

    return "".join(result_lines)
