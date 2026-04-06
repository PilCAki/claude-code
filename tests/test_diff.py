"""Tests for copilotcode_sdk.diff module."""
from __future__ import annotations

from pathlib import Path

from copilotcode_sdk.diff import (
    DiffResult,
    apply_patch,
    generate_diff,
    generate_file_diff,
    summarize_changes,
)


def test_generate_diff_identical() -> None:
    result = generate_diff("hello\n", "hello\n")
    assert not result.changed
    assert result.added_lines == 0
    assert result.removed_lines == 0
    assert result.unified_diff == ""


def test_generate_diff_addition() -> None:
    result = generate_diff("line1\n", "line1\nline2\n")
    assert result.changed
    assert result.added_lines == 1
    assert result.removed_lines == 0
    assert "+line2" in result.unified_diff


def test_generate_diff_removal() -> None:
    result = generate_diff("line1\nline2\n", "line1\n")
    assert result.changed
    assert result.removed_lines == 1
    assert "-line2" in result.unified_diff


def test_generate_diff_modification() -> None:
    result = generate_diff("old\n", "new\n")
    assert result.changed
    assert result.added_lines == 1
    assert result.removed_lines == 1


def test_generate_diff_custom_paths() -> None:
    result = generate_diff("a\n", "b\n", original_path="src/old.py", modified_path="src/new.py")
    assert "src/old.py" in result.unified_diff
    assert "src/new.py" in result.unified_diff


def test_generate_file_diff_existing(tmp_path: Path) -> None:
    target = tmp_path / "file.py"
    target.write_text("original\n", encoding="utf-8")
    result = generate_file_diff(target, "modified\n")
    assert result.changed
    assert result.added_lines == 1
    assert result.removed_lines == 1


def test_generate_file_diff_new_file(tmp_path: Path) -> None:
    target = tmp_path / "new.py"
    result = generate_file_diff(target, "content\n")
    assert result.changed
    assert result.added_lines == 1
    assert result.removed_lines == 0


def test_summary_empty() -> None:
    assert summarize_changes([]) == "No changes."


def test_summary_no_changes() -> None:
    d = DiffResult("a", "b", "", 0, 0, False)
    assert summarize_changes([d]) == "No changes."


def test_summary_with_changes() -> None:
    d1 = DiffResult("src/a.py", "src/a.py", "...", 5, 2, True)
    d2 = DiffResult("src/b.py", "src/b.py", "...", 3, 0, True)
    summary = summarize_changes([d1, d2])
    assert "2 file(s) changed" in summary
    assert "+8" in summary
    assert "-2" in summary


def test_diff_result_summary_property() -> None:
    d = DiffResult("file.py", "file.py", "...", 3, 1, True)
    assert "+3" in d.summary
    assert "-1" in d.summary


def test_apply_patch_simple() -> None:
    original = "line1\nold\nline3\n"
    diff = generate_diff(original, "line1\nnew\nline3\n")
    result = apply_patch(original, diff.unified_diff)
    assert result is not None
    assert "new" in result
    assert "old" not in result


def test_apply_patch_no_hunks() -> None:
    assert apply_patch("hello", "") is None
