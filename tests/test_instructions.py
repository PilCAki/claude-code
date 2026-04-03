from __future__ import annotations

import hashlib
from pathlib import Path

from copilotcode_sdk.instructions import InstructionBundle, load_workspace_instructions


def test_loads_claude_md_and_agents_md(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("# Project Rules\nBe concise.", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("# Agent Rules\nUse TDD.", encoding="utf-8")

    bundle = load_workspace_instructions(tmp_path)

    assert "Be concise." in bundle.content
    assert "Use TDD." in bundle.content
    assert len(bundle.sources) == 2


def test_loads_copilot_instructions(tmp_path: Path) -> None:
    gh = tmp_path / ".github"
    gh.mkdir()
    (gh / "copilot-instructions.md").write_text("Follow conventions.", encoding="utf-8")

    bundle = load_workspace_instructions(tmp_path)

    assert "Follow conventions." in bundle.content
    assert len(bundle.sources) == 1


def test_loads_rules_directory(tmp_path: Path) -> None:
    rules = tmp_path / ".claude" / "rules"
    rules.mkdir(parents=True)
    (rules / "style.md").write_text("Use snake_case.", encoding="utf-8")
    (rules / "testing.md").write_text("Always test.", encoding="utf-8")

    bundle = load_workspace_instructions(tmp_path)

    assert "snake_case" in bundle.content
    assert "Always test" in bundle.content
    assert len(bundle.sources) == 2


def test_include_expansion(tmp_path: Path) -> None:
    (tmp_path / "extra.md").write_text("Included content.", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text(
        "Main rules.\n@include extra.md\nMore rules.",
        encoding="utf-8",
    )

    bundle = load_workspace_instructions(tmp_path)

    assert "Included content." in bundle.content
    assert "Main rules." in bundle.content
    assert "More rules." in bundle.content


def test_include_depth_limit(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("@include b.md", encoding="utf-8")
    (tmp_path / "b.md").write_text("@include c.md", encoding="utf-8")
    (tmp_path / "c.md").write_text("@include d.md", encoding="utf-8")
    (tmp_path / "d.md").write_text("Too deep.", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("@include a.md", encoding="utf-8")

    bundle = load_workspace_instructions(tmp_path, max_include_depth=3)

    # depth 0: CLAUDE.md -> a.md (depth 1) -> b.md (depth 2) -> c.md (depth 3) -> d.md would be depth 4
    assert "Too deep." not in bundle.content


def test_html_comment_stripping(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text(
        "Keep this.\n<!-- Remove this. -->\nKeep this too.",
        encoding="utf-8",
    )

    bundle = load_workspace_instructions(tmp_path)

    assert "Keep this." in bundle.content
    assert "Remove this." not in bundle.content
    assert "Keep this too." in bundle.content


def test_memory_md_truncation(tmp_path: Path) -> None:
    lines = [f"Line {i}" for i in range(300)]
    (tmp_path / "MEMORY.md").write_text("\n".join(lines), encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("@include MEMORY.md", encoding="utf-8")

    bundle = load_workspace_instructions(tmp_path)

    assert "Line 0" in bundle.content
    assert "Line 199" in bundle.content
    assert "Line 200" not in bundle.content


def test_content_hash_is_deterministic(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("Same content.", encoding="utf-8")

    b1 = load_workspace_instructions(tmp_path)
    b2 = load_workspace_instructions(tmp_path)

    assert b1.content_hash == b2.content_hash
    assert b1.content_hash != ""


def test_content_hash_changes_on_different_content(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("Version 1.", encoding="utf-8")
    b1 = load_workspace_instructions(tmp_path)

    (tmp_path / "CLAUDE.md").write_text("Version 2.", encoding="utf-8")
    b2 = load_workspace_instructions(tmp_path)

    assert b1.content_hash != b2.content_hash


def test_empty_workspace_returns_empty_bundle(tmp_path: Path) -> None:
    bundle = load_workspace_instructions(tmp_path)

    assert bundle.content == ""
    assert bundle.sources == []
    assert bundle.content_hash != ""


def test_loading_order_rules_before_project_before_github(tmp_path: Path) -> None:
    rules = tmp_path / ".claude" / "rules"
    rules.mkdir(parents=True)
    (rules / "first.md").write_text("RULES_CONTENT", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("PROJECT_CONTENT", encoding="utf-8")
    gh = tmp_path / ".github"
    gh.mkdir()
    (gh / "copilot-instructions.md").write_text("GITHUB_CONTENT", encoding="utf-8")

    bundle = load_workspace_instructions(tmp_path)

    rules_pos = bundle.content.index("RULES_CONTENT")
    project_pos = bundle.content.index("PROJECT_CONTENT")
    github_pos = bundle.content.index("GITHUB_CONTENT")
    assert rules_pos < project_pos < github_pos
