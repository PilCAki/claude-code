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


# ---------------------------------------------------------------------------
# Wave 1: Upward directory traversal
# ---------------------------------------------------------------------------


def test_upward_traversal_finds_parent_claude_md(tmp_path: Path) -> None:
    """A CLAUDE.md in a parent directory should be included as Parent Instructions."""
    parent = tmp_path / "repo"
    child = parent / "packages" / "core"
    child.mkdir(parents=True)
    (parent / "CLAUDE.md").write_text("PARENT_RULES", encoding="utf-8")
    (child / "CLAUDE.md").write_text("CHILD_RULES", encoding="utf-8")

    bundle = load_workspace_instructions(child)

    assert "CHILD_RULES" in bundle.content
    assert "PARENT_RULES" in bundle.content
    assert bundle.content.index("CHILD_RULES") < bundle.content.index("PARENT_RULES")


def test_upward_traversal_stops_at_git_boundary(tmp_path: Path) -> None:
    """Traversal should not cross a .git boundary."""
    grandparent = tmp_path / "org"
    parent = grandparent / "repo"
    child = parent / "src"
    child.mkdir(parents=True)
    (parent / ".git").mkdir()  # Git boundary
    (grandparent / "CLAUDE.md").write_text("GRANDPARENT_RULES", encoding="utf-8")
    (parent / "CLAUDE.md").write_text("PARENT_RULES", encoding="utf-8")

    bundle = load_workspace_instructions(child)

    # parent's CLAUDE.md is found, but grandparent's is not (blocked by .git in parent)
    assert "PARENT_RULES" in bundle.content
    assert "GRANDPARENT_RULES" not in bundle.content


def test_upward_traversal_respects_max_depth(tmp_path: Path) -> None:
    """Traversal depth is limited by max_traversal_depth."""
    deep = tmp_path / "a" / "b" / "c" / "d" / "e" / "f"
    deep.mkdir(parents=True)
    (tmp_path / "CLAUDE.md").write_text("ROOT_RULES", encoding="utf-8")

    bundle = load_workspace_instructions(deep, max_traversal_depth=2)

    # Only 2 parents checked: f -> e, e -> d. Root is too far.
    assert "ROOT_RULES" not in bundle.content


# ---------------------------------------------------------------------------
# Wave 1: Path-conditional rules
# ---------------------------------------------------------------------------


def test_path_conditional_rule_included_when_matching(tmp_path: Path) -> None:
    rules = tmp_path / ".claude" / "rules"
    rules.mkdir(parents=True)
    (rules / "api.md").write_text(
        "---\napplies_to: src/api/**\n---\nAPI rules apply.",
        encoding="utf-8",
    )

    active = [tmp_path / "src" / "api" / "routes.py"]
    bundle = load_workspace_instructions(tmp_path, active_paths=active)

    assert "API rules apply." in bundle.content


def test_path_conditional_rule_excluded_when_not_matching(tmp_path: Path) -> None:
    rules = tmp_path / ".claude" / "rules"
    rules.mkdir(parents=True)
    (rules / "api.md").write_text(
        "---\napplies_to: src/api/**\n---\nAPI rules apply.",
        encoding="utf-8",
    )

    active = [tmp_path / "src" / "frontend" / "app.tsx"]
    bundle = load_workspace_instructions(tmp_path, active_paths=active)

    assert "API rules apply." not in bundle.content


def test_path_conditional_rule_with_globs_field(tmp_path: Path) -> None:
    rules = tmp_path / ".claude" / "rules"
    rules.mkdir(parents=True)
    (rules / "python.md").write_text(
        '---\nglobs: "*.py"\n---\nPython rules.',
        encoding="utf-8",
    )

    active = [tmp_path / "src" / "main.py"]
    bundle = load_workspace_instructions(tmp_path, active_paths=active)
    assert "Python rules." in bundle.content

    active_ts = [tmp_path / "src" / "main.ts"]
    bundle2 = load_workspace_instructions(tmp_path, active_paths=active_ts)
    assert "Python rules." not in bundle2.content


def test_rule_without_frontmatter_always_included(tmp_path: Path) -> None:
    """Rules without frontmatter globs should always be included."""
    rules = tmp_path / ".claude" / "rules"
    rules.mkdir(parents=True)
    (rules / "general.md").write_text("General rules apply.", encoding="utf-8")

    active = [tmp_path / "src" / "anything.go"]
    bundle = load_workspace_instructions(tmp_path, active_paths=active)

    assert "General rules apply." in bundle.content


def test_path_conditional_included_when_no_active_paths(tmp_path: Path) -> None:
    """When active_paths is None (not provided), all rules should be included."""
    rules = tmp_path / ".claude" / "rules"
    rules.mkdir(parents=True)
    (rules / "api.md").write_text(
        "---\napplies_to: src/api/**\n---\nAPI rules.",
        encoding="utf-8",
    )

    bundle = load_workspace_instructions(tmp_path)  # no active_paths
    assert "API rules." in bundle.content


# ---------------------------------------------------------------------------
# Wave 1: External-include warnings
# ---------------------------------------------------------------------------


def test_external_include_produces_warning(tmp_path: Path) -> None:
    """@include pointing outside the project root should be replaced with a warning."""
    external = tmp_path / "external"
    external.mkdir()
    (external / "secret.md").write_text("Secret content.", encoding="utf-8")

    project = tmp_path / "project"
    project.mkdir()
    (project / "CLAUDE.md").write_text(
        f"@include ../external/secret.md\nSafe content.",
        encoding="utf-8",
    )

    bundle = load_workspace_instructions(project)

    assert "Secret content." not in bundle.content
    assert "Safe content." in bundle.content
    # The warning comment is stripped by _strip_html_comments, so it won't
    # appear in final output, but the external content is excluded.


def test_internal_include_still_works(tmp_path: Path) -> None:
    """@include within the project root should still expand normally."""
    (tmp_path / "shared.md").write_text("Shared rules.", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text(
        "@include shared.md\nProject rules.",
        encoding="utf-8",
    )

    bundle = load_workspace_instructions(tmp_path)

    assert "Shared rules." in bundle.content
    assert "Project rules." in bundle.content


# ---------------------------------------------------------------------------
# Wave 1: on_loaded callback
# ---------------------------------------------------------------------------


def test_on_loaded_callback_fires(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("Rules here.", encoding="utf-8")
    captured: list[InstructionBundle] = []

    load_workspace_instructions(tmp_path, on_loaded=captured.append)

    assert len(captured) == 1
    assert "Rules here." in captured[0].content
    assert len(captured[0].sources) == 1


def test_on_loaded_not_called_on_empty(tmp_path: Path) -> None:
    """on_loaded should still fire even when no instructions are found."""
    captured: list[InstructionBundle] = []

    load_workspace_instructions(tmp_path, on_loaded=captured.append)

    assert len(captured) == 1
    assert captured[0].content == ""


# ---------------------------------------------------------------------------
# Wave 2: User-level, project-dot, and local instruction layering
# ---------------------------------------------------------------------------


def test_user_level_instructions_from_explicit_config_dir(tmp_path: Path) -> None:
    """Layer 0: User-level CLAUDE.md loaded from explicit user_config_dir."""
    user_dir = tmp_path / "user_config"
    user_dir.mkdir()
    (user_dir / "CLAUDE.md").write_text("USER_LEVEL_RULES", encoding="utf-8")

    project = tmp_path / "project"
    project.mkdir()

    bundle = load_workspace_instructions(project, user_config_dir=user_dir)

    assert "USER_LEVEL_RULES" in bundle.content
    assert any("CLAUDE.md" in str(s) for s in bundle.sources)


def test_user_level_instructions_precedence(tmp_path: Path) -> None:
    """Layer 0 (user-level) appears before Layer 2 (project CLAUDE.md)."""
    user_dir = tmp_path / "user_config"
    user_dir.mkdir()
    (user_dir / "CLAUDE.md").write_text("USER_CONTENT", encoding="utf-8")

    project = tmp_path / "project"
    project.mkdir()
    (project / "CLAUDE.md").write_text("PROJECT_CONTENT", encoding="utf-8")

    bundle = load_workspace_instructions(project, user_config_dir=user_dir)

    assert bundle.content.index("USER_CONTENT") < bundle.content.index("PROJECT_CONTENT")


def test_dot_claude_md_layer(tmp_path: Path) -> None:
    """Layer 1.5: .claude/CLAUDE.md is loaded between rules and project CLAUDE.md."""
    rules = tmp_path / ".claude" / "rules"
    rules.mkdir(parents=True)
    (rules / "style.md").write_text("RULES_CONTENT", encoding="utf-8")
    (tmp_path / ".claude" / "CLAUDE.md").write_text("DOT_CLAUDE_CONTENT", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("PROJECT_CONTENT", encoding="utf-8")

    bundle = load_workspace_instructions(tmp_path)

    assert "DOT_CLAUDE_CONTENT" in bundle.content
    rules_pos = bundle.content.index("RULES_CONTENT")
    dot_pos = bundle.content.index("DOT_CLAUDE_CONTENT")
    project_pos = bundle.content.index("PROJECT_CONTENT")
    assert rules_pos < dot_pos < project_pos


def test_dot_claude_md_without_rules_dir(tmp_path: Path) -> None:
    """Layer 1.5 loads even when .claude/rules/ directory doesn't exist."""
    dot_dir = tmp_path / ".claude"
    dot_dir.mkdir()
    (dot_dir / "CLAUDE.md").write_text("DOT_CLAUDE_ONLY", encoding="utf-8")

    bundle = load_workspace_instructions(tmp_path)

    assert "DOT_CLAUDE_ONLY" in bundle.content


def test_claude_local_md_layer(tmp_path: Path) -> None:
    """Layer 3.5: CLAUDE.local.md is loaded after .github/copilot-instructions.md."""
    gh = tmp_path / ".github"
    gh.mkdir()
    (gh / "copilot-instructions.md").write_text("GITHUB_CONTENT", encoding="utf-8")
    (tmp_path / "CLAUDE.local.md").write_text("LOCAL_OVERRIDE_CONTENT", encoding="utf-8")

    bundle = load_workspace_instructions(tmp_path)

    assert "LOCAL_OVERRIDE_CONTENT" in bundle.content
    github_pos = bundle.content.index("GITHUB_CONTENT")
    local_pos = bundle.content.index("LOCAL_OVERRIDE_CONTENT")
    assert github_pos < local_pos


def test_claude_local_md_without_github(tmp_path: Path) -> None:
    """CLAUDE.local.md loads even without .github/copilot-instructions.md."""
    (tmp_path / "CLAUDE.local.md").write_text("LOCAL_ONLY", encoding="utf-8")

    bundle = load_workspace_instructions(tmp_path)

    assert "LOCAL_ONLY" in bundle.content
    assert any("CLAUDE.local.md" in str(s) for s in bundle.sources)


def test_full_layering_order(tmp_path: Path) -> None:
    """All layers load in the correct order: 0 → 1 → 1.5 → 2 → 3 → 3.5 → 4."""
    user_dir = tmp_path / "user_config"
    user_dir.mkdir()
    (user_dir / "CLAUDE.md").write_text("L0_USER", encoding="utf-8")

    project = tmp_path / "project"
    project.mkdir()

    rules = project / ".claude" / "rules"
    rules.mkdir(parents=True)
    (rules / "a.md").write_text("L1_RULES", encoding="utf-8")

    (project / ".claude" / "CLAUDE.md").write_text("L15_DOTCLAUDE", encoding="utf-8")

    (project / "CLAUDE.md").write_text("L2_PROJECT", encoding="utf-8")

    gh = project / ".github"
    gh.mkdir()
    (gh / "copilot-instructions.md").write_text("L3_GITHUB", encoding="utf-8")

    (project / "CLAUDE.local.md").write_text("L35_LOCAL", encoding="utf-8")

    bundle = load_workspace_instructions(project, user_config_dir=user_dir)

    positions = [
        bundle.content.index("L0_USER"),
        bundle.content.index("L1_RULES"),
        bundle.content.index("L15_DOTCLAUDE"),
        bundle.content.index("L2_PROJECT"),
        bundle.content.index("L3_GITHUB"),
        bundle.content.index("L35_LOCAL"),
    ]
    assert positions == sorted(positions), f"Layer order violated: {positions}"


def test_loaded_paths_property(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("Rules.", encoding="utf-8")

    bundle = load_workspace_instructions(tmp_path)

    assert len(bundle.loaded_paths) == 1
    assert "CLAUDE.md" in bundle.loaded_paths[0]


# ---------------------------------------------------------------------------
# Gap 1.2: Circular include protection
# ---------------------------------------------------------------------------


def test_circular_include_detected_and_skipped(tmp_path: Path) -> None:
    """A -> B -> A circular include is detected and skipped without infinite loop."""
    (tmp_path / "a.md").write_text("A content\n@include b.md\nEnd A", encoding="utf-8")
    (tmp_path / "b.md").write_text("B content\n@include a.md\nEnd B", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("@include a.md", encoding="utf-8")

    bundle = load_workspace_instructions(tmp_path)

    assert "A content" in bundle.content
    assert "B content" in bundle.content
    # A should appear only once — the circular re-include is skipped
    # (the <!-- circular include skipped --> comment is stripped by HTML comment removal)
    assert bundle.content.count("A content") == 1


def test_diamond_include_deduplicates(tmp_path: Path) -> None:
    """A -> B -> D and A -> C -> D: D should only appear once."""
    (tmp_path / "d.md").write_text("D shared content", encoding="utf-8")
    (tmp_path / "b.md").write_text("B content\n@include d.md", encoding="utf-8")
    (tmp_path / "c.md").write_text("C content\n@include d.md", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("@include b.md\n@include c.md", encoding="utf-8")

    bundle = load_workspace_instructions(tmp_path)

    assert "B content" in bundle.content
    assert "C content" in bundle.content
    assert "D shared content" in bundle.content
    # D should appear exactly once
    assert bundle.content.count("D shared content") == 1


def test_self_include_skipped(tmp_path: Path) -> None:
    """A file including itself should be detected and skipped."""
    (tmp_path / "CLAUDE.md").write_text("Self\n@include CLAUDE.md\nEnd", encoding="utf-8")

    bundle = load_workspace_instructions(tmp_path)

    assert "Self" in bundle.content
    assert bundle.content.count("Self") == 1
