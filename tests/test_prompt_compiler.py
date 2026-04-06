from __future__ import annotations

from pathlib import Path

import pytest

from copilotcode_sdk import (
    BrandSpec,
    CopilotCodeConfig,
    PromptAssembler,
    PromptPriority,
    PromptSection,
    build_assembler,
    build_system_message,
    materialize_workspace_instructions,
    render_claude_md_template,
    render_copilot_instructions_template,
)


def test_build_system_message_includes_expected_sections(tmp_path: Path) -> None:
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
    )

    message = build_system_message(config)

    assert "# CopilotCode" in message
    assert "## Core Operating Rules" in message
    assert "## Tool Usage Rules" in message
    assert "# auto memory" in message
    assert "prompt-injection" in message


def test_build_system_message_can_disable_memory(tmp_path: Path) -> None:
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        enable_hybrid_memory=False,
    )

    message = build_system_message(config)

    assert "# auto memory" not in message


def test_templates_reference_copilotcode_branding(tmp_path: Path) -> None:
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
    )

    assert "CopilotCode" in render_claude_md_template(config)
    assert "CopilotCode" in render_copilot_instructions_template(config)
    assert "~/.copilotcode/projects/<project>/memory" in render_claude_md_template(config)


def test_brand_override_updates_template_output(tmp_path: Path) -> None:
    brand = BrandSpec(
        public_name="FutureName",
        slug="futurename",
        package_name="futurename_sdk",
        distribution_name="futurename-sdk",
        cli_name="futurename",
        app_dirname=".futurename",
    )
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        brand=brand,
    )

    message = build_system_message(config)

    assert "# FutureName" in message
    assert "~/.futurename/projects/<project>/memory" in message


def test_materialize_workspace_instructions_round_trips_templates(
    tmp_path: Path,
) -> None:
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
    )

    claude_path, copilot_path = materialize_workspace_instructions(
        tmp_path,
        config,
    )

    assert claude_path.read_text(encoding="utf-8") == render_claude_md_template(config)
    assert copilot_path.read_text(encoding="utf-8") == render_copilot_instructions_template(
        config,
    )


def test_materialize_workspace_instructions_refuses_to_overwrite(
    tmp_path: Path,
) -> None:
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
    )
    materialize_workspace_instructions(tmp_path, config)

    with pytest.raises(FileExistsError):
        materialize_workspace_instructions(tmp_path, config)


def test_build_system_message_includes_output_efficiency(tmp_path: Path) -> None:
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
    )
    message = build_system_message(config)
    assert "## Output Efficiency" in message
    assert "Go straight to the point" in message


def test_build_system_message_includes_actions_with_care(tmp_path: Path) -> None:
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
    )
    message = build_system_message(config)
    assert "## Actions With Care" in message


def test_build_system_message_includes_skill_catalog_when_dirs_provided(
    tmp_path: Path,
) -> None:
    skill_dir = tmp_path / "skills" / "my-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: Test skill.\ntype: test-type\nrequires: none\n---\n\n# My Skill\n",
        encoding="utf-8",
    )

    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
    )
    message = build_system_message(config, skill_directories=[str(tmp_path / "skills")])

    assert "## Available Skills" in message
    assert "my-skill" in message
    assert "test-type" in message


def test_build_system_message_omits_skill_catalog_when_no_dirs(tmp_path: Path) -> None:
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
    )
    message = build_system_message(config)
    assert "## Available Skills" not in message


def test_build_system_message_includes_skill_chaining_guidance(tmp_path: Path) -> None:
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
    )
    message = build_system_message(config)
    assert "check available skills" in message.lower() or "downstream skills" in message.lower()


def test_build_system_message_section_order(tmp_path: Path) -> None:
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
    )
    message = build_system_message(config)
    sections = [
        "# CopilotCode",
        "## Core Operating Rules",
        "## Tool Usage Rules",
        "## Output Efficiency",
        "## Actions With Care",
        "## Tone And Output",
        "## Session Guidance",
        "# auto memory",
    ]
    positions = [message.index(s) for s in sections]
    assert positions == sorted(positions), f"Sections out of order: {positions}"


def test_build_system_message_memory_has_examples_and_template(tmp_path: Path) -> None:
    """Memory guidance must include concrete examples and the frontmatter template."""
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
    )
    message = build_system_message(config)

    # Concrete examples showing user→assistant save pattern
    assert "[saves" in message
    # Frontmatter template with --- fences
    assert "---\nname:" in message
    # Two-step save process
    assert "Step 1" in message
    assert "Step 2" in message
    # MEMORY.md format guidance
    assert "~150 characters" in message
    # All four memory types
    for t in ("### user", "### feedback", "### project", "### reference"):
        assert t in message
    # Body structure guidance for feedback/project
    assert "**Why:**" in message
    assert "**How to apply:**" in message


# ---------------------------------------------------------------------------
# Wave 2: PromptAssembler tests
# ---------------------------------------------------------------------------


class TestPromptAssembler:
    def test_add_and_render_preserves_insertion_order(self) -> None:
        asm = PromptAssembler()
        asm.add("a", "Alpha content")
        asm.add("b", "Beta content")
        asm.add("c", "Charlie content")

        result = asm.render()

        assert result.index("Alpha") < result.index("Beta") < result.index("Charlie")

    def test_higher_priority_replaces_lower(self) -> None:
        asm = PromptAssembler()
        asm.add("rules", "Default rules", priority=PromptPriority.default)
        asm.add("rules", "Agent rules", priority=PromptPriority.agent)

        result = asm.render()

        assert "Agent rules" in result
        assert "Default rules" not in result

    def test_lower_priority_does_not_replace_higher(self) -> None:
        asm = PromptAssembler()
        asm.add("rules", "Override rules", priority=PromptPriority.override)
        asm.add("rules", "Default rules", priority=PromptPriority.default)

        result = asm.render()

        assert "Override rules" in result
        assert "Default rules" not in result

    def test_equal_priority_replaces(self) -> None:
        asm = PromptAssembler()
        asm.add("rules", "First version", priority=PromptPriority.default)
        asm.add("rules", "Second version", priority=PromptPriority.default)

        assert "Second version" in asm.render()
        assert "First version" not in asm.render()

    def test_remove_section(self) -> None:
        asm = PromptAssembler()
        asm.add("keep", "Keep this")
        asm.add("drop", "Drop this")
        asm.remove("drop")

        result = asm.render()

        assert "Keep this" in result
        assert "Drop this" not in result

    def test_remove_nonexistent_is_noop(self) -> None:
        asm = PromptAssembler()
        asm.add("a", "Content")
        asm.remove("nonexistent")  # should not raise

        assert "Content" in asm.render()

    def test_has_section(self) -> None:
        asm = PromptAssembler()
        asm.add("present", "Yes")

        assert asm.has("present") is True
        assert asm.has("absent") is False

    def test_cacheable_only_excludes_non_cacheable(self) -> None:
        asm = PromptAssembler()
        asm.add("static", "Static content", cacheable=True)
        asm.add("dynamic", "Dynamic content", cacheable=False)

        cacheable = asm.render(cacheable_only=True)

        assert "Static content" in cacheable
        assert "Dynamic content" not in cacheable

    def test_render_dynamic_excludes_cacheable(self) -> None:
        asm = PromptAssembler()
        asm.add("static", "Static content", cacheable=True)
        asm.add("dynamic", "Dynamic content", cacheable=False)

        dynamic = asm.render_dynamic()

        assert "Dynamic content" in dynamic
        assert "Static content" not in dynamic

    def test_section_names_returns_insertion_order(self) -> None:
        asm = PromptAssembler()
        asm.add("z", "Z")
        asm.add("a", "A")
        asm.add("m", "M")

        assert asm.section_names == ["z", "a", "m"]

    def test_empty_sections_are_skipped(self) -> None:
        asm = PromptAssembler()
        asm.add("nonempty", "Real content")
        asm.add("empty", "   ")

        result = asm.render()

        assert "Real content" in result
        assert result.strip() == "Real content"

    def test_add_base_sections_populates_expected_names(self, tmp_path: Path) -> None:
        config = CopilotCodeConfig(
            working_directory=tmp_path,
            memory_root=tmp_path / ".mem",
        )
        asm = PromptAssembler()
        asm.add_base_sections(config)

        expected = [
            "intro", "core_rules", "tool_rules", "tool_caveats",
            "output_efficiency", "actions_with_care", "tone_output",
            "session_guidance",
        ]
        for name in expected:
            assert asm.has(name), f"Missing base section: {name}"

    def test_priority_enum_ordering(self) -> None:
        assert PromptPriority.base < PromptPriority.default
        assert PromptPriority.default < PromptPriority.agent
        assert PromptPriority.agent < PromptPriority.skill
        assert PromptPriority.skill < PromptPriority.custom
        assert PromptPriority.custom < PromptPriority.override


class TestBuildAssembler:
    def test_returns_assembler_with_base_sections(self, tmp_path: Path) -> None:
        config = CopilotCodeConfig(
            working_directory=tmp_path,
            memory_root=tmp_path / ".mem",
        )
        asm = build_assembler(config)

        assert isinstance(asm, PromptAssembler)
        assert asm.has("intro")
        assert asm.has("core_rules")

    def test_includes_memory_guidance_by_default(self, tmp_path: Path) -> None:
        config = CopilotCodeConfig(
            working_directory=tmp_path,
            memory_root=tmp_path / ".mem",
        )
        asm = build_assembler(config)

        assert asm.has("memory_guidance")

    def test_excludes_memory_when_disabled(self, tmp_path: Path) -> None:
        config = CopilotCodeConfig(
            working_directory=tmp_path,
            memory_root=tmp_path / ".mem",
            enable_hybrid_memory=False,
        )
        asm = build_assembler(config)

        assert not asm.has("memory_guidance")

    def test_includes_task_management_when_enabled(self, tmp_path: Path) -> None:
        config = CopilotCodeConfig(
            working_directory=tmp_path,
            memory_root=tmp_path / ".mem",
            enable_tasks_v2=True,
        )
        asm = build_assembler(config)

        assert asm.has("task_management")

    def test_skill_catalog_is_non_cacheable(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "skills" / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: A skill.\ntype: t\nrequires: none\n---\n\n# Test\n",
            encoding="utf-8",
        )
        config = CopilotCodeConfig(
            working_directory=tmp_path,
            memory_root=tmp_path / ".mem",
        )
        asm = build_assembler(config, skill_directories=[str(tmp_path / "skills")])

        assert asm.has("skill_catalog")
        # Skill catalog should be in dynamic render, not cacheable-only
        assert "test-skill" in asm.render_dynamic()
        assert "test-skill" not in asm.render(cacheable_only=True)

    def test_extra_prompt_sections_added_at_custom_priority(self, tmp_path: Path) -> None:
        config = CopilotCodeConfig(
            working_directory=tmp_path,
            memory_root=tmp_path / ".mem",
            extra_prompt_sections=["Custom instruction A", "Custom instruction B"],
        )
        asm = build_assembler(config)

        assert asm.has("extra_0")
        assert asm.has("extra_1")
        rendered = asm.render()
        assert "Custom instruction A" in rendered
        assert "Custom instruction B" in rendered

    def test_assembler_render_matches_build_system_message(self, tmp_path: Path) -> None:
        """build_system_message() should produce the same output as build_assembler().render()."""
        config = CopilotCodeConfig(
            working_directory=tmp_path,
            memory_root=tmp_path / ".mem",
        )

        via_message = build_system_message(config)
        via_assembler = build_assembler(config).render()

        assert via_message == via_assembler

    def test_mcp_servers_added_as_non_cacheable(self, tmp_path: Path) -> None:
        config = CopilotCodeConfig(
            working_directory=tmp_path,
            memory_root=tmp_path / ".mem",
            mcp_servers=[
                {"name": "TestServer", "description": "A test MCP server.",
                 "tools": [{"name": "test_tool", "description": "Does testing."}]},
            ],
        )
        asm = build_assembler(config)

        assert asm.has("mcp_servers")
        assert "TestServer" in asm.render_dynamic()
        assert "TestServer" not in asm.render(cacheable_only=True)


# ---------------------------------------------------------------------------
# Wave 5: Git context, cyber risk, dynamic boundary
# ---------------------------------------------------------------------------


class TestCyberRiskInstruction:
    def test_present_in_base_sections(self, tmp_path: Path) -> None:
        config = CopilotCodeConfig(working_directory=tmp_path, memory_root=tmp_path / ".m")
        asm = PromptAssembler()
        asm.add_base_sections(config)
        rendered = asm.render()
        assert "authorized security testing" in rendered
        assert "Refuse requests for destructive" in rendered

    def test_is_cacheable(self, tmp_path: Path) -> None:
        config = CopilotCodeConfig(working_directory=tmp_path, memory_root=tmp_path / ".m")
        asm = PromptAssembler()
        asm.add_base_sections(config)
        cacheable = asm.render(cacheable_only=True)
        assert "authorized security testing" in cacheable


class TestDynamicBoundary:
    def test_boundary_in_base_sections(self, tmp_path: Path) -> None:
        from copilotcode_sdk.prompt_compiler import SYSTEM_PROMPT_DYNAMIC_BOUNDARY
        config = CopilotCodeConfig(working_directory=tmp_path, memory_root=tmp_path / ".m")
        asm = PromptAssembler()
        asm.add_base_sections(config)
        rendered = asm.render()
        assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY in rendered

    def test_boundary_is_cacheable(self, tmp_path: Path) -> None:
        from copilotcode_sdk.prompt_compiler import SYSTEM_PROMPT_DYNAMIC_BOUNDARY
        config = CopilotCodeConfig(working_directory=tmp_path, memory_root=tmp_path / ".m")
        asm = PromptAssembler()
        asm.add_base_sections(config)
        cacheable = asm.render(cacheable_only=True)
        assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY in cacheable

    def test_boundary_separates_static_from_dynamic(self, tmp_path: Path) -> None:
        from copilotcode_sdk.prompt_compiler import SYSTEM_PROMPT_DYNAMIC_BOUNDARY
        config = CopilotCodeConfig(working_directory=tmp_path, memory_root=tmp_path / ".m")
        asm = build_assembler(config)
        rendered = asm.render()
        # Boundary should exist
        assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY in rendered
        # Core rules should come before boundary
        boundary_pos = rendered.index(SYSTEM_PROMPT_DYNAMIC_BOUNDARY)
        assert rendered.index("Core Operating Rules") < boundary_pos


class TestGitContext:
    def test_gather_git_context_in_repo(self, tmp_path: Path) -> None:
        from copilotcode_sdk.prompt_compiler import gather_git_context
        # Use the actual repo we're in
        repo_root = Path(__file__).resolve().parent.parent
        ctx = gather_git_context(repo_root)
        assert "gitStatus" in ctx
        assert "Current branch:" in ctx

    def test_gather_git_context_non_repo(self, tmp_path: Path) -> None:
        from copilotcode_sdk.prompt_compiler import gather_git_context
        # Create an isolated dir that's definitely not in a git repo
        isolated = tmp_path / "no_git"
        isolated.mkdir()
        # Create a .git-less directory tree (git init would make it a repo)
        # The safest way: mock subprocess to return failure
        import unittest.mock
        with unittest.mock.patch("copilotcode_sdk.prompt_compiler.subprocess.run") as mock_run:
            mock_run.return_value = unittest.mock.MagicMock(returncode=128, stdout="")
            ctx = gather_git_context(isolated)
        assert ctx == ""

    def test_git_context_added_as_non_cacheable(self) -> None:
        from copilotcode_sdk.prompt_compiler import gather_git_context
        # Use the actual repo
        repo_root = Path(__file__).resolve().parent.parent
        config = CopilotCodeConfig(
            working_directory=repo_root,
            memory_root=repo_root / ".test_mem",
        )
        asm = build_assembler(config)
        if asm.has("git_context"):
            # Git context should be in dynamic, not cacheable
            assert "gitStatus" in asm.render_dynamic()
            assert "gitStatus" not in asm.render(cacheable_only=True)


# ---------------------------------------------------------------------------
# Stale section tracking
# ---------------------------------------------------------------------------


class TestStaleTracking:
    def test_mark_stale_and_has_stale(self) -> None:
        asm = PromptAssembler()
        asm.add("git_context", "branch: main", cacheable=False)
        assert asm.has_stale_sections() is False
        asm.mark_stale("git_context")
        assert asm.has_stale_sections() is True

    def test_refresh_clears_stale(self) -> None:
        asm = PromptAssembler()
        asm.add("git_context", "branch: main", cacheable=False)
        asm.mark_stale("git_context")
        asm.refresh_section("git_context", "branch: develop")
        assert asm.has_stale_sections() is False
        assert "develop" in asm.render_dynamic()
        assert "main" not in asm.render_dynamic()

    def test_add_clears_stale(self) -> None:
        asm = PromptAssembler()
        asm.add("git_context", "old", cacheable=False)
        asm.mark_stale("git_context")
        asm.add("git_context", "new", cacheable=False)
        assert asm.has_stale_sections() is False
