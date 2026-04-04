from __future__ import annotations

from pathlib import Path

import pytest

from copilotcode_sdk import (
    BrandSpec,
    CopilotCodeConfig,
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
