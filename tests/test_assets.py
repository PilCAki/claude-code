from __future__ import annotations

from pathlib import Path

import copilotcode_sdk
from copilotcode_sdk import BrandSpec, CopilotCodeConfig, DEFAULT_BRAND, build_system_message
from copilotcode_sdk.prompt_compiler import (
    render_claude_md_template,
    render_copilot_instructions_template,
)
from copilotcode_sdk.skill_assets import iter_skill_documents


PACKAGE_ROOT = Path(copilotcode_sdk.__file__).resolve().parent


def test_skill_asset_files_match_python_owned_source() -> None:
    for skill_name, skill_document in iter_skill_documents(DEFAULT_BRAND):
        skill_path = PACKAGE_ROOT / "skills" / skill_name / "SKILL.md"
        assert skill_path.read_text(encoding="utf-8") == skill_document


def test_template_files_match_rendered_output(tmp_path: Path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path, memory_root=tmp_path / ".mem")
    claude_template = PACKAGE_ROOT / "templates" / "CLAUDE.md"
    copilot_template = PACKAGE_ROOT / "templates" / "copilot-instructions.md"

    assert claude_template.read_text(encoding="utf-8") == render_claude_md_template(config)
    assert (
        copilot_template.read_text(encoding="utf-8")
        == render_copilot_instructions_template(config)
    )


def test_brand_spec_drives_prompts_and_templates(tmp_path: Path) -> None:
    brand = BrandSpec(
        public_name="RenamedAgent",
        slug="renamedagent",
        package_name="renamedagent_sdk",
        distribution_name="renamedagent-sdk",
        cli_name="renamedagent",
        app_dirname=".renamedagent",
    )
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        brand=brand,
    )

    message = build_system_message(config)
    claude_template = render_claude_md_template(config)

    assert "# RenamedAgent" in message
    assert "RenamedAgent" in claude_template
    assert "~/.renamedagent/projects/<project>/memory" in message
