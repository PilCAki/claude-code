from __future__ import annotations

from .agents import build_default_custom_agents
from .brand import BrandSpec, DEFAULT_BRAND
from .client import CopilotCodeClient, CopilotCodeSession
from .config import CopilotCodeConfig, DEFAULT_AGENT_NAMES, DEFAULT_SKILL_NAMES
from .memory import MemoryStore
from .permissions import PermissionPolicy
from .prompt_compiler import (
    build_system_message,
    materialize_workspace_instructions,
    render_claude_md_template,
    render_copilot_instructions_template,
)
from .extraction import build_extraction_prompt, should_extract
from .compaction import build_compaction_prompt, build_handoff_context
from .instructions import InstructionBundle, load_workspace_instructions
from .reports import CheckResult, PreflightReport, SmokeTestReport
from .skill_assets import build_skill_catalog, parse_skill_frontmatter

__all__ = [
    "BrandSpec",
    "CheckResult",
    "CopilotCodeClient",
    "CopilotCodeConfig",
    "CopilotCodeSession",
    "DEFAULT_AGENT_NAMES",
    "DEFAULT_BRAND",
    "DEFAULT_SKILL_NAMES",
    "InstructionBundle",
    "MemoryStore",
    "PermissionPolicy",
    "PreflightReport",
    "SmokeTestReport",
    "build_compaction_prompt",
    "build_default_custom_agents",
    "build_extraction_prompt",
    "build_handoff_context",
    "build_skill_catalog",
    "build_system_message",
    "load_workspace_instructions",
    "materialize_workspace_instructions",
    "parse_skill_frontmatter",
    "render_claude_md_template",
    "render_copilot_instructions_template",
    "should_extract",
]

__version__ = "0.2.0"
