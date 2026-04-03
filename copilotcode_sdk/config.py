from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from .brand import BrandSpec, DEFAULT_BRAND
from .permissions import (
    DEFAULT_APPROVED_SHELL_PREFIXES,
    PermissionPolicy,
    normalize_shell_prefixes,
)

ReasoningEffort = Literal["low", "medium", "high", "xhigh"]

DEFAULT_SKILL_NAMES: tuple[str, ...] = (
    "verify",
    "remember",
    "simplify",
    "debug",
    "batch",
    "skillify",
)

DEFAULT_AGENT_NAMES: tuple[str, ...] = (
    "researcher",
    "planner",
    "implementer",
    "verifier",
)

DEFAULT_BACKGROUND_COMPACTION_THRESHOLD = 0.80
DEFAULT_BUFFER_EXHAUSTION_THRESHOLD = 0.95


def _resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _unique_names(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in values:
        name = raw.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return tuple(ordered)


@dataclass(slots=True)
class CopilotCodeConfig:
    """Configuration for the CopilotCode wrapper around the Copilot SDK."""

    working_directory: str | Path = "."
    model: str | None = None
    reasoning_effort: ReasoningEffort | None = None
    provider: Mapping[str, Any] | None = None
    brand: BrandSpec = DEFAULT_BRAND
    cli_path: str | None = None
    cli_args: Sequence[str] = field(default_factory=tuple)
    cli_env: Mapping[str, str] | None = None
    cli_use_stdio: bool = True
    cli_port: int = 0
    cli_log_level: str = "info"
    github_token: str | None = None
    use_logged_in_user: bool | None = None
    client_name: str | None = None
    config_dir: str | Path | None = None
    copilot_config_dir: str | Path | None = None
    memory_root: str | Path | None = None
    enable_hybrid_memory: bool = True
    enabled_skills: Sequence[str] = field(
        default_factory=lambda: DEFAULT_SKILL_NAMES,
    )
    disabled_skills: Sequence[str] = field(default_factory=tuple)
    extra_skill_directories: Sequence[str | Path] = field(default_factory=tuple)
    enabled_agents: Sequence[str] = field(
        default_factory=lambda: DEFAULT_AGENT_NAMES,
    )
    extra_agents: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    default_agent: str | None = None
    infinite_sessions: bool | Mapping[str, Any] = True
    shell_timeout_ms: int = 120_000
    noisy_tool_char_limit: int = 8_000
    reminder_reinjection_interval: int = 15
    extraction_tool_call_interval: int = 20
    extraction_char_threshold: int = 50_000
    extraction_min_turn_gap: int = 10
    path_allowlist: Sequence[str | Path] = field(default_factory=tuple)
    permission_policy: PermissionPolicy = "safe"
    approved_shell_prefixes: Sequence[str] = field(
        default_factory=lambda: DEFAULT_APPROVED_SHELL_PREFIXES,
    )
    permission_handler: Callable[..., Any] | None = None
    user_input_handler: Callable[..., Any] | None = None
    on_event: Callable[[Any], None] | None = None
    include_workspace_instruction_snippets: bool = True
    enable_skill_shorthand: bool = True
    extra_prompt_sections: Sequence[str] = field(default_factory=tuple)
    enable_tasks_v2: bool = True
    task_root: str | Path | None = None
    task_reminder_turns: int = 10
    task_reminder_cooldown_turns: int = 10

    def __post_init__(self) -> None:
        self.working_directory = _resolve_path(self.working_directory)
        self.memory_root = _resolve_path(self.memory_root or self.brand.memory_home())
        self.config_dir = _resolve_path(self.config_dir or self.brand.app_config_home())
        self.copilot_config_dir = (
            _resolve_path(self.copilot_config_dir)
            if self.copilot_config_dir is not None
            else None
        )
        self.client_name = self.client_name or self.brand.client_name
        if self.task_root is not None:
            self.task_root = _resolve_path(self.task_root)
        self.extra_skill_directories = tuple(
            str(_resolve_path(path))
            for path in self.extra_skill_directories
        )
        self.path_allowlist = tuple(
            _resolve_path(path)
            for path in self.path_allowlist
        )
        self.enabled_skills = _unique_names(self.enabled_skills)
        self.disabled_skills = _unique_names(self.disabled_skills)
        self.enabled_agents = _unique_names(self.enabled_agents)
        self.cli_args = tuple(self.cli_args)
        self.approved_shell_prefixes = normalize_shell_prefixes(
            self.approved_shell_prefixes,
        )
        self.extra_prompt_sections = tuple(
            section.strip()
            for section in self.extra_prompt_sections
            if section.strip()
        )
        if self.shell_timeout_ms < 1:
            raise ValueError("shell_timeout_ms must be positive")
        if self.noisy_tool_char_limit < 128:
            raise ValueError("noisy_tool_char_limit must be at least 128")
        if self.permission_policy == "custom" and self.permission_handler is None:
            raise ValueError(
                "permission_policy='custom' requires a permission_handler.",
            )

    @property
    def working_path(self) -> Path:
        return Path(self.working_directory)

    @property
    def memory_home(self) -> Path:
        return Path(self.memory_root)

    @property
    def app_config_home(self) -> Path:
        return Path(self.config_dir)

    @property
    def copilot_config_home(self) -> Path | None:
        return (
            Path(self.copilot_config_dir)
            if self.copilot_config_dir is not None
            else None
        )

    @property
    def allowed_roots(self) -> tuple[Path, ...]:
        return (self.working_path, self.app_config_home, *self.path_allowlist)

    def resolved_infinite_session_config(self) -> dict[str, Any]:
        if self.infinite_sessions is False:
            return {"enabled": False}

        defaults = {
            "enabled": True,
            "background_compaction_threshold": DEFAULT_BACKGROUND_COMPACTION_THRESHOLD,
            "buffer_exhaustion_threshold": DEFAULT_BUFFER_EXHAUSTION_THRESHOLD,
        }
        if self.infinite_sessions is True:
            return defaults

        merged = dict(defaults)
        merged.update(dict(self.infinite_sessions))
        return merged
