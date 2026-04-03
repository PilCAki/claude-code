from __future__ import annotations

from pathlib import Path

from copilotcode_sdk.config import CopilotCodeConfig
from copilotcode_sdk.hooks import build_default_hooks
from copilotcode_sdk.memory import MemoryStore


def _build_hooks(tmp_path: Path) -> tuple[CopilotCodeConfig, MemoryStore, dict[str, object]]:
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    store.upsert_memory(
        title="Verify Flow",
        description="Auth changes should run pytest -q and hit the login endpoint.",
        memory_type="project",
        content="# Verify\nRun pytest -q and test the login flow directly.",
    )
    return config, store, build_default_hooks(config, store)


def test_session_start_hook_includes_workspace_instructions(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("Repo instructions go here.", encoding="utf-8")
    _, _, hooks = _build_hooks(tmp_path)

    result = hooks["on_session_start"]({}, {})  # type: ignore[index]

    assert "CopilotCode is active" in result["additionalContext"]
    assert "Repo instructions go here." in result["additionalContext"]
    assert "Durable Memory Index" in result["additionalContext"]


def test_user_prompt_hook_expands_skill_and_adds_memory_context(
    tmp_path: Path,
) -> None:
    _, _, hooks = _build_hooks(tmp_path)

    result = hooks["on_user_prompt_submitted"](  # type: ignore[index]
        {"prompt": "/verify auth login"},
        {},
    )

    assert "Use the `verify` skill" in result["modifiedPrompt"]
    assert "Relevant Durable Memory" in result["additionalContext"]


def test_pre_tool_use_hook_denies_paths_outside_allowed_roots(
    tmp_path: Path,
) -> None:
    _, _, hooks = _build_hooks(tmp_path)

    result = hooks["on_pre_tool_use"](  # type: ignore[index]
        {
            "toolName": "edit",
            "toolArgs": {"filePath": str(tmp_path.parent / "outside.txt")},
        },
        {},
    )

    assert result["permissionDecision"] == "deny"
    assert "outside the workspace" in result["permissionDecisionReason"]


def test_pre_tool_use_hook_injects_shell_timeout(tmp_path: Path) -> None:
    config, _, hooks = _build_hooks(tmp_path)

    result = hooks["on_pre_tool_use"](  # type: ignore[index]
        {
            "toolName": "bash",
            "toolArgs": {"command": "pytest -q"},
        },
        {},
    )

    assert result["modifiedArgs"]["timeout_ms"] == config.shell_timeout_ms


def test_post_tool_use_hook_truncates_noisy_results(tmp_path: Path) -> None:
    config, _, hooks = _build_hooks(tmp_path)
    noisy_payload = {"items": ["x" * (config.noisy_tool_char_limit + 100)]}

    result = hooks["on_post_tool_use"](  # type: ignore[index]
        {
            "toolName": "grep",
            "toolResult": noisy_payload,
        },
        {},
    )

    assert result["modifiedResult"]["truncated"] is True
    assert "most relevant lines" in result["additionalContext"]


def test_session_start_hook_can_skip_workspace_instructions_and_memory(tmp_path: Path) -> None:
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        include_workspace_instruction_snippets=False,
        enable_hybrid_memory=False,
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)

    result = hooks["on_session_start"]({}, {})  # type: ignore[index]

    assert "CopilotCode is active" in result["additionalContext"]
    assert "Workspace Instructions" not in result["additionalContext"]
    assert "Durable Memory Index" not in result["additionalContext"]


def test_user_prompt_hook_respects_disabled_shorthand(tmp_path: Path) -> None:
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        enable_skill_shorthand=False,
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)

    result = hooks["on_user_prompt_submitted"]({"prompt": "/verify auth login"}, {})  # type: ignore[index]

    assert result is None


def test_user_prompt_hook_leaves_unknown_shorthand_unchanged(tmp_path: Path) -> None:
    _, _, hooks = _build_hooks(tmp_path)

    result = hooks["on_user_prompt_submitted"]({"prompt": "/unknown task"}, {})  # type: ignore[index]

    assert result is None


def test_pre_tool_use_hook_detects_nested_path_values(tmp_path: Path) -> None:
    _, _, hooks = _build_hooks(tmp_path)

    result = hooks["on_pre_tool_use"](  # type: ignore[index]
        {
            "toolName": "edit",
            "toolArgs": {
                "nested": [
                    {"workspacePath": str(tmp_path.parent / "outside.txt")},
                ],
            },
        },
        {},
    )

    assert result["permissionDecision"] == "deny"


def test_post_tool_use_hook_adds_shell_failure_guidance(tmp_path: Path) -> None:
    _, _, hooks = _build_hooks(tmp_path)

    result = hooks["on_post_tool_use"](  # type: ignore[index]
        {
            "toolName": "bash",
            "toolResult": {"exitCode": 1, "stderr": "boom"},
        },
        {},
    )

    assert "Diagnose the existing error" in result["additionalContext"]


def test_error_hook_branches_cover_abort_retry_and_skip(tmp_path: Path) -> None:
    _, _, hooks = _build_hooks(tmp_path)

    assert hooks["on_error_occurred"]({"recoverable": False}, {})["errorHandling"] == "abort"  # type: ignore[index]
    assert hooks["on_error_occurred"]({"recoverable": True, "errorContext": "model_call"}, {})["errorHandling"] == "retry"  # type: ignore[index]
    assert hooks["on_error_occurred"]({"recoverable": True, "errorContext": "tool_execution"}, {})["errorHandling"] == "skip"  # type: ignore[index]
    assert hooks["on_error_occurred"]({"recoverable": True, "errorContext": "other"}, {})["errorHandling"] == "retry"  # type: ignore[index]


def test_session_start_hook_uses_instruction_loader(tmp_path: Path) -> None:
    rules = tmp_path / ".claude" / "rules"
    rules.mkdir(parents=True)
    (rules / "style.md").write_text("Use snake_case everywhere.", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("Project-level rules.", encoding="utf-8")
    gh = tmp_path / ".github"
    gh.mkdir()
    (gh / "copilot-instructions.md").write_text("GitHub rules.", encoding="utf-8")

    _, _, hooks = _build_hooks(tmp_path)

    result = hooks["on_session_start"]({}, {})

    ctx = result["additionalContext"]
    assert "snake_case" in ctx
    assert "Project-level rules." in ctx
    assert "GitHub rules." in ctx
    # Verify ordering: rules before project before github
    assert ctx.index("snake_case") < ctx.index("Project-level rules.")
    assert ctx.index("Project-level rules.") < ctx.index("GitHub rules.")
