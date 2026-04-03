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


from copilotcode_sdk.skill_assets import build_skill_catalog


def _build_hooks_with_skills(
    tmp_path: Path,
) -> tuple[CopilotCodeConfig, MemoryStore, dict[str, object]]:
    """Build hooks with a skill map for skill-completion testing."""
    intake_dir = tmp_path / "skills" / "excel-workbook-intake"
    intake_dir.mkdir(parents=True)
    (intake_dir / "SKILL.md").write_text(
        "---\n"
        "name: excel-workbook-intake\n"
        "description: Ingest workbook.\n"
        "type: data-intake\n"
        "outputs: outputs/intake/\n"
        "requires: none\n"
        "---\n\n# Intake\n",
        encoding="utf-8",
    )
    analysis_dir = tmp_path / "skills" / "rcm-analysis"
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "SKILL.md").write_text(
        "---\n"
        "name: rcm-analysis\n"
        "description: Analyze data.\n"
        "type: rcm-analysis\n"
        "outputs: outputs/rcm_analysis/\n"
        "requires: data-intake\n"
        "---\n\n# Analysis\n",
        encoding="utf-8",
    )

    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        extra_skill_directories=[str(tmp_path / "skills")],
        enabled_skills=(),
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    _, skill_map = build_skill_catalog([str(tmp_path / "skills")])
    hooks = build_default_hooks(config, store, skill_map=skill_map)
    return config, store, hooks


def test_post_tool_use_detects_skill_completion(tmp_path: Path) -> None:
    _, _, hooks = _build_hooks_with_skills(tmp_path)

    result = hooks["on_post_tool_use"](
        {
            "toolName": "write_file",
            "toolArgs": {"path": str(tmp_path / "outputs" / "intake" / "data.parquet")},
            "toolResult": "File written.",
        },
        {},
    )

    assert result is not None
    assert "excel-workbook-intake" in result["additionalContext"]
    assert "rcm-analysis" in result["additionalContext"]


def test_post_tool_use_no_match_for_unrelated_path(tmp_path: Path) -> None:
    _, _, hooks = _build_hooks_with_skills(tmp_path)

    result = hooks["on_post_tool_use"](
        {
            "toolName": "write_file",
            "toolArgs": {"path": str(tmp_path / "src" / "main.py")},
            "toolResult": "File written.",
        },
        {},
    )

    assert result is None


def test_post_tool_use_reinjects_reminder_after_n_calls(tmp_path: Path) -> None:
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        reminder_reinjection_interval=3,
    )
    skill_dir = tmp_path / "skills" / "test-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-skill\ndescription: Test.\ntype: test-type\nrequires: none\noutputs: outputs/test/\n---\n\n# Test\n",
        encoding="utf-8",
    )
    _, skill_map = build_skill_catalog([str(tmp_path / "skills")])
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store, skill_map=skill_map)

    # Make 2 calls — no reminder yet
    for _ in range(2):
        hooks["on_post_tool_use"](
            {"toolName": "read", "toolResult": "content"},
            {},
        )

    # 3rd call — should get reminder
    result = hooks["on_post_tool_use"](
        {"toolName": "read", "toolResult": "content"},
        {},
    )

    assert result is not None
    assert "test-skill" in result["additionalContext"]


def test_post_tool_use_no_reminder_without_skill_map(tmp_path: Path) -> None:
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        reminder_reinjection_interval=1,
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)

    result = hooks["on_post_tool_use"](
        {"toolName": "read", "toolResult": "content"},
        {},
    )

    assert result is None


def test_post_tool_use_triggers_extraction_at_threshold(tmp_path: Path) -> None:
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        extraction_tool_call_interval=3,
        extraction_min_turn_gap=0,
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)

    for _ in range(2):
        hooks["on_post_tool_use"](
            {"toolName": "read", "toolResult": "content"},
            {},
        )

    result = hooks["on_post_tool_use"](
        {"toolName": "read", "toolResult": "content"},
        {},
    )

    assert result is not None
    assert "memory extraction" in result["additionalContext"].lower()


def test_post_tool_use_extraction_respects_min_turn_gap(tmp_path: Path) -> None:
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        extraction_tool_call_interval=1,
        extraction_min_turn_gap=100,
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)

    result = hooks["on_post_tool_use"](
        {"toolName": "read", "toolResult": "content"},
        {},
    )

    assert result is None


def test_post_tool_use_suggests_available_skills(tmp_path: Path) -> None:
    intake_dir = tmp_path / "skills" / "intake"
    intake_dir.mkdir(parents=True)
    (intake_dir / "SKILL.md").write_text(
        "---\nname: intake\ndescription: Ingest.\ntype: data-intake\n"
        "outputs: outputs/intake/\nrequires: none\n---\n\n# Intake\n",
        encoding="utf-8",
    )
    analysis_dir = tmp_path / "skills" / "analysis"
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "SKILL.md").write_text(
        "---\nname: analysis\ndescription: Analyze.\ntype: rcm-analysis\n"
        "outputs: outputs/analysis/\nrequires: data-intake\n---\n\n# Analysis\n",
        encoding="utf-8",
    )

    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        extraction_tool_call_interval=999,
        extraction_min_turn_gap=999,
        reminder_reinjection_interval=0,
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    _, skill_map = build_skill_catalog([str(tmp_path / "skills")])
    hooks = build_default_hooks(config, store, skill_map=skill_map)

    # Simulate calls to pass SUGGESTION_TURN_THRESHOLD (3) and SUGGESTION_INTERVAL (15).
    # Condition: count > 3 AND count - last_suggestion >= 15, last_suggestion starts at 0.
    # First fires at call 15: 15 > 3 AND 15 - 0 >= 15 → fires.
    # Run 14 warm-up calls, then the 15th call is the final assertion call.
    for i in range(14):
        hooks["on_post_tool_use"](
            {"toolName": "read", "toolResult": "ok"},
            {},
        )

    result = hooks["on_post_tool_use"](
        {"toolName": "read", "toolResult": "ok"},
        {},
    )

    assert result is not None
    ctx = result["additionalContext"]
    assert "intake" in ctx.lower()


def test_post_tool_use_no_suggestion_before_threshold(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "intake"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: intake\ndescription: Ingest.\ntype: data-intake\n"
        "outputs: outputs/intake/\nrequires: none\n---\n\n# Intake\n",
        encoding="utf-8",
    )

    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        extraction_tool_call_interval=999,
        extraction_min_turn_gap=999,
        reminder_reinjection_interval=0,
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    _, skill_map = build_skill_catalog([str(tmp_path / "skills")])
    hooks = build_default_hooks(config, store, skill_map=skill_map)

    result = hooks["on_post_tool_use"](
        {"toolName": "read", "toolResult": "ok"},
        {},
    )

    assert result is None
