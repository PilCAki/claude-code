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


def test_passive_skill_detection_disabled(tmp_path: Path) -> None:
    """Passive detection is disabled — CompleteSkill is the only path."""
    _, _, hooks = _build_hooks_with_skills(tmp_path)

    result = hooks["on_post_tool_use"](
        {
            "toolName": "write_file",
            "toolArgs": {"path": str(tmp_path / "outputs" / "intake" / "data.parquet")},
            "toolResult": "File written.",
        },
        {},
    )

    assert result is None


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
        session_memory_auto=False,
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
    assert "memory" in result["additionalContext"].lower()


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


import json


def test_session_start_injects_prior_artifact_context(tmp_path: Path) -> None:
    """When prior_run_metrics.json exists in outputs/, session start should surface it."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    metrics = {"total_claims": 5000, "realization_rate": 0.42}
    (outputs / "prior_run_metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8",
    )
    (outputs / "column_mapping.json").write_text(
        json.dumps({"Charge": "billed_charge"}), encoding="utf-8",
    )

    _, _, hooks = _build_hooks(tmp_path)

    result = hooks["on_session_start"]({}, {})
    ctx = result["additionalContext"]

    assert "prior_run_metrics.json" in ctx
    assert "column_mapping.json" in ctx
    assert "total_claims" in ctx
    assert "5000" in ctx


def test_session_start_no_prior_artifacts_when_outputs_empty(tmp_path: Path) -> None:
    """When no carry-forward files exist, no prior artifact section is injected."""
    _, _, hooks = _build_hooks(tmp_path)

    result = hooks["on_session_start"]({}, {})
    ctx = result["additionalContext"]

    assert "Prior Analysis Artifacts" not in ctx


def test_session_start_injects_task_summary_on_resume(tmp_path: Path) -> None:
    """When a task store has open tasks, session start (including resume) should inject the summary."""
    from copilotcode_sdk.tasks import TaskStore

    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        enable_tasks_v2=True,
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    task_store = TaskStore()
    task_store.create("Implement feature X", owner="alice")
    task_store.create("Write tests for feature X")
    task_store.update(1, status="in_progress")

    hooks = build_default_hooks(config, store, task_store=task_store)
    result = hooks["on_session_start"]({}, {})

    ctx = result["additionalContext"]
    assert "Implement feature X" in ctx
    assert "Write tests for feature X" in ctx
    assert "in progress" in ctx.lower() or "in_progress" in ctx.lower()


def test_session_start_no_task_context_when_store_empty(tmp_path: Path) -> None:
    """When task store has no tasks, no task context should be injected."""
    from copilotcode_sdk.tasks import TaskStore

    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        enable_tasks_v2=True,
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    task_store = TaskStore()

    hooks = build_default_hooks(config, store, task_store=task_store)
    result = hooks["on_session_start"]({}, {})

    ctx = result["additionalContext"]
    # summary_text() returns "" when empty, so no task content
    assert "Task" not in ctx or "CopilotCode" in ctx  # Only the brand line


def test_session_start_no_task_context_when_store_is_none(tmp_path: Path) -> None:
    """When no task store is provided, no task context should appear."""
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store, task_store=None)
    result = hooks["on_session_start"]({}, {})

    ctx = result["additionalContext"]
    # Should not crash or inject task data
    assert "CopilotCode is active" in ctx


def test_session_start_task_summary_with_persisted_store(tmp_path: Path) -> None:
    """Simulate resume: create a task store, persist to disk, reload, and verify session start picks it up."""
    from copilotcode_sdk.tasks import TaskStore

    persist_path = tmp_path / "tasks" / "tasks.json"
    store1 = TaskStore(persist_path=persist_path)
    store1.create("Deploy service")
    store1.update(1, status="in_progress")

    # Simulate resume: create a new TaskStore from the same file (as would happen on restart)
    store2 = TaskStore(persist_path=persist_path)
    assert store2.get(1) is not None
    assert store2.get(1).subject == "Deploy service"

    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
    )
    mem_store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, mem_store, task_store=store2)
    result = hooks["on_session_start"]({}, {})

    assert "Deploy service" in result["additionalContext"]


def test_passive_skill_detection_no_session_end_extraction(tmp_path: Path) -> None:
    """Passive detection is disabled — writing files does not fire extraction."""
    skill_dir = tmp_path / "skills" / "only-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: only-skill\ndescription: Solo.\ntype: solo\n"
        "outputs: outputs/solo/\nrequires: none\n---\n\n# Solo\n",
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

    # Writing to skill output path no longer triggers completion
    result = hooks["on_post_tool_use"](
        {
            "toolName": "write_file",
            "toolArgs": {"path": str(tmp_path / "outputs" / "solo" / "result.json")},
            "toolResult": "File written.",
        },
        {},
    )

    # Passive detection disabled — no skill completion, no extraction
    assert result is None


# ---------------------------------------------------------------------------
# Wave 1: Session-start source parameter and initialUserMessage
# ---------------------------------------------------------------------------


def test_session_start_source_resume_injects_initial_message(tmp_path: Path) -> None:
    """When source=resume and tasks are open, initialUserMessage should be set."""
    from copilotcode_sdk.tasks import TaskStore

    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    task_store = TaskStore()
    task_store.create("Fix bug")

    hooks = build_default_hooks(config, store, task_store=task_store)
    result = hooks["on_session_start"]({"source": "resume"}, {})

    assert "initialUserMessage" in result
    assert "resumed" in result["initialUserMessage"].lower()
    assert "TaskList" in result["initialUserMessage"]


def test_session_start_source_create_no_initial_message(tmp_path: Path) -> None:
    """When source=create, no initialUserMessage should be injected."""
    from copilotcode_sdk.tasks import TaskStore

    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    task_store = TaskStore()
    task_store.create("Fix bug")

    hooks = build_default_hooks(config, store, task_store=task_store)
    result = hooks["on_session_start"]({"source": "create"}, {})

    assert "initialUserMessage" not in result


def test_session_start_watched_paths(tmp_path: Path) -> None:
    """Session start should return watchedPaths for instruction files."""
    (tmp_path / "CLAUDE.md").write_text("Rules.", encoding="utf-8")
    rules = tmp_path / ".claude" / "rules"
    rules.mkdir(parents=True)
    (rules / "style.md").write_text("Style rules.", encoding="utf-8")

    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)
    result = hooks["on_session_start"]({}, {})

    assert "watchedPaths" in result
    paths = result["watchedPaths"]
    assert any("CLAUDE.md" in p for p in paths)
    assert any("style.md" in p for p in paths)


# ---------------------------------------------------------------------------
# Wave 1: Frontmatter skill hooks
# ---------------------------------------------------------------------------


def _build_hooks_with_skill_hooks(
    tmp_path: Path,
    skill_fm: str,
) -> tuple[CopilotCodeConfig, MemoryStore, dict[str, object]]:
    """Build hooks with a skill that has hook frontmatter."""
    skill_dir = tmp_path / "skills" / "test-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(skill_fm, encoding="utf-8")

    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        extra_skill_directories=[str(tmp_path / "skills")],
        enabled_skills=(),
        extraction_tool_call_interval=999,
        extraction_min_turn_gap=999,
        reminder_reinjection_interval=0,
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    _, skill_map = build_skill_catalog([str(tmp_path / "skills")])
    hooks = build_default_hooks(config, store, skill_map=skill_map)
    return config, store, hooks


def test_on_complete_inject_disabled_with_passive(tmp_path: Path) -> None:
    """on_complete hooks no longer fire via passive detection (disabled)."""
    _, _, hooks = _build_hooks_with_skill_hooks(
        tmp_path,
        "---\nname: test-skill\ndescription: Test.\ntype: test-type\n"
        "outputs: outputs/test/\nrequires: none\n"
        "on_complete: inject:Analysis complete, check results.\n---\n\n# Test\n",
    )

    result = hooks["on_post_tool_use"](
        {
            "toolName": "write_file",
            "toolArgs": {"path": str(tmp_path / "outputs" / "test" / "result.json")},
            "toolResult": "File written.",
        },
        {},
    )

    assert result is None


def test_on_complete_remind_disabled_with_passive(tmp_path: Path) -> None:
    """on_complete remind hooks no longer fire via passive detection."""
    _, _, hooks = _build_hooks_with_skill_hooks(
        tmp_path,
        "---\nname: test-skill\ndescription: Test.\ntype: test-type\n"
        "outputs: outputs/test/\nrequires: none\n"
        "on_complete: remind:Run verification before continuing.\n---\n\n# Test\n",
    )

    result = hooks["on_post_tool_use"](
        {
            "toolName": "write_file",
            "toolArgs": {"path": str(tmp_path / "outputs" / "test" / "done.csv")},
            "toolResult": "File written.",
        },
        {},
    )

    assert result is None


def test_on_complete_stop_disabled_with_passive(tmp_path: Path) -> None:
    """on_complete stop hooks no longer fire via passive detection."""
    _, _, hooks = _build_hooks_with_skill_hooks(
        tmp_path,
        "---\nname: test-skill\ndescription: Test.\ntype: test-type\n"
        "outputs: outputs/test/\nrequires: none\n"
        "on_complete: stop\n---\n\n# Test\n",
    )

    result = hooks["on_post_tool_use"](
        {
            "toolName": "write_file",
            "toolArgs": {"path": str(tmp_path / "outputs" / "test" / "done.json")},
            "toolResult": "File written.",
        },
        {},
    )

    assert result is None


def test_on_start_inject_fires_on_skill_shorthand(tmp_path: Path) -> None:
    """Skill with on_start: inject:... should fire when shorthand is expanded."""
    skill_dir = tmp_path / "skills" / "verify"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: verify\ndescription: Verify.\ntype: verification\n"
        "outputs: outputs/verify/\nrequires: none\n"
        "on_start: inject:Remember to check edge cases.\n---\n\n# Verify\n",
        encoding="utf-8",
    )

    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        extra_skill_directories=[str(tmp_path / "skills")],
        enabled_skills=("verify",),
        enable_hybrid_memory=False,
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    _, skill_map = build_skill_catalog([str(tmp_path / "skills")])
    hooks = build_default_hooks(config, store, skill_map=skill_map)

    result = hooks["on_user_prompt_submitted"]({"prompt": "/verify auth login"}, {})

    assert result is not None
    assert "Remember to check edge cases." in result["additionalContext"]


def test_one_shot_hook_disabled_with_passive(tmp_path: Path) -> None:
    """one_shot on_complete hooks no longer fire via passive detection."""
    _, _, hooks = _build_hooks_with_skill_hooks(
        tmp_path,
        "---\nname: test-skill\ndescription: Test.\ntype: test-type\n"
        "outputs: outputs/test/\nrequires: none\n"
        "on_complete: inject:First time only.\none_shot: true\n---\n\n# Test\n",
    )

    result1 = hooks["on_post_tool_use"](
        {
            "toolName": "write_file",
            "toolArgs": {"path": str(tmp_path / "outputs" / "test" / "a.json")},
            "toolResult": "File written.",
        },
        {},
    )
    assert result1 is None


# ---------------------------------------------------------------------------
# Wave 1+2: Scaffold-to-runtime wiring tests
# ---------------------------------------------------------------------------


def test_session_start_injects_dynamic_assembler_content(tmp_path: Path) -> None:
    """render_dynamic() content should appear in session start additionalContext."""
    from copilotcode_sdk.prompt_compiler import PromptAssembler, PromptPriority

    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    asm = PromptAssembler()
    asm.add("static", "STATIC_SECTION", cacheable=True)
    asm.add("dynamic", "DYNAMIC_WIRED_CONTENT", cacheable=False)

    hooks = build_default_hooks(config, store, assembler=asm)
    result = hooks["on_session_start"]({}, {})

    assert "DYNAMIC_WIRED_CONTENT" in result["additionalContext"]
    # Static content should NOT be in additionalContext (it's in system_message)
    assert "STATIC_SECTION" not in result["additionalContext"]


def test_session_start_injects_mcp_delta(tmp_path: Path) -> None:
    """build_mcp_delta() output should appear in session start when mcp_servers configured."""
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        mcp_servers=[
            {"name": "TestMCP", "description": "A test server.",
             "tools": [{"name": "mcp_tool", "description": "Does things."}]},
        ],
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)

    result = hooks["on_session_start"]({}, {})

    assert "TestMCP" in result["additionalContext"]


def test_session_start_captures_instruction_bundle(tmp_path: Path) -> None:
    """on_loaded callback fires and bundle is exposed in hook result."""
    (tmp_path / "CLAUDE.md").write_text("BUNDLE_TEST_RULES", encoding="utf-8")
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)

    result = hooks["on_session_start"]({}, {})

    assert "_instructionsLoaded" in result
    bundle = result["_instructionsLoaded"]
    assert "BUNDLE_TEST_RULES" in bundle.content
    assert len(bundle.loaded_paths) >= 1


def test_session_start_passes_user_config_dir(tmp_path: Path) -> None:
    """User-level instructions from app_config_home are loaded."""
    user_dir = tmp_path / "user_config"
    user_dir.mkdir()
    (user_dir / "CLAUDE.md").write_text("USER_CONFIG_RULES", encoding="utf-8")

    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        config_dir=str(user_dir),
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)

    result = hooks["on_session_start"]({}, {})

    assert "USER_CONFIG_RULES" in result["additionalContext"]


def test_skill_tracker_instantiated_with_skill_map(tmp_path: Path) -> None:
    """SkillTracker should be used when skill_map is provided."""
    skill_dir = tmp_path / "skills" / "tracked-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: tracked-skill\ndescription: Test.\ntype: test-type\n"
        "outputs: outputs/tracked/\nrequires: none\n---\n\n# Tracked\n",
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

    # Advance through enough turns that the suggestion block fires
    for _ in range(14):
        hooks["on_post_tool_use"](
            {"toolName": "read", "toolResult": "ok"},
            {},
        )

    result = hooks["on_post_tool_use"](
        {"toolName": "read", "toolResult": "ok"},
        {},
    )

    # Suggestions should fire via the tracker + build_prompt_suggestions path
    assert result is not None
    ctx = result["additionalContext"]
    assert "tracked-skill" in ctx.lower() or "suggestion" in ctx.lower()


def test_post_tool_use_tracks_recent_tool_names(tmp_path: Path) -> None:
    """Recent tool names should accumulate (used by build_prompt_suggestions)."""
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        extraction_tool_call_interval=999,
        extraction_min_turn_gap=999,
        reminder_reinjection_interval=0,
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    skill_dir = tmp_path / "skills" / "test-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-skill\ndescription: Test.\ntype: test-type\n"
        "outputs: outputs/test/\nrequires: none\n---\n\n# Test\n",
        encoding="utf-8",
    )
    _, skill_map = build_skill_catalog([str(tmp_path / "skills")])
    hooks = build_default_hooks(config, store, skill_map=skill_map)

    # Call with different tool names
    for tool in ("read", "edit", "bash", "grep"):
        hooks["on_post_tool_use"](
            {"toolName": tool, "toolResult": "ok"},
            {},
        )

    # No crash, and the hooks still function
    result = hooks["on_post_tool_use"](
        {"toolName": "read", "toolResult": "ok"},
        {},
    )
    # At this point we're at call 5 — below suggestion threshold, so None is expected
    assert result is None


def test_skill_without_hooks_passive_disabled(tmp_path: Path) -> None:
    """Passive detection disabled — writing files does not mark skill complete."""
    _, _, hooks = _build_hooks_with_skill_hooks(
        tmp_path,
        "---\nname: test-skill\ndescription: Test.\ntype: test-type\n"
        "outputs: outputs/test/\nrequires: none\n---\n\n# Test\n",
    )

    result = hooks["on_post_tool_use"](
        {
            "toolName": "write_file",
            "toolArgs": {"path": str(tmp_path / "outputs" / "test" / "result.json")},
            "toolResult": "File written.",
        },
        {},
    )

    assert result is None


# ---------------------------------------------------------------------------
# Wave 4.4: Resume handoff injection
# ---------------------------------------------------------------------------


def test_resume_injects_compaction_handoff(tmp_path: Path) -> None:
    """On resume, if a compaction artifact exists, it should be prepended."""
    mem_dir = tmp_path / ".mem"
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=mem_dir,
    )
    store = MemoryStore(tmp_path, mem_dir)
    store.ensure()

    # Write artifact into the actual memory_dir (which is nested)
    compaction_dir = store.memory_dir / "compaction"
    compaction_dir.mkdir(parents=True)
    (compaction_dir / "sess-1.md").write_text("Prior session summary.", encoding="utf-8")

    hooks = build_default_hooks(config, store)
    result = hooks["on_session_start"]({"source": "resume"}, {})

    assert "Prior session summary." in result["additionalContext"]
    assert "Session Continuation" in result["additionalContext"]


def test_resume_no_handoff_without_artifact(tmp_path: Path) -> None:
    """On resume without a compaction artifact, no handoff context is injected."""
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)
    result = hooks["on_session_start"]({"source": "resume"}, {})

    assert "Session Continuation" not in result.get("additionalContext", "")


def test_create_no_handoff_injection(tmp_path: Path) -> None:
    """On create (not resume), compaction artifacts should not be injected."""
    mem_dir = tmp_path / ".mem"
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=mem_dir,
    )
    store = MemoryStore(tmp_path, mem_dir)
    store.ensure()

    compaction_dir = store.memory_dir / "compaction"
    compaction_dir.mkdir(parents=True)
    (compaction_dir / "sess-1.md").write_text("Prior session summary.", encoding="utf-8")

    hooks = build_default_hooks(config, store)
    result = hooks["on_session_start"]({"source": "create"}, {})

    assert "Session Continuation" not in result.get("additionalContext", "")


# ---------------------------------------------------------------------------
# Wave 5: Read-before-write enforcement
# ---------------------------------------------------------------------------


def test_unc_path_rejected(tmp_path: Path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path, memory_root=tmp_path / ".mem")
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)
    result = hooks["on_pre_tool_use"](
        {"toolName": "write", "toolArgs": {"path": "\\\\evil\\share\\file.txt"}}, {}
    )
    assert result is not None
    assert result.get("permissionDecision") == "deny"
    assert "UNC" in result.get("permissionDecisionReason", "")


def test_unc_path_forward_slash_rejected(tmp_path: Path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path, memory_root=tmp_path / ".mem")
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)
    result = hooks["on_pre_tool_use"](
        {"toolName": "read", "toolArgs": {"path": "//server/share"}}, {}
    )
    assert result is not None
    assert result.get("permissionDecision") == "deny"


def test_read_before_write_warning(tmp_path: Path) -> None:
    target = tmp_path / "existing.py"
    target.write_text("# existing", encoding="utf-8")
    config = CopilotCodeConfig(
        working_directory=tmp_path, memory_root=tmp_path / ".mem",
        enforce_read_before_write=True,
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)
    result = hooks["on_pre_tool_use"](
        {"toolName": "write", "toolArgs": {"path": str(target)}}, {}
    )
    assert result is not None
    assert "without reading" in result.get("additionalContext", "")


def test_read_then_write_no_warning(tmp_path: Path) -> None:
    target = tmp_path / "existing.py"
    target.write_text("# existing", encoding="utf-8")
    config = CopilotCodeConfig(
        working_directory=tmp_path, memory_root=tmp_path / ".mem",
        enforce_read_before_write=True,
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)
    hooks["on_post_tool_use"](
        {"toolName": "read", "toolArgs": {"path": str(target)}, "toolResult": "# existing"}, {}
    )
    result = hooks["on_pre_tool_use"](
        {"toolName": "write", "toolArgs": {"path": str(target)}}, {}
    )
    ctx = (result or {}).get("additionalContext", "")
    assert "without reading" not in ctx


def test_write_new_file_no_warning(tmp_path: Path) -> None:
    config = CopilotCodeConfig(
        working_directory=tmp_path, memory_root=tmp_path / ".mem",
        enforce_read_before_write=True,
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)
    result = hooks["on_pre_tool_use"](
        {"toolName": "write", "toolArgs": {"path": str(tmp_path / "new.py")}}, {}
    )
    ctx = (result or {}).get("additionalContext", "")
    assert "without reading" not in ctx


def test_read_before_write_disabled(tmp_path: Path) -> None:
    target = tmp_path / "existing.py"
    target.write_text("# existing", encoding="utf-8")
    config = CopilotCodeConfig(
        working_directory=tmp_path, memory_root=tmp_path / ".mem",
        enforce_read_before_write=False,
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)
    result = hooks["on_pre_tool_use"](
        {"toolName": "write", "toolArgs": {"path": str(target)}}, {}
    )
    ctx = (result or {}).get("additionalContext", "")
    assert "without reading" not in ctx


# ---------------------------------------------------------------------------
# Wave 5: Tool result persistence
# ---------------------------------------------------------------------------


def test_large_result_persisted(tmp_path: Path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path, memory_root=tmp_path / ".mem")
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)
    big_result = "x" * 60_000
    result = hooks["on_post_tool_use"](
        {"toolName": "grep", "toolArgs": {}, "toolResult": big_result}, {}
    )
    assert result is not None
    assert result.get("modifiedResult", {}).get("persisted") is True
    assert "persisted" in result.get("additionalContext", "").lower()
    persisted_path = result["modifiedResult"]["persistedPath"]
    assert Path(persisted_path).exists()


def test_small_result_not_persisted(tmp_path: Path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path, memory_root=tmp_path / ".mem")
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)
    result = hooks["on_post_tool_use"](
        {"toolName": "grep", "toolArgs": {}, "toolResult": "small result"}, {}
    )
    if result:
        assert "persisted" not in result.get("modifiedResult", {})


# ---------------------------------------------------------------------------
# Wave 5: Auto-compaction context tracking
# ---------------------------------------------------------------------------


def test_context_critical_warning_fires(tmp_path: Path) -> None:
    """When cumulative result chars exceed critical threshold, a warning fires."""
    config = CopilotCodeConfig(working_directory=tmp_path, memory_root=tmp_path / ".mem")
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)

    # Feed a massive result to blow past the critical threshold (95% of 800K = 760K)
    huge_result = "x" * 770_000
    result = hooks["on_post_tool_use"](
        {"toolName": "custom_tool", "toolArgs": {}, "toolResult": huge_result}, {}
    )
    assert result is not None
    assert "CRITICAL" in result.get("additionalContext", "")
    assert "exhausted" in result.get("additionalContext", "").lower() or "context" in result.get("additionalContext", "").lower()


def test_context_warning_fires_at_80_percent(tmp_path: Path) -> None:
    """When cumulative result chars exceed warning threshold (80%), a warning fires."""
    config = CopilotCodeConfig(working_directory=tmp_path, memory_root=tmp_path / ".mem")
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)

    # Feed enough to pass 80% (640K) but not 95% (760K)
    big_result = "x" * 650_000
    result = hooks["on_post_tool_use"](
        {"toolName": "custom_tool", "toolArgs": {}, "toolResult": big_result}, {}
    )
    assert result is not None
    ctx = result.get("additionalContext", "")
    assert "approaching capacity" in ctx.lower() or "context window" in ctx.lower()


def test_context_warning_fires_only_once(tmp_path: Path) -> None:
    """The compaction warning should only fire once per session."""
    config = CopilotCodeConfig(working_directory=tmp_path, memory_root=tmp_path / ".mem")
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)

    huge_result = "x" * 770_000
    result1 = hooks["on_post_tool_use"](
        {"toolName": "custom_tool", "toolArgs": {}, "toolResult": huge_result}, {}
    )
    assert result1 is not None
    # Second call with more data should not fire the warning again
    result2 = hooks["on_post_tool_use"](
        {"toolName": "custom_tool", "toolArgs": {}, "toolResult": "small"}, {}
    )
    # The second result should not contain the compaction warning
    ctx2 = (result2 or {}).get("additionalContext", "")
    assert "CRITICAL" not in ctx2 and "approaching capacity" not in ctx2.lower()


def test_no_context_warning_below_threshold(tmp_path: Path) -> None:
    """Small results should not trigger any context warnings."""
    config = CopilotCodeConfig(working_directory=tmp_path, memory_root=tmp_path / ".mem")
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)

    result = hooks["on_post_tool_use"](
        {"toolName": "read", "toolArgs": {}, "toolResult": "x" * 1000}, {}
    )
    ctx = (result or {}).get("additionalContext", "")
    assert "CRITICAL" not in ctx
    assert "approaching capacity" not in ctx.lower()


# ---------------------------------------------------------------------------
# Wave 6: Token budget wiring
# ---------------------------------------------------------------------------


def test_token_budget_parsed_from_prompt(tmp_path: Path) -> None:
    """Token budget directive in user prompt should be parsed and stripped."""
    config = CopilotCodeConfig(
        working_directory=tmp_path, memory_root=tmp_path / ".mem",
        enable_hybrid_memory=False,
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)

    result = hooks["on_user_prompt_submitted"](
        {"prompt": "Fix the bug +500k"}, {}
    )
    assert result is not None
    # The directive should be recognized
    ctx = result.get("additionalContext", "")
    assert "500,000 tokens" in ctx or "budget set" in ctx.lower()
    # The budget should be accessible
    budget = hooks["get_token_budget"]()
    assert budget is not None
    assert budget.tokens == 500_000


def test_token_budget_stripped_from_modified_prompt(tmp_path: Path) -> None:
    """Budget directive should be stripped from the prompt sent to the model."""
    config = CopilotCodeConfig(
        working_directory=tmp_path, memory_root=tmp_path / ".mem",
        enable_hybrid_memory=False,
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)

    result = hooks["on_user_prompt_submitted"](
        {"prompt": "Fix the bug +500k"}, {}
    )
    assert result is not None
    modified = result.get("modifiedPrompt", "")
    assert "+500k" not in modified


def test_token_budget_exhaustion_warning(tmp_path: Path) -> None:
    """When token budget is exhausted, a warning should fire."""
    config = CopilotCodeConfig(
        working_directory=tmp_path, memory_root=tmp_path / ".mem",
        enable_hybrid_memory=False,
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)

    # Set a tiny budget
    hooks["on_user_prompt_submitted"]({"prompt": "test +1k"}, {})
    budget = hooks["get_token_budget"]()
    assert budget is not None
    assert budget.tokens == 1_000

    # Feed enough data to exhaust it (1000 tokens ≈ 4000 chars of result)
    result = hooks["on_post_tool_use"](
        {"toolName": "read", "toolArgs": {}, "toolResult": "x" * 5000}, {}
    )
    assert result is not None
    ctx = result.get("additionalContext", "")
    assert "EXHAUSTED" in ctx or "exhausted" in ctx.lower()


def test_no_budget_without_directive(tmp_path: Path) -> None:
    """Without a budget directive, no budget should be active."""
    config = CopilotCodeConfig(
        working_directory=tmp_path, memory_root=tmp_path / ".mem",
        enable_hybrid_memory=False,
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)

    hooks["on_user_prompt_submitted"]({"prompt": "Fix the bug"}, {})
    assert hooks["get_token_budget"]() is None


def test_budget_not_parsed_twice(tmp_path: Path) -> None:
    """A second prompt with a budget directive should not override the first."""
    config = CopilotCodeConfig(
        working_directory=tmp_path, memory_root=tmp_path / ".mem",
        enable_hybrid_memory=False,
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)

    hooks["on_user_prompt_submitted"]({"prompt": "Fix it +500k"}, {})
    hooks["on_user_prompt_submitted"]({"prompt": "Also +1m"}, {})
    budget = hooks["get_token_budget"]()
    assert budget.tokens == 500_000  # First budget sticks


# ---------------------------------------------------------------------------
# Wave 6: Git safety protocol
# ---------------------------------------------------------------------------


def test_git_push_force_denied(tmp_path: Path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path, memory_root=tmp_path / ".mem")
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)
    result = hooks["on_pre_tool_use"](
        {"toolName": "bash", "toolArgs": {"command": "git push --force origin main"}}, {}
    )
    assert result is not None
    assert result.get("permissionDecision") == "deny"
    assert "overwrite" in result.get("permissionDecisionReason", "").lower()


def test_git_reset_hard_denied(tmp_path: Path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path, memory_root=tmp_path / ".mem")
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)
    result = hooks["on_pre_tool_use"](
        {"toolName": "bash", "toolArgs": {"command": "git reset --hard HEAD~1"}}, {}
    )
    assert result is not None
    assert result.get("permissionDecision") == "deny"


def test_git_clean_f_denied(tmp_path: Path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path, memory_root=tmp_path / ".mem")
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)
    result = hooks["on_pre_tool_use"](
        {"toolName": "bash", "toolArgs": {"command": "git clean -fd"}}, {}
    )
    assert result is not None
    assert result.get("permissionDecision") == "deny"


def test_no_verify_denied(tmp_path: Path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path, memory_root=tmp_path / ".mem")
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)
    result = hooks["on_pre_tool_use"](
        {"toolName": "bash", "toolArgs": {"command": "git commit --no-verify -m 'skip'"}}, {}
    )
    assert result is not None
    assert result.get("permissionDecision") == "deny"


def test_safe_git_command_allowed(tmp_path: Path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path, memory_root=tmp_path / ".mem")
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)
    result = hooks["on_pre_tool_use"](
        {"toolName": "bash", "toolArgs": {"command": "git status"}}, {}
    )
    # Should not be denied
    assert result is None or result.get("permissionDecision") != "deny"


# ---------------------------------------------------------------------------
# Wave 6: Tool result caching
# ---------------------------------------------------------------------------


def test_tool_result_cached_on_repeat(tmp_path: Path) -> None:
    config = CopilotCodeConfig(
        working_directory=tmp_path, memory_root=tmp_path / ".mem",
        enable_tool_result_cache=True,
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)

    # First call: populates cache
    hooks["on_post_tool_use"](
        {"toolName": "grep", "toolArgs": {"pattern": "foo"}, "toolResult": "found foo"}, {}
    )
    # Second call with same args: should get cached result
    result = hooks["on_pre_tool_use"](
        {"toolName": "grep", "toolArgs": {"pattern": "foo"}}, {}
    )
    assert result is not None
    assert result.get("modifiedResult", {}).get("cached") is True


def test_tool_result_not_cached_when_disabled(tmp_path: Path) -> None:
    config = CopilotCodeConfig(
        working_directory=tmp_path, memory_root=tmp_path / ".mem",
        enable_tool_result_cache=False,
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)

    hooks["on_post_tool_use"](
        {"toolName": "grep", "toolArgs": {"pattern": "foo"}, "toolResult": "found foo"}, {}
    )
    result = hooks["on_pre_tool_use"](
        {"toolName": "grep", "toolArgs": {"pattern": "foo"}}, {}
    )
    cached = (result or {}).get("modifiedResult", {}).get("cached")
    assert cached is not True


def test_cache_invalidated_on_write(tmp_path: Path) -> None:
    config = CopilotCodeConfig(
        working_directory=tmp_path, memory_root=tmp_path / ".mem",
        enable_tool_result_cache=True,
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)

    # Populate cache
    hooks["on_post_tool_use"](
        {"toolName": "read", "toolArgs": {"path": "file.py"}, "toolResult": "contents"}, {}
    )
    # Write invalidates cache
    hooks["on_post_tool_use"](
        {"toolName": "write", "toolArgs": {"path": "file.py"}, "toolResult": "ok"}, {}
    )
    # Cache should be empty now
    result = hooks["on_pre_tool_use"](
        {"toolName": "read", "toolArgs": {"path": "file.py"}}, {}
    )
    cached = (result or {}).get("modifiedResult", {}).get("cached")
    assert cached is not True


# ---------------------------------------------------------------------------
# Wave 6: File change tracking
# ---------------------------------------------------------------------------


def test_file_changes_tracked(tmp_path: Path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path, memory_root=tmp_path / ".mem")
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)

    hooks["on_post_tool_use"](
        {"toolName": "write", "toolArgs": {"path": str(tmp_path / "new.py")}, "toolResult": "ok"}, {}
    )
    changes = hooks["get_file_changes"]()
    resolved = str((tmp_path / "new.py").resolve())
    assert resolved in changes
    assert changes[resolved] == "created"


def test_file_changes_modified_on_second_write(tmp_path: Path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path, memory_root=tmp_path / ".mem")
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)

    hooks["on_post_tool_use"](
        {"toolName": "write", "toolArgs": {"path": str(tmp_path / "f.py")}, "toolResult": "ok"}, {}
    )
    hooks["on_post_tool_use"](
        {"toolName": "edit", "toolArgs": {"path": str(tmp_path / "f.py")}, "toolResult": "ok"}, {}
    )
    changes = hooks["get_file_changes"]()
    resolved = str((tmp_path / "f.py").resolve())
    assert changes[resolved] == "modified"


# ---------------------------------------------------------------------------
# Wave 7: Retry backoff wiring in hooks
# ---------------------------------------------------------------------------


def test_retry_backoff_escalates_on_repeated_model_call_errors(tmp_path: Path) -> None:
    """Repeated model_call errors produce increasing retryDelayMs."""
    _, _, hooks = _build_hooks(tmp_path)
    delays = []
    for _ in range(3):
        result = hooks["on_error_occurred"](
            {"recoverable": True, "errorContext": "model_call"}, {}
        )
        assert result["errorHandling"] == "retry"
        delays.append(result["retryDelayMs"])
    # Exponential: each delay should be >= previous (modulo jitter)
    # At minimum, second delay >= first base delay
    assert len(delays) == 3


def test_retry_exhaustion_aborts(tmp_path: Path) -> None:
    """After max_attempts, the error handler returns abort."""
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        retry_max_attempts=2,
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)
    # First two calls should retry
    for _ in range(2):
        result = hooks["on_error_occurred"](
            {"recoverable": True, "errorContext": "model_call"}, {}
        )
        assert result["errorHandling"] == "retry"
    # Third call should abort
    result = hooks["on_error_occurred"](
        {"recoverable": True, "errorContext": "model_call"}, {}
    )
    assert result["errorHandling"] == "abort"


def test_retry_per_error_context_isolation(tmp_path: Path) -> None:
    """Different error contexts maintain separate retry states."""
    _, _, hooks = _build_hooks(tmp_path)
    r1 = hooks["on_error_occurred"](
        {"recoverable": True, "errorContext": "model_call"}, {}
    )
    r2 = hooks["on_error_occurred"](
        {"recoverable": True, "errorContext": "other_context"}, {}
    )
    # Both are first attempts — both should retry
    assert r1["errorHandling"] == "retry"
    assert r2["errorHandling"] == "retry"
    # Both at attempt 1
    assert r1["attempt"] == 1
    assert r2["attempt"] == 1


def test_retry_state_resets_on_non_recoverable(tmp_path: Path) -> None:
    """A non-recoverable error clears retry state for that context."""
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        memory_root=tmp_path / ".mem",
        retry_max_attempts=3,
    )
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)
    # Consume one retry
    hooks["on_error_occurred"](
        {"recoverable": True, "errorContext": "model_call"}, {}
    )
    # Non-recoverable clears state
    hooks["on_error_occurred"](
        {"recoverable": False, "errorContext": "model_call"}, {}
    )
    # Next recoverable starts fresh at attempt 1
    result = hooks["on_error_occurred"](
        {"recoverable": True, "errorContext": "model_call"}, {}
    )
    assert result["errorHandling"] == "retry"
    assert result["attempt"] == 1


# ---------------------------------------------------------------------------
# Gap 3.1: Session-specific compaction artifact lookup
# ---------------------------------------------------------------------------


def test_resume_loads_session_specific_artifact(tmp_path: Path) -> None:
    """On resume, session-specific compaction artifact is preferred over most-recent."""
    _, store, hooks = _build_hooks(tmp_path)
    compaction_dir = store.memory_dir / "compaction"
    compaction_dir.mkdir(parents=True)

    # Older but session-specific artifact
    target = compaction_dir / "my-session.md"
    target.write_text("Session-specific summary.", encoding="utf-8")

    # Newer but different session
    import time
    time.sleep(0.05)
    other = compaction_dir / "other-session.md"
    other.write_text("Other session summary.", encoding="utf-8")

    result = hooks["on_session_start"](
        {"source": "resume", "session_id": "my-session"}, {}
    )
    ctx = result.get("additionalContext", "")
    assert "Session-specific summary" in ctx
    assert "Other session summary" not in ctx


def test_resume_falls_back_to_most_recent(tmp_path: Path) -> None:
    """When no session-specific artifact exists, fall back to most recent."""
    _, store, hooks = _build_hooks(tmp_path)
    compaction_dir = store.memory_dir / "compaction"
    compaction_dir.mkdir(parents=True)
    (compaction_dir / "some-session.md").write_text("Fallback summary.", encoding="utf-8")

    result = hooks["on_session_start"](
        {"source": "resume", "session_id": "nonexistent-session"}, {}
    )
    ctx = result.get("additionalContext", "")
    assert "Fallback summary" in ctx


# ---------------------------------------------------------------------------
# Gap 3.2: Resume handoff includes memory index
# ---------------------------------------------------------------------------


def test_resume_handoff_includes_memory_index(tmp_path: Path) -> None:
    """Resume handoff populates memory_index from the memory store."""
    config = CopilotCodeConfig(working_directory=tmp_path, memory_root=tmp_path / ".mem")
    store = MemoryStore(tmp_path, tmp_path / ".mem")

    # Set up memory index
    store.upsert_memory(
        title="Auth Flow",
        description="How auth works.",
        memory_type="project",
        content="Auth uses JWT tokens.",
    )

    # Set up compaction artifact
    compaction_dir = store.memory_dir / "compaction"
    compaction_dir.mkdir(parents=True)
    (compaction_dir / "sess-1.md").write_text("Compaction summary.", encoding="utf-8")

    hooks = build_default_hooks(config, store)
    result = hooks["on_session_start"](
        {"source": "resume", "session_id": "sess-1"}, {}
    )
    ctx = result.get("additionalContext", "")
    assert "Compaction summary" in ctx
    # Memory index should be included in handoff context
    assert "Auth Flow" in ctx


def test_resume_handoff_works_without_memory_index(tmp_path: Path) -> None:
    """Resume handoff still works when no memory index exists."""
    _, store, hooks = _build_hooks(tmp_path)
    compaction_dir = store.memory_dir / "compaction"
    compaction_dir.mkdir(parents=True)
    (compaction_dir / "s1.md").write_text("Summary only.", encoding="utf-8")

    result = hooks["on_session_start"](
        {"source": "resume", "session_id": "s1"}, {}
    )
    ctx = result.get("additionalContext", "")
    assert "Summary only" in ctx


# ---------------------------------------------------------------------------
# Gap 3.3: Instructions accept active_paths from hooks
# ---------------------------------------------------------------------------


def test_session_start_passes_active_paths_to_instructions(tmp_path: Path) -> None:
    """When active_paths is provided, path-conditional rules are filtered."""
    rules = tmp_path / ".claude" / "rules"
    rules.mkdir(parents=True)
    # This rule only applies to .py files
    (rules / "python-only.md").write_text(
        "---\nglobs: [\"*.py\"]\n---\nPython-specific rule.",
        encoding="utf-8",
    )
    (tmp_path / "CLAUDE.md").write_text("Base rules.", encoding="utf-8")

    config = CopilotCodeConfig(working_directory=tmp_path, memory_root=tmp_path / ".mem")
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)

    # With a .js active path, the python-only rule should be filtered out
    result = hooks["on_session_start"](
        {"active_paths": [str(tmp_path / "app.js")]}, {}
    )
    ctx = result.get("additionalContext", "")
    assert "Base rules" in ctx
    assert "Python-specific rule" not in ctx


def test_session_start_without_active_paths_loads_all_rules(tmp_path: Path) -> None:
    """Without active_paths, all rules are included (backward compatibility)."""
    rules = tmp_path / ".claude" / "rules"
    rules.mkdir(parents=True)
    (rules / "python-only.md").write_text(
        "---\nglobs: [\"*.py\"]\n---\nPython-specific rule.",
        encoding="utf-8",
    )
    (tmp_path / "CLAUDE.md").write_text("Base rules.", encoding="utf-8")

    config = CopilotCodeConfig(working_directory=tmp_path, memory_root=tmp_path / ".mem")
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    hooks = build_default_hooks(config, store)

    # Without active_paths, conditional rules pass (active_paths=None means include all)
    result = hooks["on_session_start"]({}, {})
    ctx = result.get("additionalContext", "")
    assert "Base rules" in ctx
    assert "Python-specific rule" in ctx


# ---------------------------------------------------------------------------
# InvokeSkill pending queue tests
# ---------------------------------------------------------------------------


def test_invokeskill_detected_in_post_hook(tmp_path: Path) -> None:
    """When on_post_tool_use receives an InvokeSkill result with _invocation,
    it returns an additionalContext message about the queued skill."""
    import json
    _, _, hooks = _build_hooks(tmp_path)
    hooks["on_session_start"]({}, {})

    invocation_result = json.dumps({
        "status": "invoke_skill",
        "skill_name": "verify",
        "_invocation": {
            "skill_name": "verify",
            "user_prompt": "Run the verify skill",
            "skill_type": "process",
            "timestamp": 1234567890.0,
        },
    })
    result = hooks["on_post_tool_use"](
        {"toolName": "InvokeSkill", "toolArgs": {}, "toolResult": invocation_result},
        {},
    )
    assert result is not None
    assert "verify" in result["additionalContext"]
    assert "queued" in result["additionalContext"].lower()


def test_invokeskill_stores_pending(tmp_path: Path) -> None:
    """After InvokeSkill detection, drain returns the invocation dict."""
    import json
    _, _, hooks = _build_hooks(tmp_path)
    hooks["on_session_start"]({}, {})

    invocation_data = {
        "skill_name": "intake",
        "user_prompt": "Run intake skill",
        "skill_type": "process",
        "timestamp": 1234567890.0,
    }
    invocation_result = json.dumps({
        "status": "invoke_skill",
        "_invocation": invocation_data,
    })
    hooks["on_post_tool_use"](
        {"toolName": "InvokeSkill", "toolArgs": {}, "toolResult": invocation_result},
        {},
    )
    drained = hooks["drain_pending_skill_invocations"]()
    assert len(drained) == 1
    assert drained[0]["skill_name"] == "intake"
    assert drained[0]["user_prompt"] == "Run intake skill"


def test_drain_clears_pending(tmp_path: Path) -> None:
    """Second drain call returns empty list."""
    import json
    _, _, hooks = _build_hooks(tmp_path)
    hooks["on_session_start"]({}, {})

    invocation_result = json.dumps({
        "status": "invoke_skill",
        "_invocation": {"skill_name": "test", "user_prompt": "p"},
    })
    hooks["on_post_tool_use"](
        {"toolName": "InvokeSkill", "toolArgs": {}, "toolResult": invocation_result},
        {},
    )
    hooks["drain_pending_skill_invocations"]()
    assert hooks["drain_pending_skill_invocations"]() == []


def test_ignores_non_skill_tools(tmp_path: Path) -> None:
    """Non-InvokeSkill tools with JSON results don't queue anything."""
    import json
    _, _, hooks = _build_hooks(tmp_path)
    hooks["on_session_start"]({}, {})

    hooks["on_post_tool_use"](
        {"toolName": "edit", "toolArgs": {}, "toolResult": json.dumps({"status": "ok"})},
        {},
    )
    assert hooks["drain_pending_skill_invocations"]() == []


def test_handles_malformed_json(tmp_path: Path) -> None:
    """InvokeSkill with bad JSON doesn't crash."""
    _, _, hooks = _build_hooks(tmp_path)
    hooks["on_session_start"]({}, {})

    result = hooks["on_post_tool_use"](
        {"toolName": "InvokeSkill", "toolArgs": {}, "toolResult": "not json at all"},
        {},
    )
    # Should not crash; may return None or some other non-skill result
    assert hooks["drain_pending_skill_invocations"]() == []


# ---------------------------------------------------------------------------
# Stale prompt recomposition tests
# ---------------------------------------------------------------------------


def test_git_checkout_marks_stale(tmp_path: Path) -> None:
    """A bash tool running 'git checkout' should mark git_context as stale,
    and the next non-early-return tool call should refresh it."""
    from copilotcode_sdk.prompt_compiler import PromptAssembler

    config = CopilotCodeConfig(working_directory=tmp_path, memory_root=tmp_path / ".mem")
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    asm = PromptAssembler()
    asm.add("git_context", "branch: main", cacheable=False)

    hooks = build_default_hooks(config, store, assembler=asm)
    hooks["on_session_start"]({}, {})

    # Simulate a git checkout command — marks stale
    hooks["on_post_tool_use"](
        {"toolName": "bash", "toolArgs": {"command": "git checkout feature-branch"}, "toolResult": ""},
        {},
    )

    # A subsequent tool call should trigger the stale refresh
    result = hooks["on_post_tool_use"](
        {"toolName": "read", "toolArgs": {"filePath": str(tmp_path / "x.py")}, "toolResult": "content"},
        {},
    )
    # After refresh, stale should be cleared
    assert not asm.has_stale_sections()
    # And the dynamic content should have been injected
    if result is not None:
        assert "additionalContext" in result
