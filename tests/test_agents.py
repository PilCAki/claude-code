from __future__ import annotations

import re
from pathlib import Path

import pytest

from copilotcode_sdk.agents import (
    COORDINATOR_PROTOCOL,
    TASK_TOOLS_ALL,
    TASK_TOOLS_READ,
    TASK_TOOLS_WRITE,
    WORKTREE_NOTICE,
    build_default_custom_agents,
    persist_agent_output,
)
from copilotcode_sdk.config import CopilotCodeConfig
from copilotcode_sdk.memory import MemoryStore


def test_build_default_custom_agents_respects_enabled_agent_filter(tmp_path) -> None:
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        enabled_agents=("planner", "verifier"),
    )

    agents = build_default_custom_agents(config)

    assert [agent["name"] for agent in agents] == ["planner", "verifier"]
    # Tasks v2 is enabled by default, so planner gets read-only task tools
    assert agents[0]["tools"] == ["read", "search", "execute", "TaskList", "TaskGet", "TaskOutput"]
    assert agents[1]["infer"] is False


def test_build_default_custom_agents_skips_unknown_and_appends_extra_agents(tmp_path) -> None:
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        enabled_agents=("planner", "unknown"),
        extra_agents=(
            {
                "name": "custom-reviewer",
                "display_name": "Custom Reviewer",
                "description": "Custom agent",
                "tools": ["read"],
                "prompt": "Inspect and report.",
            },
        ),
    )

    agents = build_default_custom_agents(config)

    assert [agent["name"] for agent in agents] == ["planner", "custom-reviewer"]
    assert "CopilotCode" in agents[0]["prompt"]


def test_researcher_prompt_includes_read_only_rules(tmp_path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path)
    agents = build_default_custom_agents(config)
    researcher = next(a for a in agents if a["name"] == "researcher")

    assert "read-only" in researcher["prompt"].lower()
    assert "file paths" in researcher["prompt"].lower()
    assert "Output contract" in researcher["prompt"]
    assert "STRICTLY PROHIBITED" in researcher["prompt"]


def test_implementer_prompt_includes_skill_awareness(tmp_path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path)
    agents = build_default_custom_agents(config)
    implementer = next(a for a in agents if a["name"] == "implementer")

    assert "skill" in implementer["prompt"].lower()
    assert "verify" in implementer["prompt"].lower()
    assert "Output contract" in implementer["prompt"]


def test_verifier_prompt_includes_adversarial_posture(tmp_path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path)
    agents = build_default_custom_agents(config)
    verifier = next(a for a in agents if a["name"] == "verifier")

    assert "wrong until proven" in verifier["prompt"].lower() or "adversarial" in verifier["prompt"].lower()
    assert "PASS" in verifier["prompt"]
    assert "FAIL" in verifier["prompt"]
    assert "Do not edit" in verifier["prompt"]


def test_planner_prompt_includes_risk_assessment(tmp_path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path)
    agents = build_default_custom_agents(config)
    planner = next(a for a in agents if a["name"] == "planner")

    assert "risk" in planner["prompt"].lower()
    assert "Output contract" in planner["prompt"]
    assert "STRICTLY PROHIBITED" in planner["prompt"]


# ---------------------------------------------------------------------------
# Agent-specific task tool scopes
# ---------------------------------------------------------------------------


def test_task_tools_constants() -> None:
    assert TASK_TOOLS_READ == ["TaskList", "TaskGet", "TaskOutput"]
    assert TASK_TOOLS_WRITE == ["TaskCreate", "TaskUpdate"]
    assert TASK_TOOLS_ALL == ["TaskList", "TaskGet", "TaskOutput", "TaskCreate", "TaskUpdate"]


def test_researcher_gets_read_only_task_tools(tmp_path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path, enable_tasks_v2=True)
    agents = build_default_custom_agents(config)
    researcher = next(a for a in agents if a["name"] == "researcher")

    for tool in TASK_TOOLS_READ:
        assert tool in researcher["tools"], f"{tool} should be in researcher tools"
    for tool in TASK_TOOLS_WRITE:
        assert tool not in researcher["tools"], f"{tool} should NOT be in researcher tools"


def test_planner_gets_read_only_task_tools(tmp_path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path, enable_tasks_v2=True)
    agents = build_default_custom_agents(config)
    planner = next(a for a in agents if a["name"] == "planner")

    for tool in TASK_TOOLS_READ:
        assert tool in planner["tools"]
    for tool in TASK_TOOLS_WRITE:
        assert tool not in planner["tools"]


def test_implementer_gets_all_task_tools(tmp_path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path, enable_tasks_v2=True)
    agents = build_default_custom_agents(config)
    implementer = next(a for a in agents if a["name"] == "implementer")

    for tool in TASK_TOOLS_ALL:
        assert tool in implementer["tools"], f"{tool} should be in implementer tools"


def test_verifier_gets_read_plus_update_task_tools(tmp_path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path, enable_tasks_v2=True)
    agents = build_default_custom_agents(config)
    verifier = next(a for a in agents if a["name"] == "verifier")

    assert "TaskList" in verifier["tools"]
    assert "TaskGet" in verifier["tools"]
    assert "TaskUpdate" in verifier["tools"]
    assert "TaskCreate" not in verifier["tools"]


def test_no_task_tools_when_tasks_disabled(tmp_path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path, enable_tasks_v2=False)
    agents = build_default_custom_agents(config)

    for agent in agents:
        for tool in TASK_TOOLS_ALL:
            assert tool not in agent["tools"], (
                f"{tool} should NOT be in {agent['name']} tools when tasks disabled"
            )


# ---------------------------------------------------------------------------
# Wave 3: Agent prompt enrichment
# ---------------------------------------------------------------------------


def test_all_agents_include_workspace_context(tmp_path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path)
    agents = build_default_custom_agents(config)

    for agent in agents:
        assert "Working directory:" in agent["prompt"], (
            f"{agent['name']} prompt missing workspace context"
        )


def test_agent_prompts_include_output_contract(tmp_path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path)
    agents = build_default_custom_agents(config)

    for agent in agents:
        assert "Output contract:" in agent["prompt"], (
            f"{agent['name']} prompt missing output contract"
        )


def test_researcher_includes_format_guidance(tmp_path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path)
    agents = build_default_custom_agents(config)
    researcher = next(a for a in agents if a["name"] == "researcher")

    assert "markdown" in researcher["prompt"].lower()


def test_verifier_includes_minimal_fix_guidance(tmp_path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path)
    agents = build_default_custom_agents(config)
    verifier = next(a for a in agents if a["name"] == "verifier")

    assert "minimal fix" in verifier["prompt"].lower()


def test_implementer_includes_task_update_guidance(tmp_path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path)
    agents = build_default_custom_agents(config)
    implementer = next(a for a in agents if a["name"] == "implementer")

    assert "task" in implementer["prompt"].lower()


def test_extra_agent_gets_workspace_context(tmp_path) -> None:
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        enabled_agents=(),
        extra_agents=(
            {
                "name": "custom",
                "display_name": "Custom",
                "description": "Custom agent",
                "tools": ["read"],
                "prompt": "Do custom work.",
            },
        ),
    )

    agents = build_default_custom_agents(config)

    assert len(agents) == 1
    assert "Working directory:" in agents[0]["prompt"]


def test_extra_agent_enrich_false_skips_context(tmp_path) -> None:
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        enabled_agents=(),
        extra_agents=(
            {
                "name": "raw",
                "display_name": "Raw",
                "description": "Raw agent",
                "tools": ["read"],
                "prompt": "Raw prompt only.",
                "enrich": False,
            },
        ),
    )

    agents = build_default_custom_agents(config)

    assert len(agents) == 1
    assert "Working directory:" not in agents[0]["prompt"]
    assert agents[0]["prompt"] == "Raw prompt only."


# ---------------------------------------------------------------------------
# Wave 4.3: Agent prompt content assertions
# ---------------------------------------------------------------------------


def _get_agent(tmp_path, name: str) -> dict:
    config = CopilotCodeConfig(working_directory=tmp_path)
    agents = build_default_custom_agents(config)
    return next(a for a in agents if a["name"] == name)


class TestResearcherPromptContent:
    def test_prohibits_file_creation(self, tmp_path) -> None:
        prompt = _get_agent(tmp_path, "researcher")["prompt"]
        assert "Creating new files" in prompt
        assert "STRICTLY PROHIBITED" in prompt

    def test_mentions_search_tools(self, tmp_path) -> None:
        prompt = _get_agent(tmp_path, "researcher")["prompt"]
        assert "Glob" in prompt
        assert "Grep" in prompt
        assert "Read" in prompt

    def test_fast_posture(self, tmp_path) -> None:
        prompt = _get_agent(tmp_path, "researcher")["prompt"]
        assert "fast agent" in prompt.lower()

    def test_disallowed_tools(self, tmp_path) -> None:
        prompt = _get_agent(tmp_path, "researcher")["prompt"]
        assert "Disallowed tools" in prompt


class TestPlannerPromptContent:
    def test_required_process_steps(self, tmp_path) -> None:
        prompt = _get_agent(tmp_path, "planner")["prompt"]
        assert "Understand requirements" in prompt
        assert "Explore thoroughly" in prompt
        assert "Design solution" in prompt
        assert "Detail the plan" in prompt

    def test_critical_files_ending(self, tmp_path) -> None:
        prompt = _get_agent(tmp_path, "planner")["prompt"]
        assert "Critical Files for Implementation" in prompt


class TestVerifierPromptContent:
    def test_change_type_strategies(self, tmp_path) -> None:
        prompt = _get_agent(tmp_path, "verifier")["prompt"]
        assert "Frontend changes" in prompt
        assert "Backend/API changes" in prompt
        assert "CLI/script changes" in prompt

    def test_universal_baseline(self, tmp_path) -> None:
        prompt = _get_agent(tmp_path, "verifier")["prompt"]
        assert "Run the build" in prompt
        assert "Run the test suite" in prompt
        assert "Run linters" in prompt

    def test_adversarial_probes(self, tmp_path) -> None:
        prompt = _get_agent(tmp_path, "verifier")["prompt"]
        assert "Concurrency" in prompt
        assert "Boundary values" in prompt
        assert "Idempotency" in prompt

    def test_verdict_format(self, tmp_path) -> None:
        prompt = _get_agent(tmp_path, "verifier")["prompt"]
        assert "VERDICT: PASS" in prompt
        assert "VERDICT: FAIL" in prompt
        assert "VERDICT: PARTIAL" in prompt

    def test_rationalizations(self, tmp_path) -> None:
        prompt = _get_agent(tmp_path, "verifier")["prompt"]
        assert "I verified by reading the code" in prompt
        assert "not verification" in prompt


class TestImplementerPromptContent:
    def test_verification_handoff(self, tmp_path) -> None:
        prompt = _get_agent(tmp_path, "implementer")["prompt"]
        assert "verification" in prompt.lower()
        assert "handoff" in prompt.lower() or "delegate" in prompt.lower()

    def test_task_update_guidance(self, tmp_path) -> None:
        prompt = _get_agent(tmp_path, "implementer")["prompt"]
        assert "in_progress" in prompt
        assert "completed" in prompt


def test_workspace_context_includes_instruction_files(tmp_path) -> None:
    # Create a CLAUDE.md so it shows up in context
    (tmp_path / "CLAUDE.md").write_text("# Instructions", encoding="utf-8")
    config = CopilotCodeConfig(working_directory=tmp_path)
    agents = build_default_custom_agents(config)
    researcher = next(a for a in agents if a["name"] == "researcher")

    assert "CLAUDE.md" in researcher["prompt"]


# ---------------------------------------------------------------------------
# Wave 4.3: Snapshot tests
# ---------------------------------------------------------------------------

SNAPSHOT_DIR = Path(__file__).parent / "snapshots" / "agents"


def _normalize_prompt(prompt: str) -> str:
    """Strip the workspace context line that contains a temp path."""
    return re.sub(r"Working directory: `[^`]+`", "Working directory: `<TMPDIR>`", prompt)


class TestAgentPromptSnapshots:
    @pytest.mark.parametrize("agent_name", ["researcher", "planner", "implementer", "verifier"])
    def test_prompt_matches_snapshot(self, tmp_path, agent_name) -> None:
        snapshot_path = SNAPSHOT_DIR / f"{agent_name}.txt"
        assert snapshot_path.exists(), f"Snapshot missing: {snapshot_path}"

        agent = _get_agent(tmp_path, agent_name)
        current = _normalize_prompt(agent["prompt"])
        expected = _normalize_prompt(snapshot_path.read_text(encoding="utf-8"))
        assert current == expected, (
            f"{agent_name} prompt changed — update snapshot at {snapshot_path} if intentional"
        )


# ---------------------------------------------------------------------------
# Wave 5: Agent memory persistence
# ---------------------------------------------------------------------------


class TestPersistAgentOutput:
    def test_persists_output(self, tmp_path) -> None:
        store = MemoryStore(tmp_path, tmp_path / ".memory")
        store.ensure()
        result = persist_agent_output(store, "researcher", "Found 3 critical files.")
        assert result is not None
        assert result.exists()
        text = result.read_text(encoding="utf-8")
        assert "Found 3 critical files" in text

    def test_returns_none_for_empty_output(self, tmp_path) -> None:
        store = MemoryStore(tmp_path, tmp_path / ".memory")
        store.ensure()
        assert persist_agent_output(store, "researcher", "") is None
        assert persist_agent_output(store, "researcher", "   ") is None

    def test_truncates_long_output(self, tmp_path) -> None:
        store = MemoryStore(tmp_path, tmp_path / ".memory")
        store.ensure()
        long_output = "x" * 20_000
        result = persist_agent_output(store, "planner", long_output, max_persist_chars=500)
        assert result is not None
        text = result.read_text(encoding="utf-8")
        assert "truncated" in text.lower()

    def test_upserts_on_same_agent(self, tmp_path) -> None:
        store = MemoryStore(tmp_path, tmp_path / ".memory")
        store.ensure()
        p1 = persist_agent_output(store, "researcher", "First run findings.")
        p2 = persist_agent_output(store, "researcher", "Second run findings.")
        assert p1 == p2  # same slug → same file
        text = p2.read_text(encoding="utf-8")
        assert "Second run findings" in text


# ---------------------------------------------------------------------------
# Wave 5: Coordinator mode protocol
# ---------------------------------------------------------------------------


class TestCoordinatorMode:
    def test_coordinator_protocol_not_added_by_default(self, tmp_path) -> None:
        config = CopilotCodeConfig(working_directory=tmp_path)
        agents = build_default_custom_agents(config)
        implementer = next(a for a in agents if a["name"] == "implementer")
        assert "Coordinator Mode" not in implementer["prompt"]

    def test_coordinator_protocol_added_when_enabled(self, tmp_path) -> None:
        config = CopilotCodeConfig(
            working_directory=tmp_path,
            enable_coordinator_mode=True,
        )
        agents = build_default_custom_agents(config)
        implementer = next(a for a in agents if a["name"] == "implementer")
        assert "Coordinator Mode" in implementer["prompt"]
        assert "task-notification" in implementer["prompt"]

    def test_coordinator_only_on_implementer(self, tmp_path) -> None:
        config = CopilotCodeConfig(
            working_directory=tmp_path,
            enable_coordinator_mode=True,
        )
        agents = build_default_custom_agents(config)
        for agent in agents:
            if agent["name"] != "implementer":
                assert "Coordinator Mode" not in agent["prompt"]

    def test_coordinator_protocol_content(self) -> None:
        assert "task-notification" in COORDINATOR_PROTOCOL
        assert "Do NOT start the next task" in COORDINATOR_PROTOCOL
        assert "do not share context" in COORDINATOR_PROTOCOL.lower()


# ---------------------------------------------------------------------------
# Wave 5: Max agent turns
# ---------------------------------------------------------------------------


class TestMaxAgentTurns:
    def test_no_max_turns_by_default(self, tmp_path) -> None:
        config = CopilotCodeConfig(working_directory=tmp_path)
        agents = build_default_custom_agents(config)
        for agent in agents:
            assert "max_turns" not in agent

    def test_max_turns_set_when_configured(self, tmp_path) -> None:
        config = CopilotCodeConfig(working_directory=tmp_path, max_agent_turns=50)
        agents = build_default_custom_agents(config)
        for agent in agents:
            assert agent["max_turns"] == 50

    def test_max_turns_on_extra_agents(self, tmp_path) -> None:
        config = CopilotCodeConfig(
            working_directory=tmp_path,
            max_agent_turns=30,
            enabled_agents=(),
            extra_agents=(
                {
                    "name": "custom",
                    "display_name": "Custom",
                    "description": "Test",
                    "tools": ["read"],
                    "prompt": "Test prompt.",
                },
            ),
        )
        agents = build_default_custom_agents(config)
        assert agents[0]["max_turns"] == 30

    def test_extra_agent_explicit_max_turns_not_overridden(self, tmp_path) -> None:
        config = CopilotCodeConfig(
            working_directory=tmp_path,
            max_agent_turns=30,
            enabled_agents=(),
            extra_agents=(
                {
                    "name": "custom",
                    "display_name": "Custom",
                    "description": "Test",
                    "tools": ["read"],
                    "prompt": "Test prompt.",
                    "max_turns": 100,
                },
            ),
        )
        agents = build_default_custom_agents(config)
        assert agents[0]["max_turns"] == 100


# ---------------------------------------------------------------------------
# Wave 5: Worktree isolation notices
# ---------------------------------------------------------------------------


class TestWorktreeNotices:
    def test_no_worktree_notice_in_normal_repo(self, tmp_path) -> None:
        # Normal repo: .git is a directory
        (tmp_path / ".git").mkdir()
        config = CopilotCodeConfig(working_directory=tmp_path)
        agents = build_default_custom_agents(config)
        for agent in agents:
            assert "Worktree Isolation" not in agent["prompt"]

    def test_worktree_notice_when_git_file(self, tmp_path) -> None:
        # Worktree: .git is a file pointing to main repo
        (tmp_path / ".git").write_text("gitdir: /some/main/.git/worktrees/branch", encoding="utf-8")
        config = CopilotCodeConfig(working_directory=tmp_path)
        agents = build_default_custom_agents(config)
        researcher = next(a for a in agents if a["name"] == "researcher")
        assert "Worktree Isolation" in researcher["prompt"]

    def test_worktree_notice_content(self) -> None:
        assert "isolated copy" in WORKTREE_NOTICE.lower()
        assert "separate branch" in WORKTREE_NOTICE.lower()
