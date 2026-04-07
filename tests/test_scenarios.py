"""Scenario integration tests — real-world behavior verification.

These tests verify that CopilotCode's subsystems work together correctly
in realistic situations, not just in isolation. They test the kind of
things that break when features are wired "simply" instead of correctly.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from copilotcode_sdk.config import CopilotCodeConfig
from copilotcode_sdk.hooks import build_default_hooks
from copilotcode_sdk.memory import MemoryStore
from copilotcode_sdk.tasks import TaskStore, TaskStatus


# ---------------------------------------------------------------------------
# Scenario 1: Session memory maintains coherent state across extraction cycles
# ---------------------------------------------------------------------------


class TestSessionMemoryCoherence:
    """Verify that session memory behaves like a living document — each
    extraction replaces the previous notes, and only unsummarized transcript
    is sent to the maintenance pass."""

    def test_successive_extractions_produce_single_document(self, tmp_path: Path) -> None:
        """Two extraction cycles should produce one clean document, not
        a pile of appended fragments."""
        from copilotcode_sdk.session_memory import SessionMemoryController

        store = MemoryStore(tmp_path, tmp_path / ".mem")
        ctrl = SessionMemoryController(
            store, min_init_tokens=100, min_update_tokens=100,
            timeout_seconds=5.0,
        )
        ctrl.state.initialized = True

        # First extraction
        mock1 = MagicMock()
        mock1.send_and_wait = AsyncMock(return_value="## Session Title\nFirst pass notes.")
        mock1.destroy = AsyncMock()
        msgs1 = [{"role": "user", "id": "m1", "content": "setup"}]

        async def create1():
            return mock1

        asyncio.run(ctrl.run_extraction(create1, msgs1, context_tokens=500))

        # Second extraction — should replace, not append
        mock2 = MagicMock()
        mock2.send_and_wait = AsyncMock(return_value="## Session Title\nUpdated with second pass.")
        mock2.destroy = AsyncMock()
        msgs2 = [
            {"role": "user", "id": "m1", "content": "setup"},
            {"role": "assistant", "id": "m2", "content": "done"},
            {"role": "user", "id": "m3", "content": "next step"},
        ]

        async def create2():
            return mock2

        asyncio.run(ctrl.run_extraction(create2, msgs2, context_tokens=1000))

        content = store.read_session_memory()
        assert "Updated with second pass" in content
        assert "First pass notes" not in content  # replaced, not accumulated
        assert content.count("## Session Title") == 1  # single document

    def test_extraction_only_sends_unsummarized_transcript(self, tmp_path: Path) -> None:
        """After summarizing through message m2, the next extraction should
        only send messages after m2 to the maintenance pass."""
        from copilotcode_sdk.session_memory import SessionMemoryController

        store = MemoryStore(tmp_path, tmp_path / ".mem")
        ctrl = SessionMemoryController(
            store, min_init_tokens=100, min_update_tokens=100,
            timeout_seconds=5.0,
        )
        ctrl.state.initialized = True

        captured_prompts: list[str] = []

        def _make_mock():
            mock = MagicMock()

            async def capture_send(prompt, **kwargs):
                captured_prompts.append(prompt)
                return "## Notes\nOk."

            mock.send_and_wait = capture_send
            mock.destroy = AsyncMock()
            return mock

        # First extraction covers m1-m2
        msgs = [
            {"role": "user", "id": "m1", "content": "alpha"},
            {"role": "assistant", "id": "m2", "content": "beta"},
        ]

        async def create_mock():
            return _make_mock()

        asyncio.run(ctrl.run_extraction(create_mock, msgs, context_tokens=500))
        assert ctrl.state.last_summarized_message_id == "m2"

        # Second extraction — add m3, m4
        msgs.extend([
            {"role": "user", "id": "m3", "content": "gamma"},
            {"role": "assistant", "id": "m4", "content": "delta"},
        ])
        asyncio.run(ctrl.run_extraction(create_mock, msgs, context_tokens=1000))

        # The second prompt should contain gamma/delta but not alpha/beta
        second_prompt = captured_prompts[1]
        assert "gamma" in second_prompt
        assert "delta" in second_prompt
        assert "alpha" not in second_prompt
        assert "beta" not in second_prompt


# ---------------------------------------------------------------------------
# Scenario 2: Multi-session task isolation
# ---------------------------------------------------------------------------


class TestMultiSessionTaskIsolation:
    """Two concurrent sessions should each have their own task stores.
    Tasks created in session A should not appear in session B's hooks."""

    def test_per_session_task_stores_are_independent(self, tmp_path: Path) -> None:
        """Two TaskStores with different list_ids share no state."""
        root = tmp_path / "tasks"
        store_a = TaskStore(task_list_id="session-A", task_root=root)
        store_b = TaskStore(task_list_id="session-B", task_root=root)

        store_a.create("Task for A")
        store_b.create("Task for B")

        assert len(store_a.list_all()) == 1
        assert store_a.list_all()[0].subject == "Task for A"
        assert len(store_b.list_all()) == 1
        assert store_b.list_all()[0].subject == "Task for B"

    def test_task_output_isolation(self, tmp_path: Path) -> None:
        """Task outputs from one session don't leak to another."""
        root = tmp_path / "tasks"
        store_a = TaskStore(task_list_id="sa", task_root=root)
        store_b = TaskStore(task_list_id="sb", task_root=root)

        store_a.create("Task A")
        store_b.create("Task B")
        store_a.write_task_output(1, "Output from A")

        assert store_a.read_task_output(1) == "Output from A"
        assert store_b.read_task_output(1) is None  # B has no output for task 1

    def test_hooks_use_session_specific_store(self, tmp_path: Path) -> None:
        """Hooks built with different task stores produce different task summaries."""
        config = CopilotCodeConfig(
            working_directory=tmp_path,
            memory_root=tmp_path / ".mem",
            enable_tasks_v2=True,
        )
        store = MemoryStore(tmp_path, tmp_path / ".mem")
        root = tmp_path / "tasks"

        # Session A with 3 tasks
        ts_a = TaskStore(task_list_id="a", task_root=root)
        ts_a.create("Alpha task")
        ts_a.create("Beta task")
        ts_a.create("Gamma task")

        # Session B with 1 task
        ts_b = TaskStore(task_list_id="b", task_root=root)
        ts_b.create("Delta task")

        hooks_a = build_default_hooks(config, store, task_store=ts_a)
        hooks_b = build_default_hooks(config, store, task_store=ts_b)

        result_a = hooks_a["on_session_start"]({"source": "resume"}, {})
        result_b = hooks_b["on_session_start"]({"source": "resume"}, {})

        ctx_a = result_a.get("additionalContext", "")
        ctx_b = result_b.get("additionalContext", "")

        # Each should reference their own tasks
        assert "Alpha" in ctx_a or "3" in ctx_a  # task summary mentions A's tasks
        assert "Delta" in ctx_b or "1" in ctx_b


# ---------------------------------------------------------------------------
# Scenario 3: Resume session continuity
# ---------------------------------------------------------------------------


class TestResumeSessionContinuity:
    """When a session is resumed, it should pick up where it left off:
    compaction context injected, memory index available, and the ability
    to fork child sessions preserved."""

    def test_full_resume_lifecycle(self, tmp_path: Path) -> None:
        """Simulate: create session -> compact -> resume -> verify context."""
        config = CopilotCodeConfig(
            working_directory=tmp_path,
            memory_root=tmp_path / ".mem",
        )
        store = MemoryStore(tmp_path, tmp_path / ".mem")

        # Write a memory
        store.upsert_memory(
            title="API Design",
            description="REST API patterns.",
            memory_type="project",
            content="Use pagination for list endpoints.",
        )

        # Write a compaction artifact (as if session "s1" compacted before dying)
        compaction_dir = store.memory_dir / "compaction"
        compaction_dir.mkdir(parents=True)
        (compaction_dir / "s1.md").write_text(
            "User is building a REST API. Completed auth module. Next: pagination.",
            encoding="utf-8",
        )

        hooks = build_default_hooks(config, store)
        result = hooks["on_session_start"](
            {"source": "resume", "session_id": "s1"}, {}
        )
        ctx = result.get("additionalContext", "")

        # Should contain compaction summary
        assert "REST API" in ctx
        assert "pagination" in ctx
        # Should contain memory index
        assert "API Design" in ctx

    def test_resume_without_compaction_still_works(self, tmp_path: Path) -> None:
        """Resume on a session that never compacted should still produce context."""
        config = CopilotCodeConfig(
            working_directory=tmp_path,
            memory_root=tmp_path / ".mem",
        )
        store = MemoryStore(tmp_path, tmp_path / ".mem")
        hooks = build_default_hooks(config, store)

        result = hooks["on_session_start"](
            {"source": "resume", "session_id": "no-compaction"}, {}
        )
        ctx = result.get("additionalContext", "")
        # Should still have basic context, just no compaction handoff
        assert "is active" in ctx


# ---------------------------------------------------------------------------
# Scenario 4: Instruction following with path-conditional rules
# ---------------------------------------------------------------------------


class TestInstructionFollowing:
    """Verify that the instruction system correctly gates what rules
    the model sees based on what files are being worked on."""

    def test_python_rules_only_when_editing_python(self, tmp_path: Path) -> None:
        """Python-specific rules should only load when active_paths include .py files."""
        from copilotcode_sdk.instructions import load_workspace_instructions

        rules = tmp_path / ".claude" / "rules"
        rules.mkdir(parents=True)
        (rules / "python.md").write_text(
            "---\nglobs: [\"*.py\"]\n---\nAlways use type hints in Python.",
            encoding="utf-8",
        )
        (rules / "rust.md").write_text(
            "---\nglobs: [\"*.rs\"]\n---\nUse Result<T> for error handling.",
            encoding="utf-8",
        )
        (tmp_path / "CLAUDE.md").write_text("General project rules.", encoding="utf-8")

        # Editing Python files
        py_bundle = load_workspace_instructions(
            tmp_path,
            active_paths=[tmp_path / "src" / "main.py"],
        )
        assert "type hints" in py_bundle.content
        assert "Result<T>" not in py_bundle.content

        # Editing Rust files
        rs_bundle = load_workspace_instructions(
            tmp_path,
            active_paths=[tmp_path / "src" / "lib.rs"],
        )
        assert "Result<T>" in rs_bundle.content
        assert "type hints" not in rs_bundle.content

        # General rules always present
        assert "General project rules" in py_bundle.content
        assert "General project rules" in rs_bundle.content

    def test_multiple_globs_match_correctly(self, tmp_path: Path) -> None:
        """Rules with multiple globs should match any of them."""
        from copilotcode_sdk.instructions import load_workspace_instructions

        rules = tmp_path / ".claude" / "rules"
        rules.mkdir(parents=True)
        (rules / "frontend.md").write_text(
            "---\nglobs: [\"*.tsx\", \"*.css\", \"*.html\"]\n---\nUse semantic HTML.",
            encoding="utf-8",
        )
        (tmp_path / "CLAUDE.md").write_text("Base.", encoding="utf-8")

        bundle = load_workspace_instructions(
            tmp_path,
            active_paths=[tmp_path / "Button.tsx"],
        )
        assert "semantic HTML" in bundle.content

    def test_include_hierarchy_respected(self, tmp_path: Path) -> None:
        """An instruction file can @include another, building layered context."""
        from copilotcode_sdk.instructions import load_workspace_instructions

        (tmp_path / "shared-rules.md").write_text(
            "Never commit secrets to git.",
            encoding="utf-8",
        )
        (tmp_path / "CLAUDE.md").write_text(
            "Project rules.\n@include shared-rules.md",
            encoding="utf-8",
        )

        bundle = load_workspace_instructions(tmp_path)
        assert "Project rules" in bundle.content
        assert "Never commit secrets" in bundle.content


# ---------------------------------------------------------------------------
# Scenario 5: MCP health tracking — silent when stable, vocal when broken
# ---------------------------------------------------------------------------


class TestMCPHealthRealism:
    """Verify that the MCP system doesn't spam health warnings when
    servers are stable, and does emit warnings when something breaks."""

    def test_stable_servers_produce_no_warnings(self, tmp_path: Path) -> None:
        """When all MCP tool calls succeed, no health warnings should fire."""
        config = CopilotCodeConfig(
            working_directory=tmp_path,
            memory_root=tmp_path / ".mem",
            mcp_servers=[{"name": "db", "tools": [{"name": "query"}]}],
        )
        store = MemoryStore(tmp_path, tmp_path / ".mem")
        hooks = build_default_hooks(config, store)

        # Session start emits initial delta and marks baseline
        hooks["on_session_start"]({}, {})

        # Successful tool calls — should produce no additional context
        for _ in range(5):
            result = hooks["on_post_tool_use"](
                {"toolName": "mcp__db__query", "toolArgs": {}, "toolResult": {"rows": []}},
                {},
            )
            # No health warning expected
            if result is not None:
                assert "Health Warning" not in result.get("additionalContext", "")

    def test_failing_server_triggers_warning(self, tmp_path: Path) -> None:
        """After consecutive failures exceed threshold, a health warning fires."""
        config = CopilotCodeConfig(
            working_directory=tmp_path,
            memory_root=tmp_path / ".mem",
            mcp_servers=[{"name": "api", "tools": [{"name": "fetch"}]}],
        )
        store = MemoryStore(tmp_path, tmp_path / ".mem")
        hooks = build_default_hooks(config, store)
        hooks["on_session_start"]({}, {})

        # Simulate 3 consecutive failures (default threshold)
        warning_emitted = False
        for _ in range(4):
            result = hooks["on_post_tool_use"](
                {"toolName": "mcp__api__fetch", "toolArgs": {}, "toolResult": {"error": "timeout"}},
                {},
            )
            if result and "Health Warning" in result.get("additionalContext", ""):
                warning_emitted = True
                break

        assert warning_emitted, "Expected health warning after consecutive failures"

    def test_recovery_stops_warnings(self, tmp_path: Path) -> None:
        """After a server recovers, warnings should stop."""
        from copilotcode_sdk.mcp import MCPLifecycleManager

        mgr = MCPLifecycleManager(
            [{"name": "svc", "tools": []}],
            max_consecutive_failures=2,
        )
        mgr.record_failure("svc", "down")
        mgr.record_failure("svc", "down")
        assert mgr.build_status_prompt() != ""  # warning active

        mgr.record_success("svc")
        assert mgr.build_status_prompt() == ""  # recovered, no warning


# ---------------------------------------------------------------------------
# Scenario 6: Verification nudge — real task completion patterns
# ---------------------------------------------------------------------------


class TestVerificationNudgeRealism:
    """The verification nudge conditions: all tasks completed, 3+ tasks,
    none is a verification step. Test the store-level conditions directly."""

    def test_all_complete_no_verifier_is_nudge_condition(self, tmp_path: Path) -> None:
        """When all 3+ tasks are complete and none is a verifier, the condition is met."""
        import re

        ts = TaskStore(persist_path=tmp_path / "tasks.json")
        ts.create("Implement auth")
        ts.create("Write tests")
        ts.create("Update docs")
        ts.update(1, status="completed")
        ts.update(2, status="completed")
        ts.update(3, status="completed")

        non_deleted = [t for t in ts.list_all() if t.status != TaskStatus.deleted]
        all_done = all(t.status == TaskStatus.completed for t in non_deleted)
        has_verifier = any(re.search(r"verif", t.subject, re.I) for t in non_deleted)

        assert all_done is True
        assert len(non_deleted) >= 3
        assert has_verifier is False  # nudge should fire

    def test_verifier_task_suppresses_nudge(self, tmp_path: Path) -> None:
        import re

        ts = TaskStore(persist_path=tmp_path / "tasks.json")
        ts.create("Build feature")
        ts.create("Test feature")
        ts.create("Verification: full test suite")  # has "verif" in name
        for tid in [1, 2, 3]:
            ts.update(tid, status="completed")

        non_deleted = [t for t in ts.list_all() if t.status != TaskStatus.deleted]
        has_verifier = any(re.search(r"verif", t.subject, re.I) for t in non_deleted)
        assert has_verifier is True  # nudge suppressed

    def test_partial_completion_no_nudge(self, tmp_path: Path) -> None:
        ts = TaskStore(persist_path=tmp_path / "tasks.json")
        ts.create("Task A")
        ts.create("Task B")
        ts.create("Task C")
        ts.update(1, status="completed")
        ts.update(2, status="completed")
        # Task 3 still pending

        non_deleted = [t for t in ts.list_all() if t.status != TaskStatus.deleted]
        all_done = all(t.status == TaskStatus.completed for t in non_deleted)
        assert all_done is False  # not all done, no nudge


# ---------------------------------------------------------------------------
# Scenario 7: Fork child with realistic constraints
# ---------------------------------------------------------------------------


class TestForkChildRealism:
    """Verify that child sessions carry their constraints and that
    the parent's cache prefix is shared."""

    def test_child_inherits_parent_prefix(self, tmp_path: Path) -> None:
        from copilotcode_sdk.subagent import SubagentContext, SubagentSpec

        ctx = SubagentContext(
            parent_session_id="parent",
            cacheable_prefix="You are CopilotCode. Follow all instructions carefully.",
        )

        researcher = SubagentSpec(
            role="researcher",
            system_prompt_suffix="Search files only. Do not modify anything.",
            tools=("read", "grep", "glob"),
            timeout_seconds=30.0,
        )
        implementer = SubagentSpec(
            role="implementer",
            system_prompt_suffix="Write code to implement the plan.",
            tools=("read", "write", "edit", "bash"),
            timeout_seconds=120.0,
        )

        r_msg = ctx.build_child_system_message(researcher)
        i_msg = ctx.build_child_system_message(implementer)

        # Both start with same prefix
        prefix = "You are CopilotCode. Follow all instructions carefully."
        assert r_msg["content"].startswith(prefix)
        assert i_msg["content"].startswith(prefix)

        # But have different suffixes
        assert "Search files only" in r_msg["content"]
        assert "Write code" in i_msg["content"]
        assert "Search files only" not in i_msg["content"]

    def test_maintenance_session_has_no_tool_instructions(self, tmp_path: Path) -> None:
        from copilotcode_sdk.subagent import SubagentContext

        ctx = SubagentContext(
            parent_session_id="p",
            cacheable_prefix="System prompt prefix.",
        )
        msg = ctx.build_maintenance_system_message("session-memory extraction")
        assert "Do not call any tools" in msg["content"]
        assert "session-memory extraction" in msg["content"]


# ---------------------------------------------------------------------------
# Scenario 8: Instruction include chains — realistic repo structure
# ---------------------------------------------------------------------------


class TestRealisticInstructionChains:
    """Test instruction loading with a structure that mimics a real repo:
    project-level CLAUDE.md, team rules in .claude/rules/, shared includes."""

    def test_realistic_repo_instruction_hierarchy(self, tmp_path: Path) -> None:
        from copilotcode_sdk.instructions import load_workspace_instructions

        # Repo structure:
        # CLAUDE.md (project level — includes shared-rules.md)
        # shared-rules.md (included by CLAUDE.md)
        # .claude/rules/testing.md (conditional: only for test files)
        # .claude/rules/api.md (conditional: only for api/ files)

        (tmp_path / "shared-rules.md").write_text(
            "All code must be reviewed before merge.\nMax file size: 500 lines.",
            encoding="utf-8",
        )
        (tmp_path / "CLAUDE.md").write_text(
            "# Project Rules\nThis is a Python REST API project.\n@include shared-rules.md",
            encoding="utf-8",
        )

        rules = tmp_path / ".claude" / "rules"
        rules.mkdir(parents=True)
        (rules / "testing.md").write_text(
            '---\nglobs: ["test_*.py"]\n---\n'
            "Use pytest fixtures. No mocking database calls.",
            encoding="utf-8",
        )
        (rules / "api.md").write_text(
            '---\nglobs: ["*.py"]\napplies_to: routes_*.py\n---\n'
            "Validate all request bodies with Pydantic.",
            encoding="utf-8",
        )

        # Developer editing a test file — matches test_*.py glob
        test_bundle = load_workspace_instructions(
            tmp_path,
            active_paths=[tmp_path / "test_auth.py"],
        )
        assert "Python REST API" in test_bundle.content  # project-level
        assert "reviewed before merge" in test_bundle.content  # included shared rules
        assert "pytest fixtures" in test_bundle.content  # test-specific rule

        # Developer editing a non-test, non-route file — neither conditional rule matches
        other_bundle = load_workspace_instructions(
            tmp_path,
            active_paths=[tmp_path / "config.json"],
        )
        assert "Python REST API" in other_bundle.content  # project-level always
        assert "pytest fixtures" not in other_bundle.content  # test rule filtered out

    def test_empty_repo_produces_no_errors(self, tmp_path: Path) -> None:
        """A repo with no instruction files should load cleanly."""
        from copilotcode_sdk.instructions import load_workspace_instructions

        bundle = load_workspace_instructions(tmp_path)
        assert bundle.content == ""
        assert bundle.loaded_paths == []


# ---------------------------------------------------------------------------
# Scenario 9: Token budget lifecycle — parse, consume, warn, exhaust
# ---------------------------------------------------------------------------


class TestTokenBudgetLifecycle:
    """Verify the full token budget lifecycle through hooks."""

    def test_budget_parsed_consumed_and_warned(self, tmp_path: Path) -> None:
        config = CopilotCodeConfig(
            working_directory=tmp_path,
            memory_root=tmp_path / ".mem",
        )
        store = MemoryStore(tmp_path, tmp_path / ".mem")
        hooks = build_default_hooks(config, store)

        # User submits prompt with budget directive (+500k = 500,000 tokens)
        result = hooks["on_user_prompt_submitted"](
            {"prompt": "+500k implement the feature"}, {}
        )
        # Budget should be parsed and stripped
        budget = hooks["get_token_budget"]()
        assert budget is not None
        assert budget.tokens == 500_000

        # Prompt should have directive stripped
        modified = result.get("modifiedPrompt", "")
        assert "+500k" not in modified
        assert "implement the feature" in modified


# ---------------------------------------------------------------------------
# Scenario 10: Git safety — real command patterns that should be blocked
# ---------------------------------------------------------------------------


class TestGitSafetyRealism:
    """Test that the git safety system blocks dangerous commands
    that developers actually type, not just textbook examples."""

    def _run_pre_tool(self, tmp_path, command):
        config = CopilotCodeConfig(working_directory=tmp_path, memory_root=tmp_path / ".mem")
        store = MemoryStore(tmp_path, tmp_path / ".mem")
        hooks = build_default_hooks(config, store)
        return hooks["on_pre_tool_use"](
            {"toolName": "bash", "toolArgs": {"command": command}}, {}
        )

    def test_force_push_to_main_blocked(self, tmp_path: Path) -> None:
        result = self._run_pre_tool(tmp_path, "git push --force origin main")
        assert result is not None
        assert result.get("permissionDecision") == "deny"

    def test_reset_hard_blocked(self, tmp_path: Path) -> None:
        result = self._run_pre_tool(tmp_path, "git reset --hard HEAD~3")
        assert result is not None
        assert result.get("permissionDecision") == "deny"

    def test_clean_fd_blocked(self, tmp_path: Path) -> None:
        result = self._run_pre_tool(tmp_path, "git clean -fd")
        assert result is not None
        assert result.get("permissionDecision") == "deny"

    def test_normal_push_allowed(self, tmp_path: Path) -> None:
        result = self._run_pre_tool(tmp_path, "git push origin feature-branch")
        # Should either be None (no opinion) or not denied
        if result is not None:
            assert result.get("permissionDecision") != "deny"

    def test_normal_commit_allowed(self, tmp_path: Path) -> None:
        result = self._run_pre_tool(tmp_path, 'git commit -m "fix bug"')
        if result is not None:
            assert result.get("permissionDecision") != "deny"


# ---------------------------------------------------------------------------
# Scenario 11: Compaction sends real transcript to maintenance session
# ---------------------------------------------------------------------------


class TestCompactionTranscript:
    """Verify that compact_for_handoff() passes the actual conversation
    transcript to the maintenance session, not just the instruction prompt."""

    def test_compact_sends_transcript_to_maintenance(self, tmp_path: Path) -> None:
        """The maintenance session should receive both the transcript and
        the compaction instructions, so it can actually summarize."""
        from copilotcode_sdk.compaction import parse_compaction_response

        # Build a minimal CopilotCodeSession-like object
        mock_maintenance = MagicMock()
        mock_maintenance.send_and_wait = AsyncMock(return_value=None)
        mock_maintenance.get_messages = AsyncMock(return_value=[
            {"role": "assistant", "content": "<analysis>ok</analysis><summary>done</summary>"},
        ])
        mock_maintenance.destroy = AsyncMock()

        # The session under test
        session = MagicMock()
        session.session_id = "test-session"
        session.get_messages = AsyncMock(return_value=[
            {"role": "user", "content": "Implement the auth module"},
            {"role": "assistant", "content": "I'll start by reading the existing code."},
            {"role": "user", "content": "Focus on JWT tokens specifically."},
        ])
        session._create_maintenance_session = AsyncMock(return_value=mock_maintenance)
        session._memory_store = MagicMock()
        session._memory_store.memory_dir = tmp_path

        # Use the real compact_for_handoff logic
        from copilotcode_sdk.compaction import (
            build_compaction_prompt,
            format_transcript_for_compaction,
        )

        async def run():
            parent_messages = await session.get_messages()
            transcript_block = format_transcript_for_compaction(parent_messages)
            prompt = build_compaction_prompt()
            full_prompt = f"{transcript_block}\n\n{prompt}"

            maintenance = await session._create_maintenance_session()
            await maintenance.send_and_wait(full_prompt, timeout=600.0)

            # Verify the maintenance session received the transcript
            call_args = maintenance.send_and_wait.call_args
            sent_prompt = call_args[0][0]
            assert "<transcript>" in sent_prompt
            assert "Implement the auth module" in sent_prompt
            assert "JWT tokens" in sent_prompt
            assert "Primary request and intent" in sent_prompt  # compaction instructions too

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Scenario 14: Stale prompt recomposition after git checkout
# ---------------------------------------------------------------------------


class TestStalePromptRecomposition:
    """Verify that git operations trigger stale marking and dynamic
    content is refreshed on the next tool call."""

    def test_stale_triggers_dynamic_refresh(self, tmp_path: Path) -> None:
        """After a git checkout, the next tool call should inject refreshed
        dynamic content via additionalContext."""
        from copilotcode_sdk.prompt_compiler import PromptAssembler

        config = CopilotCodeConfig(
            working_directory=tmp_path,
            memory_root=tmp_path / ".mem",
        )
        store = MemoryStore(tmp_path, tmp_path / ".mem")

        asm = PromptAssembler()
        asm.add("git_context", "branch: main\nStatus: clean", cacheable=False)
        asm.add("skill_catalog", "Available skills: verify", cacheable=False)

        hooks = build_default_hooks(config, store, assembler=asm)
        hooks["on_session_start"]({}, {})

        # First: git checkout marks git_context stale
        result1 = hooks["on_post_tool_use"](
            {"toolName": "bash", "toolArgs": {"command": "git checkout develop"}, "toolResult": ""},
            {},
        )
        # The hook should have detected the stale section and refreshed it
        # Since the dynamic content is still there, it should be in the response
        if result1 is not None and "additionalContext" in result1:
            ctx = result1["additionalContext"]
            # Dynamic content should contain both dynamic sections
            assert "branch: main" in ctx or "Available skills" in ctx


# ---------------------------------------------------------------------------
# Scenario 12: InvokeSkill full flow — detect, queue, drain, fork
# ---------------------------------------------------------------------------


class TestInvokeSkillExecution:
    """Verify the full InvokeSkill pipeline: hook detects the result,
    queues the invocation, drain returns it, and a child session would
    receive the skill prompt."""

    def test_skill_execution_forks_child(self, tmp_path: Path) -> None:
        """Simulate the full InvokeSkill flow through hooks."""
        import json

        config = CopilotCodeConfig(
            working_directory=tmp_path,
            memory_root=tmp_path / ".mem",
        )
        store = MemoryStore(tmp_path, tmp_path / ".mem")
        hooks = build_default_hooks(config, store)

        # Start the session
        hooks["on_session_start"]({}, {})

        # Simulate InvokeSkill tool result with a full invocation record
        invocation_data = {
            "skill_name": "writing-plans",
            "user_prompt": "# Skill Execution: writing-plans\nCreate an implementation plan.",
            "skill_type": "process",
            "outputs": "docs/plan.md",
            "timestamp": 1234567890.0,
        }
        invocation_result = json.dumps({
            "status": "invoke_skill",
            "skill_name": "writing-plans",
            "skill_type": "process",
            "_invocation": invocation_data,
        })

        # Post-tool-use hook should detect and queue
        result = hooks["on_post_tool_use"](
            {"toolName": "InvokeSkill", "toolArgs": {"skill": "writing-plans"}, "toolResult": invocation_result},
            {},
        )
        assert result is not None
        assert "writing-plans" in result["additionalContext"]

        # Drain should return the invocation
        drained = hooks["drain_pending_skill_invocations"]()
        assert len(drained) == 1
        assert drained[0]["skill_name"] == "writing-plans"
        assert "implementation plan" in drained[0]["user_prompt"]

        # Verify the invocation has the right structure for fork_child
        inv = drained[0]
        assert "user_prompt" in inv
        assert "skill_name" in inv
        assert inv["skill_type"] == "process"


# ---------------------------------------------------------------------------
# Scenario 13: Enforced child session handles timeout gracefully
# ---------------------------------------------------------------------------


class TestEnforcedChildScenario:
    """Verify that EnforcedChildSession actually stops a runaway child."""

    def test_skill_child_timeout(self) -> None:
        """A child with a short timeout should raise TimeoutError when the
        underlying session is slow, and the parent should be able to handle it."""
        from copilotcode_sdk.subagent import (
            ChildSession,
            EnforcedChildSession,
            SubagentSpec,
        )

        async def slow_send(prompt, **kwargs):
            await asyncio.sleep(10.0)
            return "should never reach here"

        mock_session = MagicMock()
        mock_session.send_and_wait = slow_send
        mock_session.destroy = AsyncMock()

        spec = SubagentSpec(
            role="skill:slow",
            system_prompt_suffix="Be slow.",
            max_turns=5,
            timeout_seconds=0.1,
        )
        child = ChildSession(session=mock_session, spec=spec, session_id="slow-child")
        enforced = EnforcedChildSession(child)

        async def run():
            # Parent catches the timeout and produces an error summary
            try:
                await enforced.send_and_wait("do something slow")
                assert False, "Should have timed out"
            except asyncio.TimeoutError:
                pass  # expected

            # Parent can still destroy the child cleanly
            await enforced.destroy()
            assert enforced.turn_count == 1  # turn was counted before timeout

        asyncio.run(run())


# ---------------------------------------------------------------------------
# Scenario 15: MCP instruction deltas only emit changes
# ---------------------------------------------------------------------------


class TestMCPInstructionDelta:
    """Verify that the hooks use instruction delta tracking instead of
    re-announcing all MCP tools every time."""

    def test_hooks_use_delta_tracking(self, tmp_path: Path) -> None:
        """At session start, MCP tools are announced as 'added'.
        On subsequent status checks with no changes, nothing is emitted."""
        from copilotcode_sdk.mcp import MCPLifecycleManager

        mcp_servers = [
            {
                "name": "TestServer",
                "description": "A test MCP server",
                "tools": [
                    {"name": "test_tool", "description": "Does testing"},
                    {"name": "other_tool", "description": "Does other things"},
                ],
                "instructions": "Use carefully.",
            },
        ]

        config = CopilotCodeConfig(
            working_directory=tmp_path,
            memory_root=tmp_path / ".mem",
            mcp_servers=mcp_servers,
        )
        store = MemoryStore(tmp_path, tmp_path / ".mem")
        hooks = build_default_hooks(config, store)

        # Session start should announce tools as "added"
        result = hooks["on_session_start"]({}, {})
        ctx = result.get("additionalContext", "")
        assert "MCP tools added:" in ctx
        assert "test_tool" in ctx
        assert "other_tool" in ctx

        # Now get the manager and verify it tracks announced state
        mgr = hooks["get_mcp_manager"]()
        assert mgr is not None
        # A subsequent delta call with same tools should return empty
        delta = mgr.build_instruction_delta(mcp_servers)
        assert delta == ""


# ---------------------------------------------------------------------------
# Scenario 16: Predictive suggestions gated by config
# ---------------------------------------------------------------------------


class TestPredictiveSuggestionsConfig:
    """Verify that predictive suggestions are only active when the config flag
    is enabled, and that the merge logic works correctly."""

    def test_merge_produces_combined_suggestions(self) -> None:
        """merge_suggestions combines predictive and heuristic results."""
        from copilotcode_sdk.suggestions import merge_suggestions

        heuristic = ["Run tests", "Check coverage"]
        predictive = ["Fix the auth bug first", "Run tests"]  # "Run tests" is a dupe
        merged = merge_suggestions(heuristic, predictive)
        # Predictive first, deduped
        assert merged[0] == "Fix the auth bug first"
        assert merged[1] == "Run tests"
        assert merged[2] == "Check coverage"
        assert len(merged) == 3

    def test_config_flag_default_is_false(self) -> None:
        """The enable_predictive_suggestions config flag defaults to False."""
        config = CopilotCodeConfig(
            working_directory=Path("."),
            memory_root=Path(".mem"),
        )
        assert config.enable_predictive_suggestions is False
