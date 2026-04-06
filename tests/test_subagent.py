"""Tests for copilotcode_sdk.subagent module."""
from __future__ import annotations

import pytest

from copilotcode_sdk.subagent import SubagentContext, SubagentSpec, build_subagent_context


class TestSubagentSpec:
    def test_defaults(self) -> None:
        spec = SubagentSpec(role="researcher", system_prompt_suffix="You are a researcher.")
        assert spec.role == "researcher"
        assert spec.tools == ()
        assert spec.max_turns == 0
        assert spec.timeout_seconds == 3600.0

    def test_frozen(self) -> None:
        spec = SubagentSpec(role="r", system_prompt_suffix="s")
        try:
            spec.role = "other"  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass


class TestSubagentContext:
    def test_build_child_system_message(self) -> None:
        ctx = SubagentContext(
            parent_session_id="parent-1",
            cacheable_prefix="You are a helpful assistant.",
        )
        spec = SubagentSpec(role="researcher", system_prompt_suffix="Search only.")
        msg = ctx.build_child_system_message(spec)
        assert msg["mode"] == "append"
        assert msg["content"].startswith("You are a helpful assistant.")
        assert "Search only." in msg["content"]

    def test_child_message_with_empty_suffix(self) -> None:
        ctx = SubagentContext(
            parent_session_id="p",
            cacheable_prefix="Base prompt.",
        )
        spec = SubagentSpec(role="worker", system_prompt_suffix="")
        msg = ctx.build_child_system_message(spec)
        assert msg["content"] == "Base prompt."

    def test_maintenance_system_message(self) -> None:
        ctx = SubagentContext(
            parent_session_id="p",
            cacheable_prefix="Prefix.",
        )
        msg = ctx.build_maintenance_system_message("extraction agent")
        assert msg["mode"] == "append"
        assert "Prefix." in msg["content"]
        assert "extraction agent" in msg["content"]
        assert "Do not call any tools" in msg["content"]

    def test_maintenance_default_description(self) -> None:
        ctx = SubagentContext(parent_session_id="p", cacheable_prefix="P.")
        msg = ctx.build_maintenance_system_message()
        assert "session-memory maintenance agent" in msg["content"]

    def test_register_child(self) -> None:
        ctx = SubagentContext(parent_session_id="p", cacheable_prefix="P.")
        assert ctx.children == []
        ctx.register_child("child-1")
        ctx.register_child("child-2")
        assert ctx.children == ["child-1", "child-2"]

    def test_cache_prefix_shared_across_children(self) -> None:
        """All children get the same cacheable prefix as the parent."""
        ctx = SubagentContext(parent_session_id="p", cacheable_prefix="SHARED_PREFIX")
        spec_a = SubagentSpec(role="a", system_prompt_suffix="Suffix A")
        spec_b = SubagentSpec(role="b", system_prompt_suffix="Suffix B")
        msg_a = ctx.build_child_system_message(spec_a)
        msg_b = ctx.build_child_system_message(spec_b)
        # Both start with the same prefix
        assert msg_a["content"].startswith("SHARED_PREFIX")
        assert msg_b["content"].startswith("SHARED_PREFIX")
        # But differ in suffix
        assert "Suffix A" in msg_a["content"]
        assert "Suffix B" in msg_b["content"]
        assert "Suffix A" not in msg_b["content"]


class TestBuildSubagentContext:
    def test_creates_context(self) -> None:
        ctx = build_subagent_context("session-42", "cached prefix text")
        assert ctx.parent_session_id == "session-42"
        assert ctx.cacheable_prefix == "cached prefix text"
        assert ctx.children == []

    def test_context_is_mutable(self) -> None:
        ctx = build_subagent_context("s1", "p")
        ctx.register_child("c1")
        assert "c1" in ctx.children


# ---------------------------------------------------------------------------
# EnforcedChildSession tests
# ---------------------------------------------------------------------------


class TestEnforcedChildSession:
    def _make_enforced(
        self, *, max_turns: int = 0, timeout_seconds: float = 3600.0, send_delay: float = 0.0,
    ):
        from unittest.mock import AsyncMock, MagicMock
        from copilotcode_sdk.subagent import ChildSession, EnforcedChildSession

        mock_session = MagicMock()
        if send_delay > 0:
            import asyncio
            async def slow_send(prompt):
                await asyncio.sleep(send_delay)
                return "done"
            mock_session.send_and_wait = slow_send
        else:
            mock_session.send_and_wait = AsyncMock(return_value="ok")
        mock_session.destroy = AsyncMock()

        spec = SubagentSpec(
            role="test",
            system_prompt_suffix="test suffix",
            max_turns=max_turns,
            timeout_seconds=timeout_seconds,
        )
        child = ChildSession(session=mock_session, spec=spec, session_id="child-1")
        return EnforcedChildSession(child)

    def test_enforced_respects_max_turns(self) -> None:
        import asyncio
        from copilotcode_sdk.subagent import MaxTurnsExceeded

        enforced = self._make_enforced(max_turns=2)

        async def run():
            await enforced.send_and_wait("turn 1")
            await enforced.send_and_wait("turn 2")
            with pytest.raises(MaxTurnsExceeded) as exc_info:
                await enforced.send_and_wait("turn 3")
            assert exc_info.value.max_turns == 2
            assert exc_info.value.actual_turns == 2

        asyncio.run(run())

    def test_enforced_respects_timeout(self) -> None:
        import asyncio

        enforced = self._make_enforced(timeout_seconds=0.05, send_delay=1.0)

        async def run():
            with pytest.raises(asyncio.TimeoutError):
                await enforced.send_and_wait("slow prompt")

        asyncio.run(run())

    def test_enforced_allows_within_limits(self) -> None:
        import asyncio

        enforced = self._make_enforced(max_turns=5)

        async def run():
            result = await enforced.send_and_wait("turn 1")
            assert result == "ok"
            assert enforced.turn_count == 1

        asyncio.run(run())

    def test_enforced_unlimited_turns(self) -> None:
        import asyncio

        enforced = self._make_enforced(max_turns=0)  # 0 = unlimited

        async def run():
            for i in range(10):
                await enforced.send_and_wait(f"turn {i}")
            assert enforced.turn_count == 10

        asyncio.run(run())
