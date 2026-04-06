"""Tests for MCP server lifecycle management."""
from __future__ import annotations

from copilotcode_sdk.mcp import (
    MCPLifecycleManager,
    MCPServerStatus,
)


def _make_manager() -> MCPLifecycleManager:
    return MCPLifecycleManager(
        [
            {"name": "ServerA", "description": "Test A", "tools": [{"name": "tool_a"}]},
            {"name": "ServerB", "description": "Test B", "tools": [{"name": "tool_b"}]},
        ],
        health_check_interval=10.0,
        max_consecutive_failures=2,
    )


def test_initial_state_is_unknown() -> None:
    mgr = _make_manager()
    assert mgr.servers["ServerA"].status == MCPServerStatus.unknown
    assert mgr.servers["ServerB"].status == MCPServerStatus.unknown


def test_mark_connected() -> None:
    mgr = _make_manager()
    mgr.mark_connected("ServerA")
    assert mgr.servers["ServerA"].status == MCPServerStatus.connected


def test_mark_disconnected() -> None:
    mgr = _make_manager()
    mgr.mark_connected("ServerA")
    mgr.mark_disconnected("ServerA", error="connection lost")
    assert mgr.servers["ServerA"].status == MCPServerStatus.disconnected
    assert mgr.servers["ServerA"].error_message == "connection lost"


def test_record_failure_transitions_to_unhealthy() -> None:
    mgr = _make_manager()
    mgr.mark_connected("ServerA")
    mgr.record_failure("ServerA", "timeout")
    assert mgr.servers["ServerA"].status == MCPServerStatus.connected  # Not yet
    mgr.record_failure("ServerA", "timeout again")
    assert mgr.servers["ServerA"].status == MCPServerStatus.unhealthy


def test_record_success_clears_failures() -> None:
    mgr = _make_manager()
    mgr.record_failure("ServerA")
    mgr.record_success("ServerA")
    assert mgr.servers["ServerA"].consecutive_failures == 0
    assert mgr.servers["ServerA"].status == MCPServerStatus.connected


def test_healthy_servers() -> None:
    mgr = _make_manager()
    mgr.mark_connected("ServerA")
    assert mgr.healthy_servers() == ["ServerA"]
    mgr.mark_connected("ServerB")
    assert sorted(mgr.healthy_servers()) == ["ServerA", "ServerB"]


def test_unhealthy_servers() -> None:
    mgr = _make_manager()
    mgr.record_failure("ServerA")
    mgr.record_failure("ServerA")  # 2 failures → unhealthy
    unhealthy = mgr.unhealthy_servers()
    assert len(unhealthy) == 1
    assert unhealthy[0].name == "ServerA"


def test_build_status_prompt_empty_when_healthy() -> None:
    mgr = _make_manager()
    mgr.mark_connected("ServerA")
    mgr.mark_connected("ServerB")
    assert mgr.build_status_prompt() == ""


def test_build_status_prompt_warns_unhealthy() -> None:
    mgr = _make_manager()
    mgr.record_failure("ServerA", "timeout")
    mgr.record_failure("ServerA", "timeout")
    prompt = mgr.build_status_prompt()
    assert "ServerA" in prompt
    assert "unhealthy" in prompt.lower()
    assert "timeout" in prompt


def test_needs_health_check_initially() -> None:
    mgr = _make_manager()
    # Unknown servers need health check
    assert mgr.needs_health_check("ServerA")


def test_needs_health_check_not_for_disconnected() -> None:
    mgr = _make_manager()
    mgr.mark_disconnected("ServerA")
    assert not mgr.needs_health_check("ServerA")


def test_unknown_server_ignored() -> None:
    mgr = _make_manager()
    # Operations on unknown server names should not crash
    mgr.mark_connected("NonExistent")
    mgr.record_failure("NonExistent")
    assert not mgr.needs_health_check("NonExistent")


# ---------------------------------------------------------------------------
# Gap 1.3: Delta state tracking
# ---------------------------------------------------------------------------


def test_snapshot_state() -> None:
    mgr = _make_manager()
    mgr.mark_connected("ServerA")
    snap = mgr.snapshot_state()
    assert snap["ServerA"] == MCPServerStatus.connected
    assert snap["ServerB"] == MCPServerStatus.unknown


def test_has_changes_true_initially() -> None:
    """Before any delta is emitted, has_changes returns True."""
    mgr = _make_manager()
    assert mgr.has_changes() is True


def test_has_changes_false_after_mark_delta() -> None:
    mgr = _make_manager()
    mgr.mark_delta_emitted()
    assert mgr.has_changes() is False


def test_has_changes_true_after_status_change() -> None:
    mgr = _make_manager()
    mgr.mark_delta_emitted()
    mgr.mark_connected("ServerA")
    assert mgr.has_changes() is True


def test_changed_servers_returns_only_changed() -> None:
    mgr = _make_manager()
    mgr.mark_connected("ServerA")
    mgr.mark_delta_emitted()
    # Only change ServerB
    mgr.mark_connected("ServerB")
    changed = mgr.changed_servers()
    assert "ServerB" in changed
    assert "ServerA" not in changed


def test_mark_delta_emitted_resets_changes() -> None:
    """After emitting delta, changes are cleared until next state change."""
    mgr = _make_manager()
    mgr.mark_connected("ServerA")
    mgr.mark_delta_emitted()
    assert mgr.has_changes() is False
    # Now change again
    mgr.mark_disconnected("ServerA")
    assert mgr.has_changes() is True
    assert "ServerA" in mgr.changed_servers()


# ---------------------------------------------------------------------------
# Instruction delta tests
# ---------------------------------------------------------------------------


_SERVERS = [
    {
        "name": "GitHub",
        "description": "GitHub API",
        "tools": [
            {"name": "gh_search", "description": "Search issues"},
            {"name": "gh_create_pr", "description": "Create PRs"},
        ],
        "instructions": "Run `gh auth login` first.",
    },
    {
        "name": "Slack",
        "description": "Slack messaging",
        "tools": [
            {"name": "slack_post", "description": "Post a message"},
        ],
        "instructions": "",
    },
]


def test_delta_first_call_announces_all() -> None:
    """First call to build_instruction_delta announces all tools as added."""
    mgr = MCPLifecycleManager(_SERVERS)
    delta = mgr.build_instruction_delta(_SERVERS)
    assert "MCP tools added:" in delta
    assert "gh_search" in delta
    assert "gh_create_pr" in delta
    assert "slack_post" in delta
    assert "MCP tools removed:" not in delta


def test_delta_no_change_returns_empty() -> None:
    """Second identical call returns empty string."""
    mgr = MCPLifecycleManager(_SERVERS)
    mgr.build_instruction_delta(_SERVERS)  # first: announces all
    delta = mgr.build_instruction_delta(_SERVERS)  # second: nothing changed
    assert delta == ""


def test_delta_detects_removed_tool() -> None:
    """Removing a tool from a server shows it in the removed block."""
    mgr = MCPLifecycleManager(_SERVERS)
    mgr.build_instruction_delta(_SERVERS)

    # Remove gh_create_pr from GitHub
    updated = [
        {
            "name": "GitHub",
            "description": "GitHub API",
            "tools": [{"name": "gh_search", "description": "Search issues"}],
            "instructions": "Run `gh auth login` first.",
        },
        _SERVERS[1],  # Slack unchanged
    ]
    delta = mgr.build_instruction_delta(updated)
    assert "MCP tools removed:" in delta
    assert "gh_create_pr" in delta
    assert "MCP tools added:" not in delta  # nothing new


def test_delta_detects_changed_instructions() -> None:
    """Changing a server's instruction text shows in the delta."""
    mgr = MCPLifecycleManager(_SERVERS)
    mgr.build_instruction_delta(_SERVERS)

    updated = [
        {
            "name": "GitHub",
            "description": "GitHub API",
            "tools": _SERVERS[0]["tools"],
            "instructions": "Use a personal access token.",
        },
        _SERVERS[1],
    ]
    delta = mgr.build_instruction_delta(updated)
    assert "MCP instructions changed:" in delta
    assert "personal access token" in delta
