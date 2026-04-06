"""MCP (Model Context Protocol) instruction delta handling.

Builds prompt sections that describe available MCP servers and their tools
so the agent knows what external capabilities are available and how to use them.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


# ---------------------------------------------------------------------------
# Server lifecycle management
# ---------------------------------------------------------------------------


class MCPServerStatus(Enum):
    """Health state of an MCP server."""
    connected = "connected"
    disconnected = "disconnected"
    unhealthy = "unhealthy"
    unknown = "unknown"


@dataclass
class MCPServerState:
    """Runtime state for a single MCP server."""
    name: str
    config: Mapping[str, Any]
    status: MCPServerStatus = MCPServerStatus.unknown
    last_health_check: float = 0.0
    consecutive_failures: int = 0
    error_message: str = ""


class MCPLifecycleManager:
    """Manages MCP server connections, health checks, and status tracking.

    This is a local state tracker — it does not manage actual network
    connections. The underlying Copilot SDK handles transport. This class
    tracks health status and provides prompt-level awareness of server state.
    """

    def __init__(
        self,
        servers: Sequence[Mapping[str, Any]],
        *,
        health_check_interval: float = 60.0,
        max_consecutive_failures: int = 3,
    ) -> None:
        self._health_check_interval = health_check_interval
        self._max_consecutive_failures = max_consecutive_failures
        self._servers: dict[str, MCPServerState] = {}
        for server in servers:
            name = server.get("name", "unnamed")
            self._servers[name] = MCPServerState(name=name, config=server)

    @property
    def servers(self) -> dict[str, MCPServerState]:
        return dict(self._servers)

    def mark_connected(self, name: str) -> None:
        """Mark a server as connected."""
        if name in self._servers:
            self._servers[name].status = MCPServerStatus.connected
            self._servers[name].consecutive_failures = 0
            self._servers[name].error_message = ""

    def mark_disconnected(self, name: str, *, error: str = "") -> None:
        """Mark a server as disconnected."""
        if name in self._servers:
            self._servers[name].status = MCPServerStatus.disconnected
            self._servers[name].error_message = error

    def record_failure(self, name: str, error: str = "") -> None:
        """Record a health check failure."""
        if name not in self._servers:
            return
        state = self._servers[name]
        state.consecutive_failures += 1
        state.error_message = error
        state.last_health_check = time.monotonic()
        if state.consecutive_failures >= self._max_consecutive_failures:
            state.status = MCPServerStatus.unhealthy

    def record_success(self, name: str) -> None:
        """Record a successful health check."""
        if name not in self._servers:
            return
        state = self._servers[name]
        state.consecutive_failures = 0
        state.status = MCPServerStatus.connected
        state.last_health_check = time.monotonic()
        state.error_message = ""

    def needs_health_check(self, name: str) -> bool:
        """Check if a server is due for a health check."""
        if name not in self._servers:
            return False
        state = self._servers[name]
        if state.status == MCPServerStatus.disconnected:
            return False
        elapsed = time.monotonic() - state.last_health_check
        return elapsed >= self._health_check_interval

    def healthy_servers(self) -> list[str]:
        """Return names of servers that are connected and healthy."""
        return [
            name for name, state in self._servers.items()
            if state.status == MCPServerStatus.connected
        ]

    def unhealthy_servers(self) -> list[MCPServerState]:
        """Return states of unhealthy servers."""
        return [
            state for state in self._servers.values()
            if state.status == MCPServerStatus.unhealthy
        ]

    def snapshot_state(self) -> dict[str, MCPServerStatus]:
        """Return a snapshot of current server statuses."""
        return {name: s.status for name, s in self._servers.items()}

    def mark_delta_emitted(self) -> None:
        """Record the current state as the last-emitted baseline for delta tracking."""
        self._last_emitted: dict[str, MCPServerStatus] = self.snapshot_state()

    def has_changes(self) -> bool:
        """Whether any server status changed since the last emitted delta."""
        if not hasattr(self, "_last_emitted"):
            return True
        return self.snapshot_state() != self._last_emitted

    def changed_servers(self) -> list[str]:
        """Return names of servers whose status changed since last delta."""
        if not hasattr(self, "_last_emitted"):
            return list(self._servers.keys())
        current = self.snapshot_state()
        return [n for n, s in current.items() if self._last_emitted.get(n) != s]

    def build_instruction_delta(
        self,
        servers: Sequence[Mapping[str, Any]],
    ) -> str:
        """Build an instruction delta describing added/removed tools and changed instructions.

        Compares current server configs against previously-announced state.
        Returns formatted text blocks for changes, or empty string if nothing changed.
        Updates internal announced state after computing the delta.
        """
        if not hasattr(self, "_announced_tools"):
            self._announced_tools: dict[str, set[str]] = {}
        if not hasattr(self, "_announced_instructions"):
            self._announced_instructions: dict[str, str] = {}

        added_parts: list[str] = []
        removed_parts: list[str] = []
        instruction_parts: list[str] = []

        current_tools: dict[str, set[str]] = {}
        current_instructions: dict[str, str] = {}
        tool_descriptions: dict[str, str] = {}  # tool_name → description

        for server in servers:
            name = server.get("name", "unnamed")
            tools = server.get("tools", [])
            tool_names = {t.get("name", "?") for t in tools}
            current_tools[name] = tool_names
            for t in tools:
                tn = t.get("name", "?")
                td = t.get("description", "")
                if td:
                    tool_descriptions[tn] = td

            instr = server.get("instructions", "")
            current_instructions[name] = instr

            prev_tools = self._announced_tools.get(name, set())
            new_tools = tool_names - prev_tools
            gone_tools = prev_tools - tool_names

            for tn in sorted(new_tools):
                desc = tool_descriptions.get(tn, "")
                added_parts.append(f"- `{tn}`{' — ' + desc if desc else ''} (server: {name})")
            for tn in sorted(gone_tools):
                removed_parts.append(f"- `{tn}` (server: {name})")

            prev_instr = self._announced_instructions.get(name, "")
            if instr and instr != prev_instr:
                instruction_parts.append(f"- **{name}**: {instr}")

        # Check for entirely removed servers
        for name in set(self._announced_tools) - {s.get("name", "unnamed") for s in servers}:
            for tn in sorted(self._announced_tools[name]):
                removed_parts.append(f"- `{tn}` (server: {name})")

        # Update announced state
        self._announced_tools = current_tools
        self._announced_instructions = current_instructions

        # Build output
        blocks: list[str] = []
        if added_parts:
            blocks.append("**MCP tools added:**\n" + "\n".join(added_parts))
        if removed_parts:
            blocks.append("**MCP tools removed:**\n" + "\n".join(removed_parts))
        if instruction_parts:
            blocks.append("**MCP instructions changed:**\n" + "\n".join(instruction_parts))

        return "\n\n".join(blocks)

    def build_status_prompt(self) -> str:
        """Build a prompt section describing current MCP server health."""
        unhealthy = self.unhealthy_servers()
        if not unhealthy:
            return ""
        lines = ["**MCP Server Health Warning:**"]
        for state in unhealthy:
            lines.append(
                f"- **{state.name}** is unhealthy "
                f"({state.consecutive_failures} consecutive failures"
                f"{': ' + state.error_message if state.error_message else ''}). "
                "Tools from this server may be unavailable."
            )
        return "\n".join(lines)


def build_mcp_prompt_section(
    servers: Sequence[Mapping[str, Any]],
) -> str:
    """Build a prompt section describing available MCP servers and tools.

    Each server entry should have at minimum:
      - ``name``: display name of the server
      - ``description``: what the server provides

    Optional fields:
      - ``tools``: list of tool descriptors (each with ``name`` and ``description``)
      - ``instructions``: extra usage instructions for this server
      - ``auth_required``: whether authentication is needed
    """
    if not servers:
        return ""

    lines = [
        "## MCP Servers",
        "",
        "The following MCP (Model Context Protocol) servers are available. "
        "Use their tools when a task matches the server's domain.",
        "",
    ]

    for server in servers:
        name = server.get("name", "unnamed")
        desc = server.get("description", "")
        auth = server.get("auth_required", False)

        lines.append(f"### {name}")
        if desc:
            lines.append(desc)
        if auth:
            lines.append("**Note:** This server requires authentication.")

        tools = server.get("tools", [])
        if tools:
            lines.append("")
            lines.append("Available tools:")
            for tool in tools:
                tool_name = tool.get("name", "unknown")
                tool_desc = tool.get("description", "")
                if tool_desc:
                    lines.append(f"- `{tool_name}` — {tool_desc}")
                else:
                    lines.append(f"- `{tool_name}`")

        instructions = server.get("instructions", "")
        if instructions:
            lines.append("")
            lines.append(instructions)

        lines.append("")

    return "\n".join(lines).strip()


def build_mcp_delta(
    servers: Sequence[Mapping[str, Any]],
    *,
    active_tools: Sequence[str] | None = None,
) -> str:
    """Build a delta prompt describing MCP context changes.

    Use this for per-turn injection when MCP server state changes
    (e.g., new tools become available or a server disconnects).

    If *active_tools* is provided, only those tools are described as available.
    """
    if not servers:
        return ""

    active_set = set(active_tools) if active_tools is not None else None

    parts: list[str] = []
    for server in servers:
        name = server.get("name", "unnamed")
        tools = server.get("tools", [])

        if active_set is not None:
            tools = [t for t in tools if t.get("name") in active_set]

        if not tools:
            continue

        tool_names = [t.get("name", "?") for t in tools]
        parts.append(f"**{name}**: {', '.join(tool_names)}")

    if not parts:
        return ""

    return (
        "**MCP context update.** The following MCP tools are currently available:\n"
        + "\n".join(f"- {p}" for p in parts)
    )


def validate_mcp_server_config(server: Mapping[str, Any]) -> list[str]:
    """Validate a single MCP server configuration entry.

    Returns a list of warning messages (empty if valid).
    """
    warnings: list[str] = []
    if "name" not in server:
        warnings.append("MCP server config missing 'name' field")
    if "description" not in server:
        warnings.append(
            f"MCP server '{server.get('name', 'unnamed')}' missing 'description' field"
        )
    tools = server.get("tools", [])
    for i, tool in enumerate(tools):
        if "name" not in tool:
            warnings.append(
                f"MCP server '{server.get('name', 'unnamed')}' tool #{i} missing 'name'"
            )
    return warnings
