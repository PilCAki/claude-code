from __future__ import annotations

from copilotcode_sdk.mcp import (
    build_mcp_prompt_section,
    build_mcp_delta,
    validate_mcp_server_config,
)


def _sample_servers() -> list[dict]:
    return [
        {
            "name": "GitHub",
            "description": "GitHub API integration for issues and PRs.",
            "tools": [
                {"name": "gh_search", "description": "Search GitHub issues."},
                {"name": "gh_create_pr", "description": "Create a pull request."},
            ],
            "instructions": "Authenticate with `gh auth login` first.",
        },
        {
            "name": "Slack",
            "description": "Post messages to Slack channels.",
            "auth_required": True,
            "tools": [
                {"name": "slack_post", "description": "Post a message."},
            ],
        },
    ]


class TestBuildMcpPromptSection:
    def test_empty_servers_returns_empty(self) -> None:
        assert build_mcp_prompt_section([]) == ""

    def test_includes_server_names(self) -> None:
        section = build_mcp_prompt_section(_sample_servers())
        assert "### GitHub" in section
        assert "### Slack" in section

    def test_includes_tool_names(self) -> None:
        section = build_mcp_prompt_section(_sample_servers())
        assert "`gh_search`" in section
        assert "`gh_create_pr`" in section
        assert "`slack_post`" in section

    def test_includes_descriptions(self) -> None:
        section = build_mcp_prompt_section(_sample_servers())
        assert "GitHub API integration" in section
        assert "Search GitHub issues" in section

    def test_includes_auth_warning(self) -> None:
        section = build_mcp_prompt_section(_sample_servers())
        assert "authentication" in section.lower()

    def test_includes_instructions(self) -> None:
        section = build_mcp_prompt_section(_sample_servers())
        assert "gh auth login" in section

    def test_server_without_tools(self) -> None:
        servers = [{"name": "Empty", "description": "No tools."}]
        section = build_mcp_prompt_section(servers)
        assert "### Empty" in section
        assert "No tools." in section


class TestBuildMcpDelta:
    def test_empty_servers(self) -> None:
        assert build_mcp_delta([]) == ""

    def test_lists_all_tools(self) -> None:
        delta = build_mcp_delta(_sample_servers())
        assert "MCP context update" in delta
        assert "gh_search" in delta
        assert "slack_post" in delta

    def test_active_tools_filter(self) -> None:
        delta = build_mcp_delta(
            _sample_servers(),
            active_tools=["gh_search"],
        )
        assert "gh_search" in delta
        assert "gh_create_pr" not in delta
        assert "slack_post" not in delta

    def test_active_tools_empty_returns_empty(self) -> None:
        delta = build_mcp_delta(
            _sample_servers(),
            active_tools=["nonexistent"],
        )
        assert delta == ""


class TestValidateMcpServerConfig:
    def test_valid_config(self) -> None:
        warnings = validate_mcp_server_config(
            {"name": "Test", "description": "A test server."}
        )
        assert warnings == []

    def test_missing_name(self) -> None:
        warnings = validate_mcp_server_config({"description": "No name."})
        assert any("name" in w for w in warnings)

    def test_missing_description(self) -> None:
        warnings = validate_mcp_server_config({"name": "Test"})
        assert any("description" in w for w in warnings)

    def test_tool_missing_name(self) -> None:
        warnings = validate_mcp_server_config(
            {"name": "Test", "description": "D", "tools": [{"description": "No name"}]}
        )
        assert any("tool" in w.lower() for w in warnings)
