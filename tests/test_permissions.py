from __future__ import annotations

from dataclasses import dataclass, field

from copilotcode_sdk import DEFAULT_BRAND
from copilotcode_sdk.permissions import (
    _command_matches_prefix,
    _extract_request_paths,
    _paths_are_allowed,
    build_permission_handler,
    normalize_shell_prefixes,
)


@dataclass
class FakePermissionRequest:
    tool_name: str
    read_only: bool = False
    path: str | None = None
    possible_paths: list[str] = field(default_factory=list)
    full_command_text: str = ""


def test_safe_policy_approves_read_only_requests(tmp_path) -> None:
    handler = build_permission_handler(
        policy="safe",
        permission_handler=None,
        allowed_roots=(tmp_path,),
        approved_shell_prefixes=("pytest",),
        brand=DEFAULT_BRAND,
    )

    result = handler(
        FakePermissionRequest(tool_name="read", read_only=True),
        {"toolName": "read"},
    )

    assert result.kind == "approved"


def test_safe_policy_allows_workspace_write(tmp_path) -> None:
    target = tmp_path / "inside.txt"
    handler = build_permission_handler(
        policy="safe",
        permission_handler=None,
        allowed_roots=(tmp_path,),
        approved_shell_prefixes=("pytest",),
        brand=DEFAULT_BRAND,
    )

    result = handler(
        FakePermissionRequest(tool_name="edit", path=str(target)),
        {"toolName": "edit"},
    )

    assert result.kind == "approved"


def test_safe_policy_denies_write_outside_roots(tmp_path) -> None:
    handler = build_permission_handler(
        policy="safe",
        permission_handler=None,
        allowed_roots=(tmp_path,),
        approved_shell_prefixes=("pytest",),
        brand=DEFAULT_BRAND,
    )

    result = handler(
        FakePermissionRequest(
            tool_name="edit",
            path=str(tmp_path.parent / "outside.txt"),
        ),
        {"toolName": "edit"},
    )

    assert result.kind != "approved"


def test_safe_policy_approves_allowed_shell_prefix(tmp_path) -> None:
    handler = build_permission_handler(
        policy="safe",
        permission_handler=None,
        allowed_roots=(tmp_path,),
        approved_shell_prefixes=("pytest",),
        brand=DEFAULT_BRAND,
    )

    result = handler(
        FakePermissionRequest(
            tool_name="bash",
            full_command_text="pytest -q tests/test_memory.py",
            possible_paths=[str(tmp_path / "tests" / "test_memory.py")],
        ),
        {"toolName": "bash"},
    )

    assert result.kind == "approved"


def test_safe_policy_denies_shell_control_operators(tmp_path) -> None:
    handler = build_permission_handler(
        policy="safe",
        permission_handler=None,
        allowed_roots=(tmp_path,),
        approved_shell_prefixes=("pytest",),
        brand=DEFAULT_BRAND,
    )

    result = handler(
        FakePermissionRequest(
            tool_name="bash",
            full_command_text="pytest -q && git status",
        ),
        {"toolName": "bash"},
    )

    assert result.kind != "approved"
    assert "control operators" in (result.message or "")


def test_approve_all_policy_returns_sdk_handler(tmp_path) -> None:
    handler = build_permission_handler(
        policy="approve_all",
        permission_handler=None,
        allowed_roots=(tmp_path,),
        approved_shell_prefixes=("pytest",),
        brand=DEFAULT_BRAND,
    )

    assert callable(handler)


def test_custom_handler_passthrough_is_respected(tmp_path) -> None:
    sentinel = object()

    def custom_handler(request, invocation):
        return sentinel

    handler = build_permission_handler(
        policy="safe",
        permission_handler=custom_handler,
        allowed_roots=(tmp_path,),
        approved_shell_prefixes=("pytest",),
        brand=DEFAULT_BRAND,
    )

    assert handler(None, {}) is sentinel


def test_safe_policy_denies_missing_shell_command_text(tmp_path) -> None:
    handler = build_permission_handler(
        policy="safe",
        permission_handler=None,
        allowed_roots=(tmp_path,),
        approved_shell_prefixes=("pytest",),
        brand=DEFAULT_BRAND,
    )

    result = handler(
        FakePermissionRequest(tool_name="bash", full_command_text=""),
        {"toolName": "bash"},
    )

    assert result.kind != "approved"
    assert "missing" in (result.message or "")


def test_safe_policy_allows_generic_request_with_possible_paths_inside_roots(tmp_path) -> None:
    handler = build_permission_handler(
        policy="safe",
        permission_handler=None,
        allowed_roots=(tmp_path,),
        approved_shell_prefixes=("pytest",),
        brand=DEFAULT_BRAND,
    )

    result = handler(
        FakePermissionRequest(
            tool_name="unknown",
            possible_paths=[str(tmp_path / "inside.txt")],
        ),
        {"toolName": "unknown"},
    )

    assert result.kind == "approved"


def test_safe_policy_denies_shell_with_disallowed_paths(tmp_path) -> None:
    handler = build_permission_handler(
        policy="safe",
        permission_handler=None,
        allowed_roots=(tmp_path,),
        approved_shell_prefixes=("pytest",),
        brand=DEFAULT_BRAND,
    )

    result = handler(
        FakePermissionRequest(
            tool_name="bash",
            full_command_text="pytest -q",
            possible_paths=[str(tmp_path.parent / "outside.txt")],
        ),
        {"toolName": "bash"},
    )

    assert result.kind != "approved"
    assert "outside the approved roots" in (result.message or "")


def test_permission_helpers_normalize_and_match_prefixes(tmp_path) -> None:
    assert normalize_shell_prefixes((" pytest  -q ", "PYTEST -q", "")) == ("pytest -q",)
    assert _command_matches_prefix("pytest -q tests/test_memory.py", ("pytest -q",)) is True
    assert _command_matches_prefix("python -m pytest", ("pytest -q",)) is False
    paths = _extract_request_paths(
        FakePermissionRequest(
            tool_name="edit",
            path=str(tmp_path / "one.txt"),
            possible_paths=[str(tmp_path / "two.txt")],
        ),
    )
    assert len(paths) == 2
    assert _paths_are_allowed(paths, (tmp_path.resolve(),)) is True
    assert _paths_are_allowed(paths + ((tmp_path.parent / "outside.txt").resolve(),), (tmp_path.resolve(),)) is False


def test_permission_property_style_prefix_normalization() -> None:
    import pytest

    hypothesis = pytest.importorskip("hypothesis")
    strategies = pytest.importorskip("hypothesis.strategies")

    @hypothesis.given(
        strategies.lists(
            strategies.sampled_from([" pytest ", "PYTEST", "git   status", "git status "]),
            min_size=1,
            max_size=6,
        ),
    )
    def run(values: list[str]) -> None:
        normalized = normalize_shell_prefixes(tuple(values))
        assert len(normalized) == len({value.lower() for value in normalized})
        assert all(prefix == " ".join(prefix.split()) for prefix in normalized)

    run()
