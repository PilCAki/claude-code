from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Callable, Iterable, Literal, Sequence

from .brand import BrandSpec

PermissionPolicy = Literal["safe", "approve_all", "custom"]

READ_ONLY_TOOL_NAMES = {
    "read",
    "view",
    "grep",
    "glob",
    "search",
    "search_codebase",
    "list_directory",
}
WRITE_TOOL_NAMES = {
    "edit",
    "write",
    "multi_edit",
    "notebook_edit",
}
SHELL_TOOL_NAMES = {"bash", "shell", "execute", "powershell"}
SHELL_CONTROL_PATTERN = re.compile(r"(;|&&|\|\||\||>|<|\$\(|`)")

DEFAULT_APPROVED_SHELL_PREFIXES: tuple[str, ...] = (
    "git status",
    "git diff",
    "git log",
    "git show",
    "ls",
    "dir",
    "pwd",
    "Get-ChildItem",
    "Get-Location",
    "cat",
    "type",
    "pytest",
    "py -3 -m pytest",
    "python -m pytest",
)


def normalize_shell_prefixes(
    values: Sequence[str],
) -> tuple[str, ...]:
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        prefix = " ".join(value.strip().split())
        if not prefix:
            continue
        lowered = prefix.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(prefix)
    return tuple(normalized)


def build_permission_handler(
    *,
    policy: PermissionPolicy,
    permission_handler: Callable[..., Any] | None,
    allowed_roots: Iterable[Path],
    approved_shell_prefixes: Sequence[str],
    brand: BrandSpec,
) -> Callable[..., Any]:
    from copilot.types import PermissionHandler, PermissionRequestResult

    if permission_handler is not None:
        return permission_handler

    if policy == "approve_all":
        return PermissionHandler.approve_all

    if policy == "custom":
        raise ValueError("permission_policy='custom' requires a permission_handler.")

    roots = tuple(path.resolve(strict=False) for path in allowed_roots)
    approved_prefixes = normalize_shell_prefixes(approved_shell_prefixes)

    def safe_handler(request: Any, invocation: dict[str, str]) -> PermissionRequestResult:
        tool_name = str(
            getattr(request, "tool_name", None)
            or invocation.get("toolName")
            or "",
        ).lower()
        request_paths = _extract_request_paths(request)

        if getattr(request, "read_only", False) or tool_name in READ_ONLY_TOOL_NAMES:
            return PermissionRequestResult(kind="approved")

        if tool_name in WRITE_TOOL_NAMES and _paths_are_allowed(request_paths, roots):
            return PermissionRequestResult(kind="approved")

        if tool_name in SHELL_TOOL_NAMES:
            command_text = " ".join(
                str(getattr(request, "full_command_text", "") or "").split(),
            )
            if not command_text:
                return PermissionRequestResult(
                    kind="denied-no-approval-rule-and-could-not-request-from-user",
                    message="Shell command text was missing, so the safe policy denied it.",
                )
            if SHELL_CONTROL_PATTERN.search(command_text):
                return PermissionRequestResult(
                    kind="denied-no-approval-rule-and-could-not-request-from-user",
                    message=(
                        "Shell command uses control operators or redirection. "
                        f"{brand.public_name} safe mode requires an explicit override."
                    ),
                )
            if not _command_matches_prefix(command_text, approved_prefixes):
                return PermissionRequestResult(
                    kind="denied-no-approval-rule-and-could-not-request-from-user",
                    message=(
                        "Shell command prefix is not in the approved safe list. "
                        "Use approve_all or extend approved_shell_prefixes for this workflow."
                    ),
                )
            if not _paths_are_allowed(request_paths, roots):
                return PermissionRequestResult(
                    kind="denied-no-approval-rule-and-could-not-request-from-user",
                    message="Shell command references paths outside the approved roots.",
                )
            return PermissionRequestResult(kind="approved")

        if request_paths and _paths_are_allowed(request_paths, roots):
            return PermissionRequestResult(kind="approved")

        return PermissionRequestResult(
            kind="denied-no-approval-rule-and-could-not-request-from-user",
            message=f"{brand.public_name} safe mode denied this permission request.",
        )

    return safe_handler


def _extract_request_paths(request: Any) -> tuple[Path, ...]:
    found: list[Path] = []
    for raw in (
        getattr(request, "path", None),
        *(getattr(request, "possible_paths", None) or []),
    ):
        if not raw:
            continue
        found.append(Path(str(raw)).expanduser().resolve(strict=False))
    return tuple(found)


def _paths_are_allowed(paths: Sequence[Path], roots: Sequence[Path]) -> bool:
    if not paths:
        return True
    for path in paths:
        if any(path.is_relative_to(root) for root in roots):
            continue
        return False
    return True


def _command_matches_prefix(command_text: str, approved_prefixes: Sequence[str]) -> bool:
    lowered = command_text.lower()
    for prefix in approved_prefixes:
        candidate = prefix.lower()
        if lowered == candidate or lowered.startswith(candidate + " "):
            return True
    return False
