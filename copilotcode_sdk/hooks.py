from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

from .config import CopilotCodeConfig, DEFAULT_SKILL_NAMES
from .instructions import load_workspace_instructions
from .memory import MemoryStore

PATH_KEYS = ("path", "file", "dir", "directory", "cwd", "workspace")
SHELL_TOOL_NAMES = {"bash", "shell", "execute", "powershell"}
NOISY_TOOL_NAMES = {
    "glob",
    "grep",
    "search_codebase",
    "list_directory",
    "view",
    "read",
}


def build_default_hooks(
    config: CopilotCodeConfig,
    memory_store: MemoryStore,
) -> dict[str, Any]:
    allowed_roots = tuple(
        path.resolve(strict=False)
        for path in (
            *config.allowed_roots,
            memory_store.memory_dir,
        )
    )

    # Ring buffer for shell command repeat detection (anti-loop).
    _recent_shell: list[tuple[str, str]] = []

    def on_session_start(_: dict[str, Any], __: dict[str, str]) -> dict[str, Any] | None:
        memory_store.ensure()
        parts: list[str] = [
            f"{config.brand.public_name} is active for `{memory_store.project_root}`.",
        ]
        if config.include_workspace_instruction_snippets:
            instruction_bundle = load_workspace_instructions(config.working_path)
            if instruction_bundle.content:
                parts.append(instruction_bundle.content)
        if config.enable_hybrid_memory:
            parts.append(memory_store.build_index_context())
        return {"additionalContext": "\n\n".join(parts)}

    def on_user_prompt_submitted(
        input_data: dict[str, Any],
        _: dict[str, str],
    ) -> dict[str, Any] | None:
        prompt = str(input_data.get("prompt", ""))
        modified_prompt = _expand_skill_shorthand(prompt) if config.enable_skill_shorthand else prompt
        additional_parts: list[str] = []

        if config.enable_hybrid_memory:
            memory_context = memory_store.build_relevant_context(modified_prompt or prompt)
            if memory_context:
                additional_parts.append(memory_context)

        response: dict[str, Any] = {}
        if modified_prompt != prompt:
            response["modifiedPrompt"] = modified_prompt
        if additional_parts:
            response["additionalContext"] = "\n\n".join(additional_parts)
        return response or None

    def on_pre_tool_use(
        input_data: dict[str, Any],
        _: dict[str, str],
    ) -> dict[str, Any] | None:
        tool_name = str(input_data.get("toolName", "")).lower()
        tool_args = input_data.get("toolArgs")
        response: dict[str, Any] = {}

        violating_path = _find_disallowed_path(
            tool_args,
            allowed_roots,
            config.working_path,
        )
        if violating_path is not None:
            response["permissionDecision"] = "deny"
            response["permissionDecisionReason"] = (
                f"Path `{violating_path}` is outside the workspace, app config, and memory roots managed by {config.brand.public_name}."
            )

        if tool_name in SHELL_TOOL_NAMES and isinstance(tool_args, dict):
            modified_args = deepcopy(tool_args)
            if "timeout_ms" not in modified_args:
                modified_args["timeout_ms"] = config.shell_timeout_ms
            if modified_args != tool_args:
                response["modifiedArgs"] = modified_args

        return response or None

    def on_post_tool_use(
        input_data: dict[str, Any],
        _: dict[str, str],
    ) -> dict[str, Any] | None:
        tool_name = str(input_data.get("toolName", "")).lower()
        tool_result = input_data.get("toolResult")
        result_text = _stringify_result(tool_result)
        if not result_text:
            return None

        if tool_name in NOISY_TOOL_NAMES and len(result_text) > config.noisy_tool_char_limit:
            truncated = result_text[: config.noisy_tool_char_limit].rstrip() + "..."
            return {
                "modifiedResult": {
                    "truncated": True,
                    "originalLength": len(result_text),
                    "content": truncated,
                },
                "additionalContext": (
                    "The previous tool returned a large result. Focus on the most relevant lines instead of echoing the full payload."
                ),
            }

        if tool_name in SHELL_TOOL_NAMES and _tool_result_failed(tool_result):
            return {
                "additionalContext": (
                    "The shell command failed. Diagnose the existing error before retrying or switching tactics."
                ),
            }

        # Repeat detection for shell commands: warn when the same command
        # produces the same output multiple times (likely an unfalsifiable loop).
        if tool_name in SHELL_TOOL_NAMES:
            cmd = ""
            if isinstance(input_data.get("toolArgs"), dict):
                cmd = str(input_data["toolArgs"].get("command", ""))
            sig = (cmd, result_text[:500])
            repeat_count = _recent_shell.count(sig)
            _recent_shell.append(sig)
            if len(_recent_shell) > 10:
                _recent_shell.pop(0)
            if repeat_count >= 1:
                return {
                    "additionalContext": (
                        f"WARNING: This is the same command with the same output you saw {repeat_count} time(s) before. "
                        "Repeating the same action will not produce a different result. "
                        "If you suspect the output looks wrong due to formatting, redirect to a file "
                        "(e.g. `python script.py > out.txt`) and read it with `view` to see the real output. "
                        "If the output is actually correct, stop editing and declare success."
                    ),
                }

        return None

    def on_error_occurred(
        input_data: dict[str, Any],
        _: dict[str, str],
    ) -> dict[str, Any] | None:
        recoverable = bool(input_data.get("recoverable"))
        error_context = str(input_data.get("errorContext", ""))
        if not recoverable:
            return {"errorHandling": "abort"}
        if error_context == "model_call":
            return {"errorHandling": "retry", "retryCount": 1}
        if error_context == "tool_execution":
            return {"errorHandling": "skip"}
        return {"errorHandling": "retry", "retryCount": 1}

    return {
        "on_session_start": on_session_start,
        "on_user_prompt_submitted": on_user_prompt_submitted,
        "on_pre_tool_use": on_pre_tool_use,
        "on_post_tool_use": on_post_tool_use,
        "on_error_occurred": on_error_occurred,
    }


def _expand_skill_shorthand(prompt: str) -> str:
    stripped = prompt.strip()
    pattern = re.compile(r"^/([a-zA-Z0-9_-]+)(?:\s+(.*))?$", re.DOTALL)
    match = pattern.match(stripped)
    if not match:
        return prompt
    skill_name = match.group(1).lower()
    if skill_name not in DEFAULT_SKILL_NAMES:
        return prompt
    argument_text = (match.group(2) or "").strip()
    if argument_text:
        return f"Use the `{skill_name}` skill for this request.\n\nUser request:\n{argument_text}"
    return f"Use the `{skill_name}` skill for this request."


def _build_workspace_instruction_context(root: Path) -> str:
    snippets: list[str] = []
    for relative_path in (
        Path("CLAUDE.md"),
        Path("AGENTS.md"),
        Path(".github/copilot-instructions.md"),
    ):
        candidate = root / relative_path
        if not candidate.exists():
            continue
        text = candidate.read_text(encoding="utf-8")
        snippet = text[:6_000].rstrip()
        if len(text) > len(snippet):
            snippet += "\n..."
        snippets.append(f"## Workspace Instructions: `{relative_path}`\n{snippet}")
    return "\n\n".join(snippets)


def _find_disallowed_path(
    value: Any,
    allowed_roots: tuple[Path, ...],
    working_directory: Path,
    *,
    key_hint: str = "",
) -> Path | None:
    for path_str in _iter_candidate_paths(value, key_hint=key_hint):
        candidate = Path(path_str)
        if not candidate.is_absolute():
            candidate = working_directory / candidate
        resolved = candidate.expanduser().resolve(strict=False)
        if any(resolved.is_relative_to(root) for root in allowed_roots):
            continue
        return resolved
    return None


def _iter_candidate_paths(value: Any, *, key_hint: str = "") -> list[str]:
    found: list[str] = []
    lower_hint = key_hint.lower()
    if isinstance(value, dict):
        for key, child in value.items():
            found.extend(_iter_candidate_paths(child, key_hint=str(key)))
        return found
    if isinstance(value, list):
        for item in value:
            found.extend(_iter_candidate_paths(item, key_hint=key_hint))
        return found
    if isinstance(value, str):
        if value.startswith(("http://", "https://")):
            return found
        if any(token in lower_hint for token in PATH_KEYS):
            found.append(value)
    return found


def _stringify_result(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False)
    except TypeError:
        return repr(result)


def _tool_result_failed(result: Any) -> bool:
    if isinstance(result, dict):
        if result.get("exitCode") not in (None, 0):
            return True
        if result.get("error"):
            return True
    return False
