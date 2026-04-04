from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

from .config import CopilotCodeConfig, DEFAULT_SKILL_NAMES
from .extraction import should_extract, build_extraction_prompt, build_session_end_extraction_prompt
from .instructions import load_workspace_instructions
from .memory import MemoryStore
from .tasks import TaskStore

SUGGESTION_TURN_THRESHOLD = 3
SUGGESTION_INTERVAL = 15

TASK_TOOL_NAMES = {"taskcreate", "taskupdate", "tasklist", "taskget"}

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
WRITE_TOOL_NAMES = {"write_file", "write", "edit", "create_file"}


def _parse_hook_action(raw: str) -> tuple[str, str]:
    """Parse a frontmatter hook action string like ``inject:Remember to test``.

    Returns ``(action_type, payload)`` where action_type is one of
    ``inject``, ``remind``, or ``stop``.
    """
    if ":" in raw:
        action_type, _, payload = raw.partition(":")
        return action_type.strip().lower(), payload.strip()
    return raw.strip().lower(), ""


def _execute_hook_action(
    action_type: str,
    payload: str,
    skill_name: str,
) -> dict[str, Any] | None:
    """Execute a parsed hook action and return a hook response dict."""
    if action_type == "inject":
        return {"additionalContext": payload}
    if action_type == "remind":
        return {
            "additionalContext": (
                f"<system-reminder>\n"
                f"Skill **{skill_name}** reminder: {payload}\n"
                f"</system-reminder>"
            ),
        }
    if action_type == "stop":
        return {
            "additionalContext": (
                f"The **{skill_name}** skill has signaled completion. "
                f"Stop current work and report your results."
            ),
        }
    return None


def _find_suggestable_skills(
    skill_map: dict[str, dict[str, str]],
    completed_skills: set[str],
) -> list[str]:
    """Find skills that are not started and whose prerequisites are met."""
    completed_types = {
        fm.get("type", "")
        for name, fm in skill_map.items()
        if name in completed_skills and fm.get("type")
    }

    suggestable = []
    for name, fm in skill_map.items():
        if name in completed_skills:
            continue
        requires = fm.get("requires", "").strip().lower()
        if not requires or requires == "none" or requires in completed_types:
            suggestable.append(name)
    return suggestable


def _check_skill_completion(
    tool_name: str,
    tool_args: Any,
    skill_map: dict[str, dict[str, str]],
    completed_skills: set[str],
    fired_one_shots: set[str] | None = None,
) -> str | None:
    """Check if a tool call's output path matches a skill's outputs pattern.

    Returns an additionalContext nudge string, or None.
    Fires ``on_complete`` frontmatter actions when a skill completes.
    """
    if tool_name.lower() not in WRITE_TOOL_NAMES:
        return None
    if not isinstance(tool_args, dict):
        return None

    written_path = ""
    for key in ("path", "filePath", "file_path", "file"):
        if key in tool_args:
            written_path = str(tool_args[key])
            break
    if not written_path:
        return None

    # Normalize to forward slashes for matching
    written_lower = written_path.replace("\\", "/").lower()

    for skill_name, fm in skill_map.items():
        if skill_name in completed_skills:
            continue
        outputs = fm.get("outputs", "")
        if not outputs:
            continue
        # Extract just the path portion (before descriptive text, placeholders, or parens)
        output_path = outputs.split("(")[0].split("<")[0].strip()
        # If it still has spaces, take only the first token (the path)
        if " " in output_path:
            output_path = output_path.split()[0]
        # Normalize output pattern
        output_pattern = output_path.replace("\\", "/").lower().rstrip("/")
        # Check if the written path contains the output pattern
        if output_pattern and output_pattern in written_lower:
            completed_skills.add(skill_name)

            # Fire on_complete frontmatter action if present
            on_complete_raw = fm.get("on_complete", "")
            if on_complete_raw:
                is_one_shot = fm.get("one_shot", "").lower() in ("true", "yes", "1")
                one_shot_key = f"{skill_name}:on_complete"
                if is_one_shot and fired_one_shots is not None and one_shot_key in fired_one_shots:
                    pass  # Already fired
                else:
                    if fired_one_shots is not None:
                        fired_one_shots.add(one_shot_key)
                    action_type, payload = _parse_hook_action(on_complete_raw)
                    action_result = _execute_hook_action(action_type, payload, skill_name)
                    if action_result:
                        return action_result.get("additionalContext")

            # Find downstream skills
            skill_type = fm.get("type", "")
            downstream = [
                s["name"]
                for s in skill_map.values()
                if s.get("requires") == skill_type and s["name"] not in completed_skills
            ] if skill_type else []
            parts = [
                f"You appear to have completed the **{skill_name}** skill.",
            ]
            if downstream:
                names = ", ".join(downstream)
                parts.append(
                    f"Check the skill catalog \u2014 the following downstream skills may now be triggered: {names}. "
                    "Read their SKILL.md files for methodology."
                )
            else:
                parts.append(
                    "Check the skill catalog for any remaining skills."
                )
            return " ".join(parts)

    return None


def _build_skill_reminder(
    skill_map: dict[str, dict[str, str]],
    completed_skills: set[str],
) -> str:
    """Build a compact skill-status reminder for reinjection."""
    lines = ["**Skill status reminder:**"]
    for name, fm in skill_map.items():
        status = "DONE" if name in completed_skills else "not started"
        desc = fm.get("description", "")
        if len(desc) > 60:
            desc = desc[:57] + "..."
        lines.append(f"- {name} [{status}]: {desc}")
    lines.append("")
    lines.append(
        "After completing a skill, check if downstream skills should be triggered next. "
        "Do not stop if there are remaining skills whose prerequisites are now met."
    )
    return "\n".join(lines)


def build_default_hooks(
    config: CopilotCodeConfig,
    memory_store: MemoryStore,
    *,
    skill_map: dict[str, dict[str, str]] | None = None,
    task_store: TaskStore | None = None,
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
    _completed_skills: set[str] = set()
    _fired_one_shots: set[str] = set()
    _tool_call_count = [0]  # mutable counter in list for closure
    _last_extraction_turn = [0]
    _last_suggestion_turn = [0]
    _total_result_chars = [0]
    _turns_since_task_use = [0]
    _last_task_reminder_turn = [0]

    def on_session_start(
        input_data: dict[str, Any],
        __: dict[str, str],
    ) -> dict[str, Any] | None:
        memory_store.ensure()
        source = input_data.get("source", "create")
        parts: list[str] = [
            f"{config.brand.public_name} is active for `{memory_store.project_root}`.",
        ]
        if config.include_workspace_instruction_snippets:
            instruction_bundle = load_workspace_instructions(config.working_path)
            if instruction_bundle.content:
                parts.append(instruction_bundle.content)
        if config.enable_hybrid_memory:
            parts.append(memory_store.build_index_context())
        if task_store is not None:
            task_summary = task_store.summary_text()
            if task_summary:
                parts.append(task_summary)

        # Detect prior-run artifacts in outputs/
        prior_context = _build_prior_artifact_context(config.working_path)
        if prior_context:
            parts.append(prior_context)
            # Programmatic memory: persist prior-run summary
            if config.enable_hybrid_memory:
                _write_prior_run_memory(memory_store, config.working_path)

        result: dict[str, Any] = {"additionalContext": "\n\n".join(parts)}

        # On resume with open tasks, inject an initial user message nudge
        if source == "resume" and task_store is not None and task_store.has_open_tasks():
            result["initialUserMessage"] = (
                "Session resumed. Check TaskList for open tasks and continue where you left off."
            )

        # Watched paths: monitor instruction files for changes
        watched: list[str] = []
        for name in ("CLAUDE.md", "AGENTS.md", ".github/copilot-instructions.md"):
            candidate = config.working_path / name
            if candidate.is_file():
                watched.append(str(candidate))
        rules_dir = config.working_path / ".claude" / "rules"
        if rules_dir.is_dir():
            for rule_file in sorted(rules_dir.glob("*.md")):
                watched.append(str(rule_file))
        if watched:
            result["watchedPaths"] = watched

        return result

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

        # Fire on_start frontmatter action when a skill shorthand is invoked
        if modified_prompt != prompt and skill_map:
            invoked_skill = _extract_skill_name_from_prompt(modified_prompt)
            if invoked_skill and invoked_skill in skill_map:
                fm = skill_map[invoked_skill]
                on_start_raw = fm.get("on_start", "")
                if on_start_raw:
                    is_one_shot = fm.get("one_shot", "").lower() in ("true", "yes", "1")
                    one_shot_key = f"{invoked_skill}:on_start"
                    if not (is_one_shot and one_shot_key in _fired_one_shots):
                        _fired_one_shots.add(one_shot_key)
                        action_type, payload = _parse_hook_action(on_start_raw)
                        action_result = _execute_hook_action(action_type, payload, invoked_skill)
                        if action_result and "additionalContext" in action_result:
                            additional_parts.append(action_result["additionalContext"])

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
        tool_args = input_data.get("toolArgs")
        tool_result = input_data.get("toolResult")
        _tool_call_count[0] += 1

        result_text = _stringify_result(tool_result)

        # Skill-completion detection (with frontmatter hook dispatch)
        if skill_map:
            nudge = _check_skill_completion(
                tool_name, tool_args, skill_map, _completed_skills,
                fired_one_shots=_fired_one_shots,
            )
            if nudge:
                # Programmatic memory: record skill completion
                if config.enable_hybrid_memory:
                    _write_skill_completion_memory(
                        memory_store, _completed_skills, skill_map, tool_args,
                    )

                # Session-end extraction when all skills are done
                all_done = len(_completed_skills) >= len(skill_map)
                if all_done and config.enable_hybrid_memory:
                    end_prompt = build_session_end_extraction_prompt(
                        memory_dir=str(memory_store.memory_dir),
                        project_root=str(memory_store.project_root),
                    )
                    nudge = nudge + "\n\n" + end_prompt
                return {"additionalContext": nudge}

        # Track result size for extraction threshold
        _total_result_chars[0] += len(result_text)

        # Memory extraction check
        if config.enable_hybrid_memory and should_extract(
            tool_call_count=_tool_call_count[0],
            total_chars=_total_result_chars[0],
            last_extraction_turn=_last_extraction_turn[0],
            current_turn=_tool_call_count[0],
            tool_call_interval=config.extraction_tool_call_interval,
            char_threshold=config.extraction_char_threshold,
            min_turn_gap=config.extraction_min_turn_gap,
        ):
            _last_extraction_turn[0] = _tool_call_count[0]
            _total_result_chars[0] = 0
            extraction_prompt = build_extraction_prompt(
                memory_dir=str(memory_store.memory_dir),
                project_root=str(memory_store.project_root),
            )
            return {"additionalContext": extraction_prompt}

        # System-reminder reinjection every N tool calls
        if (
            skill_map
            and config.reminder_reinjection_interval > 0
            and _tool_call_count[0] % config.reminder_reinjection_interval == 0
        ):
            reminder = _build_skill_reminder(skill_map, _completed_skills)
            return {"additionalContext": reminder}

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

        # Repeat detection for shell commands
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

        # Track task tool usage for stale-task reminders
        if tool_name in TASK_TOOL_NAMES:
            _turns_since_task_use[0] = 0
        else:
            _turns_since_task_use[0] += 1

        # Stale-task reminder
        if (
            task_store is not None
            and config.task_reminder_turns > 0
            and _turns_since_task_use[0] >= config.task_reminder_turns
            and _tool_call_count[0] - _last_task_reminder_turn[0] >= config.task_reminder_cooldown_turns
            and task_store.has_open_tasks()
        ):
            _last_task_reminder_turn[0] = _tool_call_count[0]
            task_summary = task_store.summary_text()
            return {
                "additionalContext": (
                    f"<system-reminder>\n"
                    f"The task tools haven't been used recently. "
                    f"You have open tasks — check if any need status updates.\n\n"
                    f"{task_summary}\n\n"
                    f"Use TaskUpdate to mark tasks in_progress or completed as appropriate. "
                    f"Do NOT mention this reminder to the user.\n"
                    f"</system-reminder>"
                ),
            }

        # Prompt suggestion: nudge toward available skills
        if (
            skill_map
            and _tool_call_count[0] > SUGGESTION_TURN_THRESHOLD
            and _tool_call_count[0] - _last_suggestion_turn[0] >= SUGGESTION_INTERVAL
        ):
            suggestable = _find_suggestable_skills(skill_map, _completed_skills)
            if suggestable:
                _last_suggestion_turn[0] = _tool_call_count[0]
                names = ", ".join(suggestable)
                return {
                    "additionalContext": (
                        f"Reminder: the following skills are available and their prerequisites are met: {names}. "
                        "Check the skill catalog if you haven't started these yet."
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


def _extract_skill_name_from_prompt(expanded_prompt: str) -> str | None:
    """Extract the skill name from an expanded skill prompt like 'Use the `verify` skill...'"""
    match = re.match(r"Use the `([a-zA-Z0-9_-]+)` skill", expanded_prompt)
    return match.group(1).lower() if match else None


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


def _write_prior_run_memory(memory_store: MemoryStore, working_directory: Path) -> None:
    """Programmatically record prior-run metrics as a project memory."""
    try:
        outputs = working_directory / "outputs"
        prior_metrics = outputs / "prior_run_metrics.json"
        if not prior_metrics.exists():
            return

        data = json.loads(prior_metrics.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return

        # Build a readable summary
        lines = ["Prior analysis metrics carried forward from a previous run:\n"]
        for key, value in list(data.items())[:12]:
            lines.append(f"- {key}: {value}")

        memory_store.upsert_memory(
            title="Prior run metrics",
            description="Key metrics from the previous analysis run, carried forward for comparison",
            memory_type="project",
            content="\n".join(lines),
            slug="prior-run-metrics",
        )
    except Exception:
        pass


def _write_skill_completion_memory(
    memory_store: MemoryStore,
    completed_skills: set[str],
    skill_map: dict[str, dict[str, str]],
    tool_args: Any,
) -> None:
    """Programmatically write a project memory when a skill completes.

    This is the backstop for agents that don't write memory voluntarily.
    Records which skills have been completed and what output was produced.
    """
    try:
        from datetime import datetime, timezone

        just_completed = completed_skills - {s for s in completed_skills}  # noqa: we need all
        # Figure out which skill just completed by looking at what's new
        # We track via a cumulative record updated on each completion
        completed_list = sorted(completed_skills)
        total = len(skill_map)

        # Build a summary of completed skills with their output paths
        skill_lines = []
        for name in completed_list:
            fm = skill_map.get(name, {})
            outputs = fm.get("outputs", "unknown")
            skill_type = fm.get("type", "unknown")
            skill_lines.append(f"- **{name}** (type: {skill_type}) → outputs: {outputs}")

        # Include the file path that triggered completion
        written_path = ""
        if isinstance(tool_args, dict):
            for key in ("path", "filePath", "file_path", "file"):
                if key in tool_args:
                    written_path = str(tool_args[key])
                    break

        content = (
            f"Skills completed: {len(completed_list)}/{total}\n\n"
            + "\n".join(skill_lines)
            + (f"\n\nLast artifact written: `{written_path}`" if written_path else "")
            + f"\n\nUpdated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
        )

        memory_store.upsert_memory(
            title="Session skill progress",
            description=f"Skills completed ({len(completed_list)}/{total}) and their output paths",
            memory_type="project",
            content=content,
            slug="session-skill-progress",
        )
    except Exception:
        pass  # Never let memory writing crash the session


def _build_prior_artifact_context(working_directory: Path) -> str:
    """Read canonical carry-forward files from outputs/ and format a summary."""
    outputs = working_directory / "outputs"
    prior_metrics = outputs / "prior_run_metrics.json"
    column_mapping = outputs / "column_mapping.json"

    if not prior_metrics.exists() and not column_mapping.exists():
        return ""

    lines = [
        "## Prior Analysis Artifacts",
        "The following artifacts from a prior analysis were found in `outputs/`:",
    ]

    if prior_metrics.exists():
        lines.append("- `prior_run_metrics.json` — verified metrics from a prior run")
        try:
            data = json.loads(prior_metrics.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for key, value in list(data.items())[:8]:
                    if isinstance(value, (int, float)) and value is not None:
                        lines.append(f"  - {key}: {value}")
        except (json.JSONDecodeError, OSError):
            pass

    if column_mapping.exists():
        lines.append("- `column_mapping.json` — canonical column name mapping from a prior run")

    return "\n".join(lines)
