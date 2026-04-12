from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import time

from .compaction import build_handoff_context
from .config import CopilotCodeConfig, DEFAULT_SKILL_NAMES
from .extraction import should_extract, build_extraction_prompt, build_enforce_extraction_prompt, build_session_end_extraction_prompt
from .instructions import InstructionBundle, load_workspace_instructions
from .mcp import build_mcp_delta, MCPLifecycleManager
from .memory import MemoryStore
from .prompt_compiler import PromptAssembler
from .skill_assets import SkillTracker
from .suggestions import build_prompt_suggestions, format_suggestions_prompt
from .tasks import TaskStore
from .events import EventBus, tool_called as _tool_called_event, tool_result as _tool_result_event, tool_denied as _tool_denied_event, file_changed as _file_changed_event
from .retry import RetryPolicy, RetryState, build_retry_response
from .token_budget import TokenBudget, parse_token_budget, strip_budget_directive, format_budget_status

SUGGESTION_TURN_THRESHOLD = 3
SUGGESTION_INTERVAL = 15

# Auto-compaction context tracking
DEFAULT_CONTEXT_WARNING_THRESHOLD = 0.80
DEFAULT_CONTEXT_CRITICAL_THRESHOLD = 0.95
DEFAULT_MAX_CONTEXT_CHARS = 800_000  # ~200K tokens at 4 chars/token

TASK_TOOL_NAMES = {"taskcreate", "taskupdate", "tasklist", "taskget"}

def _ensure_dict(obj: Any) -> Any:
    """Convert SessionEvent-like objects to dicts so isinstance checks work.

    If *obj* has a ``to_dict`` method (e.g. SessionEvent dataclass wrappers),
    call it.  Otherwise return *obj* unchanged.
    """
    if obj is None or isinstance(obj, (dict, str, int, float, bool, list)):
        return obj
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    # Fallback: try vars() for plain dataclasses / attrs
    try:
        return vars(obj)
    except TypeError:
        return obj


PATH_KEYS = ("path", "file", "dir", "directory", "cwd", "workspace")
SHELL_TOOL_NAMES = {"bash", "shell", "execute", "powershell"}
NOISY_TOOL_NAMES = {
    "glob",
    "grep",
    "search_codebase",
    "list_directory",
    "view",
    "read",
    "bash",
    "shell",
    "execute",
    "powershell",
}
WRITE_TOOL_NAMES = {"write_file", "write", "edit", "create_file"}
READ_TOOL_NAMES = {"read", "read_file", "view", "cat"}
DELETE_TOOL_NAMES = {"delete", "rm", "remove"}
# Tools whose results are safe to cache (pure reads, no side effects)
CACHEABLE_TOOL_NAMES = {"glob", "grep", "search_codebase", "read", "read_file", "view"}

# Tool result persistence threshold (chars)
DEFAULT_PERSIST_THRESHOLD = 50_000
PREVIEW_SIZE = 2_000

# Git safety: dangerous patterns that should be denied or warned about
GIT_DANGEROUS_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bgit\s+push\s+.*--force\b", "git push --force can overwrite remote history"),
    (r"\bgit\s+push\s+-f\b", "git push -f can overwrite remote history"),
    (r"\bgit\s+reset\s+--hard\b", "git reset --hard discards all uncommitted changes"),
    (r"\bgit\s+checkout\s+--\s+\.", "git checkout -- . discards all uncommitted changes"),
    (r"\bgit\s+clean\s+-[a-zA-Z]*f", "git clean -f permanently deletes untracked files"),
    (r"\bgit\s+branch\s+-D\b", "git branch -D force-deletes a branch without merge check"),
    (r"\bgit\s+stash\s+drop\b", "git stash drop permanently removes stashed changes"),
    (r"\bgit\s+rebase\s.*-i\b", "git rebase -i requires interactive input (not supported)"),
    (r"\bgit\s+add\s+-i\b", "git add -i requires interactive input (not supported)"),
    (r"--no-verify", "--no-verify skips safety hooks"),
)


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
    """Passive skill-completion detection — DISABLED.

    Skill completion is now handled exclusively by the CompleteSkill tool.
    The model must explicitly call CompleteSkill after finishing a skill's
    work; writing files to the output directory does NOT mark a skill
    as complete.

    This function is kept as a no-op to preserve the call-site contract.
    """
    return None

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
    not_done = [n for n in skill_map if n not in completed_skills]
    if not_done:
        lines.append("")
        lines.append(
            f"Skills remaining: **{', '.join(not_done)}**. "
            "Use InvokeSkill to execute the next skill whose prerequisites are met."
        )
    else:
        lines.append("")
        lines.append("All skills complete.")
    return "\n".join(lines)


def build_default_hooks(
    config: CopilotCodeConfig,
    memory_store: MemoryStore,
    *,
    skill_map: dict[str, dict[str, str]] | None = None,
    task_store: TaskStore | None = None,
    assembler: PromptAssembler | None = None,
    completed_skills: set[str] | None = None,
    session_memory_controller: Any | None = None,
    event_bus: EventBus | None = None,
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
    _recent_tool_names: list[str] = []
    _completed_skills: set[str] = completed_skills if completed_skills is not None else set()
    _fired_one_shots: set[str] = set()
    _loaded_instructions: list[InstructionBundle] = []
    _skill_tracker = SkillTracker() if skill_map else None
    _invoked_via_invoke_skill: set[str] = set()  # skills properly delegated
    _tool_call_count = [0]  # mutable counter in list for closure
    _last_extraction_turn = [0]
    _last_suggestion_turn = [0]
    _total_result_chars = [0]
    _turns_since_task_use = [0]
    _last_task_reminder_turn = [0]
    _read_file_state: dict[str, float] = {}  # path → timestamp of last read
    _tool_results_dir: Path | None = None
    _estimated_context_chars = [0]  # cumulative estimate of context size
    _compaction_warned = [False]  # whether we've already warned about context size
    _token_budget: list[TokenBudget | None] = [None]  # active token budget from user directive
    _budget_warning_fired = [False]
    _tool_result_cache: dict[str, str] = {}  # hash(tool_name+args) → result
    _file_changes: dict[str, str] = {}  # path → "created" | "modified" | "deleted"
    _active_paths: set[str] = set()  # paths touched by tool operations (reads + writes)
    # Repeated failure detection: track consecutive failures of the same tool+error
    _consecutive_failures: dict[str, int] = {}  # "tool:error_prefix" → count
    _FAILURE_THRESHOLD = 3  # inject guidance after this many consecutive failures
    _pending_skill_invocations: list[dict[str, Any]] = []  # queued InvokeSkill invocations
    _mcp_manager: MCPLifecycleManager | None = (
        MCPLifecycleManager(list(config.mcp_servers)) if config.mcp_servers else None
    )
    _retry_policy = RetryPolicy(
        base_delay_ms=config.retry_base_delay_ms,
        max_delay_ms=config.retry_max_delay_ms,
        max_attempts=config.retry_max_attempts,
        jitter=config.retry_jitter,
    )
    # Per-error-context retry states (reset after success)
    _retry_states: dict[str, RetryState] = {}

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
            # Pass active_paths from input_data for path-conditional rule filtering
            active_paths_raw = input_data.get("active_paths") or []
            active_paths = [Path(p) for p in active_paths_raw] if active_paths_raw else None
            instruction_bundle = load_workspace_instructions(
                config.working_path,
                active_paths=active_paths,
                on_loaded=lambda b: _loaded_instructions.append(b),
                user_config_dir=config.app_config_home,
            )
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

        # Inject non-cacheable prompt sections (skill catalog, MCP servers, etc.)
        if assembler is not None:
            dynamic_content = assembler.render_dynamic()
            if dynamic_content:
                parts.append(dynamic_content)

        # MCP server context — use instruction delta tracking when manager is available
        if config.mcp_servers:
            if _mcp_manager is not None:
                mcp_context = _mcp_manager.build_instruction_delta(config.mcp_servers)
                _mcp_manager.mark_delta_emitted()
            else:
                mcp_context = build_mcp_delta(config.mcp_servers)
            if mcp_context:
                parts.append(mcp_context)

        # Store instruction bundle for client capture
        if _loaded_instructions:
            pass  # captured via closure; client reads from hook result below

        result: dict[str, Any] = {"additionalContext": "\n\n".join(parts)}

        # Expose loaded instructions for client capture
        if _loaded_instructions:
            result["_instructionsLoaded"] = _loaded_instructions[0]

        # On resume with open tasks, inject an initial user message nudge
        if source == "resume" and task_store is not None and task_store.has_open_tasks():
            result["initialUserMessage"] = (
                "Session resumed. Check TaskList for open tasks and continue where you left off."
            )

        # On resume, inject compaction handoff artifact if available
        if source == "resume":
            compaction_dir = memory_store.memory_dir / "compaction"
            if compaction_dir.is_dir():
                artifact_path = None
                # Try session-specific artifact first
                resume_session_id = input_data.get("session_id")
                if resume_session_id:
                    candidate = compaction_dir / f"{resume_session_id}.md"
                    if candidate.is_file():
                        artifact_path = candidate
                # Fall back to most-recent artifact
                if artifact_path is None:
                    artifacts = sorted(compaction_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if artifacts:
                        artifact_path = artifacts[0]
                if artifact_path is not None:
                    try:
                        summary = artifact_path.read_text(encoding="utf-8")
                        if summary.strip():
                            # Populate memory index for richer handoff context
                            _memory_index = ""
                            try:
                                idx_path = memory_store.index_path
                                if idx_path.exists():
                                    _memory_index = idx_path.read_text(encoding="utf-8")
                            except OSError:
                                pass
                            handoff = build_handoff_context(
                                compaction_summary=summary,
                                memory_index=_memory_index,
                            )
                            # Prepend to existing additionalContext
                            existing = result.get("additionalContext", "")
                            result["additionalContext"] = handoff + "\n\n" + existing if existing else handoff
                    except OSError:
                        pass

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

        # Parse token budget directives (+500k, +1.5m, "use 500k tokens")
        if _token_budget[0] is None:
            budget = parse_token_budget(modified_prompt)
            if budget is not None:
                _token_budget[0] = budget
                modified_prompt = strip_budget_directive(modified_prompt)
                additional_parts.append(
                    f"Token budget set: {budget.tokens:,} tokens. "
                    "Work efficiently within this budget."
                )

        if config.enable_hybrid_memory:
            memory_context = memory_store.build_relevant_context(modified_prompt or prompt)
            if memory_context:
                additional_parts.append(memory_context)

        # Fire on_start frontmatter action when a skill shorthand is invoked
        if modified_prompt != prompt and skill_map:
            invoked_skill = _extract_skill_name_from_prompt(modified_prompt)
            if invoked_skill and invoked_skill in skill_map:
                if _skill_tracker is not None:
                    _skill_tracker.record_invocation(invoked_skill)
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
        tool_args = _ensure_dict(input_data.get("toolArgs"))
        response: dict[str, Any] = {}

        # Emit tool_called event
        if event_bus is not None:
            event_bus.emit(_tool_called_event(tool_name=tool_name))

        # Track skills properly invoked via InvokeSkill
        if tool_name == "invokeskill" and isinstance(tool_args, dict):
            sk = str(tool_args.get("skill", "")).strip()
            if sk:
                _invoked_via_invoke_skill.add(sk)

        # Block the CLI's built-in 'skill' slash command when InvokeSkill
        # is available. The 'skill' command loads skill methodology into the
        # current session, letting the model bypass InvokeSkill's child
        # isolation, verification gate, and anti-gaming enforcement.
        if tool_name == "skill" and skill_map:
            skill_arg = ""
            if isinstance(tool_args, dict):
                skill_arg = str(tool_args.get("skill", ""))
            if skill_arg in skill_map:
                response["permissionDecision"] = "deny"
                response["permissionDecisionReason"] = (
                    f"The /skill command is disabled for workspace skills. "
                    f"Use InvokeSkill(skill=\"{skill_arg}\") instead — it runs "
                    f"the skill in an isolated child session with quality "
                    f"verification. You MUST NOT do skill work directly."
                )
                return response

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

        # UNC path rejection (prevents NTLM credential leaks on Windows)
        if isinstance(tool_args, dict):
            for key in ("path", "filePath", "file_path", "file"):
                val = tool_args.get(key, "")
                if isinstance(val, str) and (val.startswith("\\\\") or val.startswith("//")):
                    response["permissionDecision"] = "deny"
                    response["permissionDecisionReason"] = (
                        f"UNC paths (\\\\...) are not allowed to prevent credential leaks."
                    )
                    break

        # Read-before-write enforcement
        if config.enforce_read_before_write and tool_name in WRITE_TOOL_NAMES and isinstance(tool_args, dict):
            write_path = ""
            for key in ("path", "filePath", "file_path", "file"):
                if key in tool_args:
                    write_path = str(tool_args[key])
                    break
            if write_path:
                resolved = _resolve_path(write_path, config.working_path)
                # Only enforce for existing files (new file creation is fine)
                if resolved.exists() and str(resolved) not in _read_file_state:
                    response["additionalContext"] = (
                        f"Warning: You are writing to `{write_path}` without reading it first. "
                        "Read the file before modifying it to understand existing code."
                    )

        # Git safety protocol: detect dangerous git commands
        if tool_name in SHELL_TOOL_NAMES and isinstance(tool_args, dict):
            cmd = str(tool_args.get("command", ""))
            for pattern, reason in GIT_DANGEROUS_PATTERNS:
                if re.search(pattern, cmd, re.IGNORECASE):
                    response["permissionDecision"] = "deny"
                    response["permissionDecisionReason"] = (
                        f"Blocked dangerous command: {reason}. "
                        "Use a safer alternative or ask the user for explicit permission."
                    )
                    break

        # Tool result cache: return cached result for identical read-only calls
        if (
            config.enable_tool_result_cache
            and tool_name in CACHEABLE_TOOL_NAMES
            and isinstance(tool_args, dict)
            and "permissionDecision" not in response
        ):
            cache_key = _tool_cache_key(tool_name, tool_args)
            if cache_key in _tool_result_cache:
                response["modifiedResult"] = {
                    "cached": True,
                    "content": _tool_result_cache[cache_key],
                }
                response["additionalContext"] = (
                    "Returned cached result for identical tool call."
                )
                return response

        if tool_name in SHELL_TOOL_NAMES and isinstance(tool_args, dict):
            modified_args = deepcopy(tool_args)
            if "timeout_ms" not in modified_args:
                modified_args["timeout_ms"] = config.shell_timeout_ms
            if modified_args != tool_args:
                response["modifiedArgs"] = modified_args

        # Emit tool_denied if permission was denied
        if event_bus is not None and response.get("permissionDecision") == "deny":
            event_bus.emit(_tool_denied_event(
                tool_name=tool_name,
                reason=response.get("permissionDecisionReason", ""),
            ))

        return response or None

    def on_post_tool_use(
        input_data: dict[str, Any],
        _: dict[str, str],
    ) -> dict[str, Any] | None:
        tool_name = str(input_data.get("toolName", "")).lower()
        tool_args = _ensure_dict(input_data.get("toolArgs"))
        tool_result = _ensure_dict(input_data.get("toolResult"))
        _tool_call_count[0] += 1

        # --- Repeated failure detection ---
        # Check if this tool call resulted in an error
        _tool_error = None
        if isinstance(tool_result, dict):
            _tool_error = tool_result.get("error") or tool_result.get("stderr")
        elif isinstance(tool_result, str) and "error" in tool_result.lower()[:50]:
            _tool_error = tool_result
        # Also check the input_data for error field (some tools report here)
        if not _tool_error:
            _raw_err = input_data.get("error")
            if _raw_err:
                _tool_error = str(_raw_err)

        if _tool_error:
            error_str = str(_tool_error)[:100]
            failure_key = f"{tool_name}:{error_str}"
            _consecutive_failures[failure_key] = _consecutive_failures.get(failure_key, 0) + 1
            fail_count = _consecutive_failures[failure_key]

            if fail_count >= _FAILURE_THRESHOLD:
                guidance = (
                    f"STUCK LOOP DETECTED: You have attempted `{tool_name}` "
                    f"{fail_count} times with the same error:\n"
                    f"  {error_str}\n\n"
                    f"Repeating the same approach will not work. "
                    f"Stop and diagnose: why is this failing? What is different "
                    f"about what you're sending vs what the tool expects?\n\n"
                    f"You MUST try a fundamentally different strategy. "
                    f"If you're trying to create or write a file, consider:\n"
                    f"- Writing via a shell command instead\n"
                    f"- Breaking the work into smaller steps\n"
                    f"- Using a different tool entirely\n"
                    f"- Simplifying what you're producing"
                )
                # Reset counter so we don't spam every turn
                _consecutive_failures[failure_key] = 0
                return {
                    "additionalContext": (
                        f"<system-reminder>\n{guidance}\n</system-reminder>"
                    ),
                }
        else:
            # Success — clear all failure counters for this tool
            keys_to_clear = [k for k in _consecutive_failures if k.startswith(f"{tool_name}:")]
            for k in keys_to_clear:
                del _consecutive_failures[k]

        # Emit tool_result event
        if event_bus is not None:
            result_str = _stringify_result(tool_result)
            event_bus.emit(_tool_result_event(
                tool_name=tool_name,
                result_chars=len(result_str),
            ))

        # Warn if CompleteSkill is called for a skill that wasn't delegated
        # via InvokeSkill — this means the model did the work directly,
        # bypassing child isolation and the verification pipeline.
        if tool_name == "completeskill" and skill_map and isinstance(tool_args, dict):
            sk = str(tool_args.get("skill", "")).strip()
            if sk and sk not in _invoked_via_invoke_skill:
                return {
                    "additionalContext": (
                        f"WARNING: You called CompleteSkill for '{sk}' but you "
                        f"never invoked it via InvokeSkill. You appear to have "
                        f"done the skill work directly in this session, bypassing "
                        f"child isolation and the verification pipeline. "
                        f"You MUST use InvokeSkill to delegate skill work to a "
                        f"child agent. Call InvokeSkill(skill=\"{sk}\", force=true) "
                        f"to redo this skill properly."
                    ),
                }

        # Detect InvokeSkill results and queue for async execution
        if tool_name == "invokeskill":
            result_str = _stringify_result(tool_result)
            try:
                parsed = json.loads(result_str) if isinstance(result_str, str) else tool_result
                invocation = parsed.get("_invocation") if isinstance(parsed, dict) else None
                if invocation and isinstance(invocation, dict):
                    _pending_skill_invocations.append(invocation)
                    skill_name = invocation.get("skill_name", "unknown")
                    return {
                        "additionalContext": (
                            f"Skill **{skill_name}** invocation queued. "
                            "Execution will begin shortly in a child session. "
                            "Continue with other work or wait for results."
                        ),
                    }
            except (json.JSONDecodeError, TypeError, ValueError):
                pass  # malformed result — fall through to normal processing

        # Increment SMC tool-call counter for mid-session extraction thresholds
        if session_memory_controller is not None:
            session_memory_controller.record_tool_call()

        # Track estimated context size for auto-compaction warnings
        result_text_len = len(_stringify_result(input_data.get("toolResult")))
        _estimated_context_chars[0] += result_text_len
        max_context = config.max_context_chars
        infinite_cfg = config.resolved_infinite_session_config()
        if infinite_cfg.get("enabled"):
            warning_threshold = infinite_cfg.get(
                "background_compaction_threshold", DEFAULT_CONTEXT_WARNING_THRESHOLD,
            )
            critical_threshold = infinite_cfg.get(
                "buffer_exhaustion_threshold", DEFAULT_CONTEXT_CRITICAL_THRESHOLD,
            )
        else:
            warning_threshold = DEFAULT_CONTEXT_WARNING_THRESHOLD
            critical_threshold = DEFAULT_CONTEXT_CRITICAL_THRESHOLD

        context_ratio = _estimated_context_chars[0] / max_context if max_context > 0 else 0
        if context_ratio >= critical_threshold and not _compaction_warned[0]:
            _compaction_warned[0] = True
            return {
                "additionalContext": (
                    "<system-reminder>\n"
                    "CRITICAL: Context window is nearly exhausted "
                    f"(~{context_ratio:.0%} of estimated capacity). "
                    "You MUST save your progress immediately:\n"
                    "1. Update all task statuses\n"
                    "2. Save any unsaved learnings to memory\n"
                    "3. Produce a handoff summary of current work state\n"
                    "The session may be compacted or terminated soon.\n"
                    "</system-reminder>"
                ),
            }
        if context_ratio >= warning_threshold and not _compaction_warned[0]:
            _compaction_warned[0] = True
            return {
                "additionalContext": (
                    "<system-reminder>\n"
                    "Context window is approaching capacity "
                    f"(~{context_ratio:.0%} of estimated limit). "
                    "Consider saving important findings to memory and "
                    "updating task statuses. Focus on completing current work "
                    "efficiently — avoid large exploratory reads.\n"
                    "</system-reminder>"
                ),
            }

        # Token budget consumption tracking
        if _token_budget[0] is not None:
            # Estimate tokens consumed this turn (~4 chars/token for result)
            _token_budget[0].consumed += result_text_len // 4
            budget = _token_budget[0]
            if budget.exhausted and not _budget_warning_fired[0]:
                _budget_warning_fired[0] = True
                return {
                    "additionalContext": (
                        "<system-reminder>\n"
                        f"Token budget EXHAUSTED. {format_budget_status(budget)} "
                        "Wrap up your current task immediately and report results.\n"
                        "</system-reminder>"
                    ),
                }
            if budget.progress >= 0.80 and not _budget_warning_fired[0]:
                _budget_warning_fired[0] = True
                return {
                    "additionalContext": (
                        "<system-reminder>\n"
                        f"{format_budget_status(budget)} "
                        "Approaching budget limit — prioritize finishing current work.\n"
                        "</system-reminder>"
                    ),
                }

        # Track read files for read-before-write enforcement and active paths
        if tool_name in READ_TOOL_NAMES and isinstance(tool_args, dict):
            for key in ("path", "filePath", "file_path", "file"):
                if key in tool_args:
                    resolved = _resolve_path(str(tool_args[key]), config.working_path)
                    _read_file_state[str(resolved)] = time.monotonic()
                    _active_paths.add(str(resolved))
                    break

        # Populate tool result cache for cacheable tools
        if (
            config.enable_tool_result_cache
            and tool_name in CACHEABLE_TOOL_NAMES
            and isinstance(tool_args, dict)
        ):
            result_str = _stringify_result(tool_result)
            if result_str:
                cache_key = _tool_cache_key(tool_name, tool_args)
                _tool_result_cache[cache_key] = result_str
                # Evict oldest entries if cache is too large
                if len(_tool_result_cache) > config.tool_result_cache_max_size:
                    oldest = next(iter(_tool_result_cache))
                    del _tool_result_cache[oldest]

        # File change tracking
        if isinstance(tool_args, dict):
            for key in ("path", "filePath", "file_path", "file"):
                if key in tool_args:
                    file_path = str(tool_args[key])
                    resolved = str(_resolve_path(file_path, config.working_path))
                    if tool_name in WRITE_TOOL_NAMES:
                        if resolved not in _file_changes:
                            change_type = "created"
                        else:
                            change_type = "modified"
                        _file_changes[resolved] = change_type
                        _active_paths.add(resolved)
                        if event_bus is not None:
                            event_bus.emit(_file_changed_event(path=resolved, change_type=change_type))
                    elif tool_name in DELETE_TOOL_NAMES:
                        _file_changes[resolved] = "deleted"
                        if event_bus is not None:
                            event_bus.emit(_file_changed_event(path=resolved, change_type="deleted"))
                    break
            # Invalidate cache when files change (writes invalidate reads of same path)
            if config.enable_tool_result_cache and tool_name in WRITE_TOOL_NAMES:
                # Clear all cached reads — file state has changed
                _tool_result_cache.clear()

        # Mark git context stale when git-mutating commands are detected
        if assembler is not None and tool_name in ("bash", "shell", "execute"):
            cmd = str(tool_args.get("command", "")) if isinstance(tool_args, dict) else ""
            _GIT_MUTATING = ("git checkout", "git commit", "git merge", "git rebase", "git pull", "git switch")
            if any(gc in cmd for gc in _GIT_MUTATING):
                assembler.mark_stale("git_context")

        # MCP server health tracking: detect MCP tool calls by prefix
        if _mcp_manager is not None and tool_name.startswith("mcp__"):
            # Extract server name: mcp__<server>__<tool>
            parts = tool_name.split("__", 2)
            if len(parts) >= 2:
                server_name = parts[1]
                if _tool_result_failed(tool_result):
                    _mcp_manager.record_failure(server_name, _stringify_result(tool_result)[:200])
                else:
                    _mcp_manager.record_success(server_name)
                # Inject health warning only when server state actually changed
                if _mcp_manager.has_changes():
                    health_prompt = _mcp_manager.build_status_prompt()
                    if health_prompt:
                        _mcp_manager.mark_delta_emitted()
                        return {"additionalContext": health_prompt}

        # Track recent tool names for suggestions
        _recent_tool_names.append(tool_name)
        if len(_recent_tool_names) > 20:
            _recent_tool_names.pop(0)

        # Advance skill tracker turn
        if _skill_tracker is not None:
            _skill_tracker.advance_turn()

        result_text = _stringify_result(tool_result)

        # Skill-completion detection (with frontmatter hook dispatch)
        if skill_map:
            nudge = _check_skill_completion(
                tool_name, tool_args, skill_map, _completed_skills,
                fired_one_shots=_fired_one_shots,
            )
            if nudge:
                # Record completion in skill tracker
                if _skill_tracker is not None:
                    for s in _completed_skills:
                        if _skill_tracker.completion_count(s) == 0:
                            _skill_tracker.record_completion(s)

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
        if config.enable_hybrid_memory and not config.session_memory_auto and should_extract(
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
            if config.extraction_mode == "enforce":
                extraction_prompt = build_enforce_extraction_prompt(
                    memory_dir=str(memory_store.memory_dir),
                    session_memory_path=str(memory_store.session_memory_path),
                )
            else:
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

        # Large result persistence: persist to disk, send preview
        if len(result_text) > DEFAULT_PERSIST_THRESHOLD:
            persisted_path = _persist_tool_result(
                config, result_text, tool_name, _tool_call_count[0],
            )
            if persisted_path is not None:
                preview = result_text[:PREVIEW_SIZE].rstrip()
                return {
                    "modifiedResult": {
                        "persisted": True,
                        "originalLength": len(result_text),
                        "persistedPath": str(persisted_path),
                        "content": preview,
                    },
                    "additionalContext": (
                        f"Tool result was {len(result_text):,} chars — persisted to "
                        f"`{persisted_path}`. Only the first {PREVIEW_SIZE} chars are shown above. "
                        "Read the persisted file if you need the full output."
                    ),
                }

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
            if isinstance(tool_args, dict):
                cmd = str(tool_args.get("command", ""))
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

        # Prompt suggestion: nudge toward available skills and next actions
        if (
            _tool_call_count[0] > SUGGESTION_TURN_THRESHOLD
            and _tool_call_count[0] - _last_suggestion_turn[0] >= SUGGESTION_INTERVAL
        ):
            open_tasks = task_store.list_open() if task_store else []
            completed_task_count = (
                sum(1 for t in task_store.list_all() if t.status.value == "completed")
                if task_store else 0
            )
            suggestions = build_prompt_suggestions(
                skill_map=skill_map or {},
                completed_skills=_completed_skills,
                open_tasks=open_tasks,
                completed_task_count=completed_task_count,
                session_turn=_tool_call_count[0],
                recent_tools=_recent_tool_names,
            )
            if suggestions:
                _last_suggestion_turn[0] = _tool_call_count[0]
                formatted = format_suggestions_prompt(suggestions)
                return {"additionalContext": formatted}
            # Fallback to simple skill suggestions via tracker
            if skill_map:
                if _skill_tracker is not None:
                    suggestable = _skill_tracker.top_surfaceable(skill_map, _completed_skills)
                else:
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

        # Refresh stale dynamic sections (e.g., git context after checkout)
        if assembler is not None and assembler.has_stale_sections():
            refreshed_dynamic = assembler.render_dynamic()
            assembler._stale_sections.clear()
            if refreshed_dynamic:
                return {"additionalContext": refreshed_dynamic}

        return None

    def on_error_occurred(
        input_data: dict[str, Any],
        _: dict[str, str],
    ) -> dict[str, Any] | None:
        recoverable = bool(input_data.get("recoverable"))
        error_context = str(input_data.get("errorContext", ""))
        if not recoverable:
            # Clear retry state on non-recoverable — nothing to retry
            _retry_states.pop(error_context, None)
            return {"errorHandling": "abort"}
        if error_context == "tool_execution":
            # Tool errors are skipped, not retried
            _retry_states.pop(error_context, None)
            return {"errorHandling": "skip"}
        # Model call and other recoverable errors: exponential backoff
        if error_context not in _retry_states:
            _retry_states[error_context] = RetryState(policy=_retry_policy)
        state = _retry_states[error_context]
        response = build_retry_response(state, error_context)
        # If we've exhausted retries, clean up the state
        if response.get("errorHandling") == "abort":
            _retry_states.pop(error_context, None)
        return response  # type: ignore[return-value]

    def on_success(error_context: str = "") -> None:
        """Call after a successful operation to reset retry state."""
        _retry_states.pop(error_context, None)

    def get_token_budget() -> TokenBudget | None:
        """Accessor for the active token budget (if any)."""
        return _token_budget[0]

    def get_file_changes() -> dict[str, str]:
        """Accessor for file changes tracked during this session."""
        return dict(_file_changes)

    def get_mcp_manager() -> MCPLifecycleManager | None:
        """Accessor for the MCP lifecycle manager."""
        return _mcp_manager

    def drain_pending_skill_invocations() -> list[dict[str, Any]]:
        """Return and clear all queued InvokeSkill invocations."""
        drained = list(_pending_skill_invocations)
        _pending_skill_invocations.clear()
        return drained

    def get_tool_call_count() -> int:
        """Accessor for the total tool call count."""
        return _tool_call_count[0]

    def get_recent_shell() -> list[tuple[str, str]]:
        """Accessor for recent shell command signatures."""
        return list(_recent_shell)

    def get_read_file_state() -> dict[str, float]:
        """Accessor for read-file tracking state (path → timestamp)."""
        return dict(_read_file_state)

    def get_estimated_context_chars() -> int:
        """Accessor for the cumulative estimated context size in chars."""
        return _estimated_context_chars[0]

    def get_tool_result_cache_size() -> int:
        """Accessor for the current tool result cache size."""
        return len(_tool_result_cache)

    def get_compaction_warned() -> bool:
        """Accessor for whether the compaction/context warning has already fired."""
        return _compaction_warned[0]

    return {
        "on_session_start": on_session_start,
        "on_user_prompt_submitted": on_user_prompt_submitted,
        "on_pre_tool_use": on_pre_tool_use,
        "on_post_tool_use": on_post_tool_use,
        "on_error_occurred": on_error_occurred,
        "get_token_budget": get_token_budget,
        "get_file_changes": get_file_changes,
        "get_mcp_manager": get_mcp_manager,
        "drain_pending_skill_invocations": drain_pending_skill_invocations,
        "get_tool_call_count": get_tool_call_count,
        "get_recent_shell": get_recent_shell,
        "get_read_file_state": get_read_file_state,
        "get_estimated_context_chars": get_estimated_context_chars,
        "get_tool_result_cache_size": get_tool_result_cache_size,
        "get_compaction_warned": get_compaction_warned,
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
    result = _ensure_dict(result)
    try:
        return json.dumps(result, ensure_ascii=False)
    except TypeError:
        return repr(result)


def _tool_result_failed(result: Any) -> bool:
    result = _ensure_dict(result)
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
        args_dict = _ensure_dict(tool_args)
        if isinstance(args_dict, dict):
            for key in ("path", "filePath", "file_path", "file"):
                if key in args_dict:
                    written_path = str(args_dict[key])
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


def _tool_cache_key(tool_name: str, tool_args: dict[str, Any]) -> str:
    """Build a deterministic cache key from tool name and arguments."""
    try:
        args_str = json.dumps(tool_args, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        args_str = repr(tool_args)
    raw = f"{tool_name}:{args_str}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _resolve_path(path_str: str, working_directory: Path) -> Path:
    """Resolve a path relative to the working directory."""
    candidate = Path(path_str)
    if not candidate.is_absolute():
        candidate = working_directory / candidate
    return candidate.expanduser().resolve(strict=False)


def _persist_tool_result(
    config: CopilotCodeConfig,
    result_text: str,
    tool_name: str,
    call_index: int,
) -> Path | None:
    """Persist a large tool result to disk and return the file path."""
    try:
        results_dir = config.working_path / ".copilotcode" / "tool-results"
        results_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{call_index:04d}-{tool_name}.txt"
        persist_path = results_dir / filename
        persist_path.write_text(result_text, encoding="utf-8")
        return persist_path
    except OSError:
        return None
