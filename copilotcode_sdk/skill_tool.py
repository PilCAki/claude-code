"""SkillTool — a tool the agent can call to invoke a workspace skill.

Mirrors Claude Code's SkillTool pattern: when the agent calls
``InvokeSkill(skill="excel-workbook-intake")``, we fork a child session
with the full SKILL.md content injected as the user prompt. The child
runs to completion, and the result comes back as a tool_result.

Memory inheritance follows Claude Code's approach:
- CLAUDE.md / workspace instructions: inherited (same system prompt prefix)
- Memory index (MEMORY.md contents): injected into the child's prompt
- Session memory: NOT inherited (only the parent runs extraction)
- File cache: isolated (child gets a fresh workspace view)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

from .memory import MemoryStore
from .skill_assets import parse_skill_frontmatter


# ---------------------------------------------------------------------------
# Tool prompt — this is what the model sees in the tool description
# ---------------------------------------------------------------------------

def build_skill_tool_prompt(
    skill_map: dict[str, dict[str, str]],
) -> str:
    """Build the prompt/description for the InvokeSkill tool.

    Mirrors Claude Code: compact skill listing inside the tool description
    so the model can match tasks to skills by reading the tool definition.
    """
    lines = [
        "Execute a workspace skill by forking a focused sub-session.",
        "",
        "When your task matches a skill's scope, invoke it here instead of "
        "doing the work directly. The skill runs in an isolated session with "
        "full methodology loaded — you get the result back when it finishes.",
        "",
        "How to invoke:",
        '- InvokeSkill(skill="excel-workbook-intake")',
        '- InvokeSkill(skill="rcm-analysis", context="Prior metrics available in outputs/prior_run_metrics.json")',
        "",
        "Important:",
        "- This is a BLOCKING REQUIREMENT: when a skill matches your current "
        "task, invoke it via this tool BEFORE attempting the work yourself.",
        "- After a skill completes, check if downstream skills are now unblocked.",
        "- Execute skills in dependency order — a skill's `requires` must be "
        "completed before it can run.",
        "- Do not invoke a skill that is already running or completed.",
        "",
    ]

    if skill_map:
        lines.append("Available skills:")
        for name, fm in skill_map.items():
            desc = fm.get("description", "")
            if len(desc) > 120:
                desc = desc[:117] + "..."
            stype = fm.get("type", "")
            requires = fm.get("requires", "none")
            lines.append(f"- {name}: {desc} [type: {stype}, requires: {requires}]")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool parameters schema
# ---------------------------------------------------------------------------

SKILL_TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "skill": {
            "type": "string",
            "description": (
                "Name of the skill to invoke. Must match one of the "
                "available skills listed in this tool's description."
            ),
        },
        "context": {
            "type": "string",
            "description": (
                "Optional context to pass to the skill session. Use this to "
                "provide dataset paths, prior artifact locations, or any "
                "information the skill needs to do its work. If omitted, "
                "the skill will receive the working directory context."
            ),
        },
        "force": {
            "type": "boolean",
            "description": (
                "Set to true to re-invoke a skill that was already completed "
                "in this session (e.g., for a different dataset)."
            ),
            "default": False,
        },
    },
    "required": ["skill"],
}


# ---------------------------------------------------------------------------
# Skill execution
# ---------------------------------------------------------------------------

def _read_skill_content(skill_map: dict[str, dict[str, str]], skill_name: str) -> str | None:
    """Read the full SKILL.md content for a skill.

    Also appends a note listing other files in the skill directory so the
    model knows about reference files it can read_file to access.

    Returns None if the skill doesn't exist or the file can't be read.
    """
    fm = skill_map.get(skill_name)
    if fm is None:
        return None
    skill_path = fm.get("_path")
    if not skill_path:
        return None
    skill_md = Path(skill_path)
    if not skill_md.is_file():
        return None
    content = skill_md.read_text(encoding="utf-8")

    # List sibling files/dirs in the skill directory so the model can discover
    # reference files (templates, style guides, etc.)
    skill_dir = skill_md.parent
    siblings: list[str] = []
    for item in sorted(skill_dir.iterdir()):
        if item.name == "SKILL.md":
            continue
        if item.is_dir():
            # List contents of subdirectories (e.g., references/)
            for sub in sorted(item.rglob("*")):
                if sub.is_file():
                    siblings.append(str(sub.relative_to(skill_dir)))
        elif item.is_file():
            siblings.append(item.name)
    if siblings:
        content += (
            "\n\n---\n"
            "**This skill directory also contains the following files. "
            "Use read_file to access them:**\n"
            + "\n".join(f"- `{s}`" for s in siblings)
        )
    return content


def _build_skill_user_prompt(
    skill_name: str,
    skill_content: str,
    context: str,
    memory_context: str,
    working_directory: str,
    completed_skills: Sequence[str] = (),
) -> str:
    """Build the user prompt for the forked skill session.

    Includes:
    - Full SKILL.md content (methodology, expected outputs, quality rubrics)
    - Memory context (MEMORY.md index + relevant memories)
    - Any caller-provided context (dataset paths, prior artifacts)
    - Skill execution framing
    """
    parts = [
        f"# Skill Execution: {skill_name}",
        "",
        "You are executing a workspace skill. Follow the methodology below exactly.",
        f"Working directory: `{working_directory}`",
        "",
    ]

    if completed_skills:
        parts.append("Previously completed skills in this session: "
                      + ", ".join(completed_skills))
        parts.append("")

    parts.extend([
        "---",
        "",
        skill_content,
        "",
        "---",
        "",
    ])

    if context:
        parts.extend([
            "## Context from orchestrator",
            "",
            context,
            "",
        ])

    if memory_context:
        parts.extend([
            "## Project memory",
            "",
            memory_context,
            "",
        ])

    parts.extend([
        "## Instructions",
        "",
        "1. Follow the skill methodology above step by step.",
        "2. Write all outputs to the paths specified in the skill definition.",
        "3. Verify your outputs match the quality rubric before finishing.",
        "4. Do NOT call CompleteSkill or InvokeSkill — verification and completion",
        "   are handled automatically after you finish.",
    ])

    return "\n".join(parts)


def build_skill_tool(
    skill_map: dict[str, dict[str, str]],
    memory_store: MemoryStore,
    working_directory: str,
    completed_skills: set[str],
    session_holder: list[Any] | None = None,
) -> Any:
    """Build the InvokeSkill tool for the Copilot SDK.

    Returns a ``copilot.types.Tool`` that, when called by the agent, forks
    a child session with the full skill content and returns the result.

    When *session_holder* is provided and populated, the tool handler forks
    a child session inline and waits for it to complete. The child gets the
    full SKILL.md as its prompt and runs autonomously with all tools.
    Without a session, falls back to returning metadata (no fork).
    """
    from copilot.types import Tool, ToolInvocation, ToolResult
    import logging as _logging
    _skill_logger = _logging.getLogger("copilotcode.verifier")

    async def _handle_invoke_skill(invocation: ToolInvocation) -> ToolResult:
        args = invocation.arguments or {}
        skill_name = (args.get("skill") or "").strip()

        if not skill_name:
            return ToolResult(
                text_result_for_llm="Error: 'skill' is required.",
                result_type="error",
            )

        if skill_name not in skill_map:
            available = ", ".join(sorted(skill_map.keys()))
            return ToolResult(
                text_result_for_llm=(
                    f"Error: skill '{skill_name}' not found. "
                    f"Available skills: {available}"
                ),
                result_type="error",
            )

        force = bool(args.get("force", False))
        if skill_name in completed_skills and not force:
            return ToolResult(
                text_result_for_llm=(
                    f"Skill '{skill_name}' was already completed in this session. "
                    "If you need to re-run it (e.g., for a different dataset), "
                    "pass force=true. Otherwise, check for downstream skills "
                    "that may now be unblocked."
                ),
            )

        # Check prerequisites
        fm = skill_map[skill_name]
        requires = fm.get("requires", "").strip().lower()
        if requires and requires != "none":
            completed_types = {
                skill_map[s].get("type", "").lower()
                for s in completed_skills
                if s in skill_map
            }
            if requires not in completed_types:
                # Find a skill that provides the required type
                provider = next(
                    (n for n, f in skill_map.items()
                     if f.get("type", "").lower() == requires),
                    None,
                )
                invoke_hint = (
                    f'1. Call InvokeSkill(skill="{provider}") first\n'
                    if provider else
                    f"1. Invoke a skill of type '{requires}' first\n"
                )
                return ToolResult(
                    text_result_for_llm=(
                        f"Error: skill '{skill_name}' requires '{requires}' "
                        f"but no skill of that type has been completed yet. "
                        f"You MUST:\n"
                        f"{invoke_hint}"
                        f"2. Do the skill work\n"
                        f"3. Call CompleteSkill(skill=\"...\", ...) when done\n"
                        f"4. THEN you can invoke {skill_name}\n\n"
                        f"Do NOT create workaround scripts. Do NOT skip prerequisites."
                    ),
                    result_type="error",
                )

        # Read the full SKILL.md content
        skill_content = _read_skill_content(skill_map, skill_name)
        if skill_content is None:
            return ToolResult(
                text_result_for_llm=(
                    f"Error: could not read SKILL.md for '{skill_name}'. "
                    "The skill file may be missing or unreadable."
                ),
                result_type="error",
            )

        # Build memory context for the child
        memory_context = memory_store.build_index_context()

        # Build the full prompt
        context = args.get("context", "")
        user_prompt = _build_skill_user_prompt(
            skill_name=skill_name,
            skill_content=skill_content,
            context=context,
            memory_context=memory_context,
            working_directory=working_directory,
            completed_skills=sorted(completed_skills),
        )

        # --- Fork a child session to execute the skill ---
        session = None
        if session_holder and len(session_holder) > 0:
            session = session_holder[0]

        if session is None:
            _skill_logger.warning(
                "NO SESSION for InvokeSkill(%s) — returning metadata only, "
                "no child agent will be forked!", skill_name,
            )
            return ToolResult(
                text_result_for_llm=json.dumps({
                    "status": "invoke_skill_no_session",
                    "skill_name": skill_name,
                    "skill_type": fm.get("type", ""),
                    "requires": requires or "none",
                    "outputs": fm.get("outputs", ""),
                    "description": fm.get("description", ""),
                    "warning": "No session available — skill was NOT executed by a child agent.",
                }),
            )

        # --- Fork → verify → retry loop (all enforced by code) ---
        from .subagent import SubagentSpec
        from .verifier import (
            MAX_VERIFICATION_ATTEMPTS,
            MAX_VERIFIER_MALFUNCTIONS,
            VerificationExhaustedError,
            format_fail_feedback,
            run_verification,
            write_failure_trace,
        )

        outputs_rel = fm.get("outputs", "").strip()
        verify_output_dir = (
            Path(working_directory) / outputs_rel if outputs_rel
            else Path(working_directory)
        )

        # Read prior metrics for the verifier
        prior_metrics: str | None = None
        metrics_path = Path(working_directory) / "prior_run_metrics.json"
        if metrics_path.is_file():
            try:
                prior_metrics = metrics_path.read_text(encoding="utf-8")
            except Exception:
                pass

        attempt_counts = 0
        malfunction_counts = 0
        attempt_history: list[dict[str, Any]] = []
        total_start = time.time()

        # Fork child ONCE — keep alive for the entire verify/fix cycle
        _skill_logger.info("FORK %s — spawning child agent", skill_name)

        def _child_event_handler(event) -> None:
            """Log child agent events so we can see what the child is doing.

            ``event`` may be a raw SessionEvent object (from the copilot SDK)
            or a dict — we handle both.
            """
            try:
                # Extract event type — works with both raw objects and dicts
                if isinstance(event, dict):
                    raw_type = event.get("type", "")
                else:
                    raw_type = getattr(event, "type", "")
                etype = raw_type.value if hasattr(raw_type, "value") else str(raw_type)
            except Exception:
                return

            if etype == "tool.execution_start":
                if isinstance(event, dict):
                    data = event.get("data", {})
                    tool_name = data.get("tool_name", "?") if isinstance(data, dict) else "?"
                else:
                    data = getattr(event, "data", None)
                    tool_name = getattr(data, "tool_name", "?") if data else "?"
                _skill_logger.info("CHILD %s — tool: %s", skill_name, tool_name)
            elif etype == "tool.execution_complete":
                if isinstance(event, dict):
                    data = event.get("data", {})
                    tool_name = data.get("tool_name", "?") if isinstance(data, dict) else "?"
                    error = data.get("error") if isinstance(data, dict) else None
                else:
                    data = getattr(event, "data", None)
                    tool_name = getattr(data, "tool_name", "?") if data else "?"
                    error = getattr(data, "error", None) if data else None
                if error:
                    _skill_logger.warning(
                        "CHILD %s — tool DONE (error): %s — %s",
                        skill_name, tool_name, str(error)[:200],
                    )
                else:
                    _skill_logger.info("CHILD %s — tool DONE: %s", skill_name, tool_name)
            elif etype in ("turn.started", "turn.completed"):
                _skill_logger.info("CHILD %s — %s", skill_name, etype)
            elif etype == "session.started":
                _skill_logger.info("CHILD %s — session started", skill_name)
            else:
                _skill_logger.debug("CHILD %s — event: %s", skill_name, etype)

        try:
            spec = SubagentSpec(
                role=f"skill:{skill_name}",
                system_prompt_suffix="",
                max_turns=100,  # generous — child may need many turns across retries
                timeout_seconds=3600.0,  # 1 hour total for skill + retries
            )
            child = await session.fork_child(spec, on_event=_child_event_handler)
        except Exception as exc:
            _skill_logger.error("FORK %s — spawn failed: %s", skill_name, exc)
            return ToolResult(
                text_result_for_llm=f"Error: could not fork child for '{skill_name}': {exc}",
                result_type="error",
            )

        try:
            # --- Initial work ---
            _skill_logger.info("FORK %s — child starting work", skill_name)
            try:
                await child.send_and_wait(user_prompt, timeout=1800.0)
                raw_result = await child.get_last_response_text()
                if not raw_result:
                    raw_result = "(child produced no text response)"
            except Exception as child_exc:
                _skill_logger.error(
                    "FORK %s — child send_and_wait raised: %s", skill_name, child_exc,
                )
                return ToolResult(
                    text_result_for_llm=f"Error: child agent for '{skill_name}' failed: {child_exc}",
                    result_type="error",
                )
            _skill_logger.info(
                "FORK %s — child finished (%d chars). Output dir exists: %s",
                skill_name, len(raw_result), verify_output_dir.exists(),
            )
            # Log first 500 chars of child result for debugging
            _skill_logger.info(
                "FORK %s — child result preview: %.500s",
                skill_name, raw_result[:500],
            )

            # --- Verify/fix loop ---
            max_loops = MAX_VERIFICATION_ATTEMPTS + MAX_VERIFIER_MALFUNCTIONS + 3
            for loop_idx in range(max_loops):
                # Check output dir has real content
                if outputs_rel:
                    if not verify_output_dir.exists():
                        _skill_logger.warning(
                            "GATE %s: output dir '%s' missing", skill_name, outputs_rel,
                        )
                        attempt_counts += 1
                        if attempt_counts >= MAX_VERIFICATION_ATTEMPTS:
                            trace_path = write_failure_trace(
                                skill_name, attempt_history, verify_output_dir,
                                Path(working_directory),
                            )
                            raise VerificationExhaustedError(skill_name, trace_path)
                        # Send feedback to SAME child
                        await child.send_and_wait(
                            f"Output directory '{outputs_rel}' does not exist. "
                            f"You must write outputs to this directory. Fix this now.",
                            timeout=1800.0,
                        )
                        raw_result = await child.get_last_response_text()
                        continue

                    total_bytes = sum(
                        f.stat().st_size for f in verify_output_dir.rglob("*")
                        if f.is_file()
                    )
                    if total_bytes < MIN_OUTPUT_BYTES:
                        _skill_logger.warning(
                            "GATE %s: output dir has only %d bytes", skill_name, total_bytes,
                        )
                        attempt_counts += 1
                        if attempt_counts >= MAX_VERIFICATION_ATTEMPTS:
                            trace_path = write_failure_trace(
                                skill_name, attempt_history, verify_output_dir,
                                Path(working_directory),
                            )
                            raise VerificationExhaustedError(skill_name, trace_path)
                        await child.send_and_wait(
                            f"Output directory '{outputs_rel}' has only {total_bytes} "
                            f"bytes — this looks like placeholders. Write real outputs.",
                            timeout=1800.0,
                        )
                        raw_result = await child.get_last_response_text()
                        continue

                # Run verification
                current_attempt = attempt_counts + 1
                vresult = await run_verification(
                    skill_name=skill_name,
                    skill_content=skill_content,
                    output_dir=verify_output_dir,
                    workspace=Path(working_directory),
                    fork_child=session.fork_child,
                    prior_metrics=prior_metrics,
                    attempt_num=current_attempt,
                )

                if vresult.passed:
                    _skill_logger.info("PASS %s — verification passed", skill_name)
                    completed_skills.add(skill_name)
                    total_elapsed = round(time.time() - total_start, 1)
                    return ToolResult(
                        text_result_for_llm=json.dumps({
                            "status": "skill_complete",
                            "skill_name": skill_name,
                            "skill_type": fm.get("type", ""),
                            "outputs": outputs_rel,
                            "elapsed_seconds": total_elapsed,
                            "result_summary": raw_result[:2000] if isinstance(raw_result, str) else "",
                        }),
                    )
                elif vresult.is_malfunction:
                    malfunction_counts += 1
                    attempt_history.append({
                        "verdict": "MALFUNCTION",
                        "raw_output": vresult.raw_output,
                        "timestamp": time.time(),
                    })
                    if malfunction_counts >= MAX_VERIFIER_MALFUNCTIONS + 1:
                        attempt_counts += 1
                        malfunction_counts = 0
                        if attempt_counts >= MAX_VERIFICATION_ATTEMPTS:
                            trace_path = write_failure_trace(
                                skill_name, attempt_history, verify_output_dir,
                                Path(working_directory),
                            )
                            raise VerificationExhaustedError(skill_name, trace_path)
                    # Re-run verification only (malfunction is verifier's fault)
                    continue
                else:
                    # FAIL or PARTIAL — send feedback to same child
                    attempt_counts += 1
                    malfunction_counts = 0
                    attempt_history.append({
                        "verdict": vresult.verdict,
                        "failed_checks": vresult.failed_checks,
                        "raw_output": vresult.raw_output,
                        "timestamp": time.time(),
                    })
                    if attempt_counts >= MAX_VERIFICATION_ATTEMPTS:
                        trace_path = write_failure_trace(
                            skill_name, attempt_history, verify_output_dir,
                            Path(working_directory),
                        )
                        raise VerificationExhaustedError(skill_name, trace_path)
                    feedback = format_fail_feedback(
                        vresult.failed_checks, attempt_counts,
                        MAX_VERIFICATION_ATTEMPTS,
                    )
                    _skill_logger.info(
                        "FAIL %s attempt %d — sending feedback to child",
                        skill_name, attempt_counts,
                    )
                    await child.send_and_wait(
                        f"## VERIFICATION FAILED\n\n{feedback}\n\n"
                        f"Fix the issues above and ensure all outputs are correct.",
                        timeout=1800.0,
                    )
                    raw_result = await child.get_last_response_text()
                    continue

            # Safety net
            return ToolResult(
                text_result_for_llm=f"Error: skill '{skill_name}' loop exceeded safety limit.",
                result_type="error",
            )
        finally:
            await child.destroy()

    prompt = build_skill_tool_prompt(skill_map)

    return Tool(
        name="InvokeSkill",
        description=prompt,
        handler=_handle_invoke_skill,
        parameters=SKILL_TOOL_PARAMETERS,
        skip_permission=True,
    )


# ---------------------------------------------------------------------------
# CompleteSkill tool — explicit skill completion signal
# ---------------------------------------------------------------------------

COMPLETE_SKILL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "skill": {
            "type": "string",
            "description": "Name of the skill that was completed.",
        },
        "source_file": {
            "type": "string",
            "description": "The primary input file that was processed.",
        },
        "output_summary": {
            "type": "string",
            "description": "Brief summary of what was produced and where.",
        },
        "row_count": {
            "type": "integer",
            "description": "Number of data rows processed (for provenance).",
        },
    },
    "required": ["skill", "output_summary"],
}

MIN_OUTPUT_BYTES = 1_000  # reject empty/placeholder outputs


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from sync context, handling already-running loops."""
    import asyncio
    import concurrent.futures

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


def build_complete_skill_tool(
    skill_map: dict[str, dict[str, str]],
    completed_skills: set[str],
    working_directory: str,
    session_holder: list[Any] | None = None,
) -> Any:
    """Build the CompleteSkill tool for explicit skill completion.

    The model calls this after finishing a skill's work. This is the
    primary mechanism for marking a skill as complete — passive file-write
    detection in hooks is kept only as a backup.

    If *session_holder* contains a session (index 0), verification is run
    before marking the skill complete.  When no session is available
    (None or empty list), verification is skipped for backwards
    compatibility.
    """
    from copilot.types import Tool, ToolInvocation, ToolResult
    from .verifier import (
        MAX_VERIFICATION_ATTEMPTS,
        MAX_VERIFIER_MALFUNCTIONS,
        VerificationExhaustedError,
        format_fail_feedback,
        run_verification,
        write_failure_trace,
    )

    import logging as _logging
    _skill_logger = _logging.getLogger("copilotcode.verifier")

    # Per-skill tracking (closure state, persists across calls)
    _attempt_counts: dict[str, int] = {}
    _malfunction_counts: dict[str, int] = {}
    _attempt_history: dict[str, list[dict[str, Any]]] = {}

    async def _handle_complete_skill(invocation: ToolInvocation) -> ToolResult:
        args = invocation.arguments or {}
        skill_name = (args.get("skill") or "").strip()

        if not skill_name:
            return ToolResult(
                text_result_for_llm="Error: 'skill' is required.",
                result_type="error",
            )

        if skill_name not in skill_map:
            available = ", ".join(sorted(skill_map.keys()))
            return ToolResult(
                text_result_for_llm=(
                    f"Error: skill '{skill_name}' not found. "
                    f"Available skills: {available}"
                ),
                result_type="error",
            )

        # Check the skill's output directory has real content
        fm = skill_map[skill_name]
        outputs_rel = fm.get("outputs", "").strip()
        if outputs_rel:
            output_dir = Path(working_directory) / outputs_rel
            if not output_dir.exists():
                _skill_logger.warning(
                    "GATE REJECT %s: output dir '%s' does not exist",
                    skill_name, outputs_rel,
                )
                return ToolResult(
                    text_result_for_llm=(
                        f"Error: output directory '{outputs_rel}' does not exist. "
                        f"You haven't produced the expected outputs yet. "
                        f"Complete the skill work first, then call CompleteSkill."
                    ),
                    result_type="error",
                )
            total_bytes = sum(
                f.stat().st_size for f in output_dir.rglob("*") if f.is_file()
            )
            if total_bytes < MIN_OUTPUT_BYTES:
                _skill_logger.warning(
                    "GATE REJECT %s: output dir '%s' has only %d bytes (min %d)",
                    skill_name, outputs_rel, total_bytes, MIN_OUTPUT_BYTES,
                )
                return ToolResult(
                    text_result_for_llm=(
                        f"Error: output directory '{outputs_rel}' has only "
                        f"{total_bytes} bytes — looks like placeholders. "
                        f"Complete the skill work with real outputs first."
                    ),
                    result_type="error",
                )

        # --- Verification gate ---
        session = None
        if session_holder and len(session_holder) > 0:
            session = session_holder[0]

        if session is not None:
            # Read skill content for the verifier
            skill_content = _read_skill_content(skill_map, skill_name)
            if skill_content is None:
                skill_content = f"(Skill definition for '{skill_name}' could not be read.)"

            # Read prior_run_metrics.json if it exists
            prior_metrics: str | None = None
            metrics_path = Path(working_directory) / "prior_run_metrics.json"
            if metrics_path.is_file():
                try:
                    prior_metrics = metrics_path.read_text(encoding="utf-8")
                except Exception:
                    prior_metrics = None

            # Resolve the output directory for verification
            if outputs_rel:
                verify_output_dir = Path(working_directory) / outputs_rel
            else:
                verify_output_dir = Path(working_directory)

            # Initialize tracking for this skill if needed
            if skill_name not in _attempt_counts:
                _attempt_counts[skill_name] = 0
                _malfunction_counts[skill_name] = 0
                _attempt_history[skill_name] = []

            # Run verification (pass current attempt number for log naming)
            current_attempt = _attempt_counts[skill_name] + 1
            vresult = await run_verification(
                skill_name=skill_name,
                skill_content=skill_content,
                output_dir=verify_output_dir,
                workspace=Path(working_directory),
                fork_child=session.fork_child,
                prior_metrics=prior_metrics,
                attempt_num=current_attempt,
            )

            if vresult.passed:
                # PASS — reset counters and fall through to mark complete
                _skill_logger.info(
                    "PASS %s — verification passed, marking complete", skill_name,
                )
                _malfunction_counts[skill_name] = 0
            elif vresult.is_malfunction:
                _malfunction_counts[skill_name] += 1
                consecutive_malfunctions = _malfunction_counts[skill_name]

                # Record in history
                _attempt_history[skill_name].append({
                    "verdict": "MALFUNCTION",
                    "raw_output": vresult.raw_output,
                    "timestamp": time.time(),
                })

                if consecutive_malfunctions >= MAX_VERIFIER_MALFUNCTIONS + 1:
                    # 3rd consecutive malfunction counts as a fail attempt
                    _attempt_counts[skill_name] += 1
                    _malfunction_counts[skill_name] = 0
                    attempt = _attempt_counts[skill_name]

                    if attempt >= MAX_VERIFICATION_ATTEMPTS:
                        trace_path = write_failure_trace(
                            skill_name=skill_name,
                            history=_attempt_history[skill_name],
                            output_dir=verify_output_dir,
                            workspace=Path(working_directory),
                        )
                        raise VerificationExhaustedError(skill_name, trace_path)

                return ToolResult(
                    text_result_for_llm=(
                        f"Verification MALFUNCTION for skill '{skill_name}' "
                        f"(consecutive: {consecutive_malfunctions}). "
                        f"The verifier encountered an error: {vresult.raw_output} "
                        f"Try calling CompleteSkill again."
                    ),
                    result_type="error",
                )
            else:
                # FAIL or PARTIAL
                _attempt_counts[skill_name] += 1
                _malfunction_counts[skill_name] = 0  # reset consecutive malfunctions
                attempt = _attempt_counts[skill_name]

                # Record in history
                _attempt_history[skill_name].append({
                    "verdict": vresult.verdict,
                    "failed_checks": vresult.failed_checks,
                    "raw_output": vresult.raw_output,
                    "timestamp": time.time(),
                })

                if attempt >= MAX_VERIFICATION_ATTEMPTS:
                    # Exhausted — write trace and raise
                    trace_path = write_failure_trace(
                        skill_name=skill_name,
                        history=_attempt_history[skill_name],
                        output_dir=verify_output_dir,
                        workspace=Path(working_directory),
                    )
                    raise VerificationExhaustedError(skill_name, trace_path)

                feedback = format_fail_feedback(
                    failed_checks=vresult.failed_checks,
                    attempt=attempt,
                    max_attempts=MAX_VERIFICATION_ATTEMPTS,
                )
                return ToolResult(
                    text_result_for_llm=feedback,
                    result_type="error",
                )

        # Mark as complete
        completed_skills.add(skill_name)

        # Build provenance record
        source_file = args.get("source_file", "")
        output_summary = args.get("output_summary", "")
        row_count = args.get("row_count")

        provenance = {
            "skill": skill_name,
            "type": fm.get("type", ""),
            "source_file": source_file,
            "output_summary": output_summary,
            "row_count": row_count,
            "timestamp": time.time(),
        }

        # Find downstream skills that are now unblocked
        skill_type = fm.get("type", "").lower()
        unblocked = []
        for name, other_fm in skill_map.items():
            req = other_fm.get("requires", "").strip().lower()
            if req and req == skill_type and name not in completed_skills:
                unblocked.append(name)

        result_parts = [
            f"Skill '{skill_name}' marked as complete.",
        ]
        if source_file:
            result_parts.append(f"Source: {source_file}")
        if row_count is not None:
            result_parts.append(f"Rows processed: {row_count}")
        if unblocked:
            result_parts.append(
                f"Now unblocked: {', '.join(unblocked)}. "
                f"You should invoke these next."
            )
        else:
            result_parts.append("No downstream skills to unblock.")

        return ToolResult(
            text_result_for_llm="\n".join(result_parts),
        )

    return Tool(
        name="CompleteSkill",
        description=(
            "Report that a skill has been completed. Call this AFTER you have "
            "written all outputs and verified they are correct. This is REQUIRED — "
            "without it, downstream skills that depend on this one will be blocked. "
            "Do NOT call this until your outputs actually exist and are substantive."
        ),
        handler=_handle_complete_skill,
        parameters=COMPLETE_SKILL_PARAMETERS,
        skip_permission=True,
    )
