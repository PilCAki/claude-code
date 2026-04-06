from __future__ import annotations

from dataclasses import replace
import importlib.util
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING, Any, Callable
import uuid

from .agents import build_default_custom_agents
from .compaction import CompactionMode, CompactionResult, build_compaction_prompt, format_transcript_for_compaction, parse_compaction_response
from .config import CopilotCodeConfig, DEFAULT_SKILL_NAMES
from .hooks import build_default_hooks
from .instructions import InstructionBundle
from .memory import MemoryStore
from .permissions import build_permission_handler
from .prompt_compiler import (
    PromptAssembler,
    build_assembler,
    build_system_message,
    materialize_workspace_instructions,
)
from .reports import CheckResult, PreflightReport, SmokeTestReport
from .skill_assets import build_skill_catalog
from .events import (
    EventBus,
    cost_accumulated as _cost_event,
    model_switched as _model_switched_event,
    session_destroyed as _session_destroyed_event,
    session_started as _session_started_event,
    turn_completed as _turn_completed_event,
    turn_started as _turn_started_event,
)
from .session_memory import SessionMemoryController
from .subagent import SubagentContext, SubagentSpec, build_subagent_context
from .session_state import SessionState, SessionStatus
from .model_cost import UsageCost, calculate_cost
from .tokenizer import count_message_tokens
from .tasks import TaskRecord, TaskStore
from .task_tools import build_task_tools
from .skill_tool import build_complete_skill_tool, build_skill_tool

if TYPE_CHECKING:
    from copilot import CopilotClient as SDKCopilotClient
    from copilot.session import CopilotSession as SDKCopilotSession


def _load_copilot_sdk() -> tuple[Any, Any, Any, Any]:
    try:
        from copilot import CopilotClient
        from copilot.types import PermissionHandler, SubprocessConfig
        import copilot
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "github-copilot-sdk is required to use CopilotCodeClient. "
            "Install it with `pip install github-copilot-sdk`.",
        ) from exc
    return CopilotClient, PermissionHandler, SubprocessConfig, copilot


def _last_turn_has_tool_calls(messages: list[Any]) -> bool:
    """Check if the most recent assistant turn contains tool calls.

    Handles both plain dicts (``{"role": "assistant", "content": ...}``)
    and SDK ``SessionEvent.to_dict()`` output
    (``{"type": "assistant.message", "data": {"tool_requests": [...], ...}}``).
    """
    for msg in reversed(messages):
        # Determine role — plain dict or SDK event dict
        event_type: str = msg.get("type", "")
        role: str = msg.get("role", "")
        data: dict = msg.get("data") or {}
        if not role and isinstance(data, dict):
            role = data.get("role", "")

        is_assistant = (
            role == "assistant"
            or str(event_type).startswith("assistant.")
            or str(event_type).startswith("tool.")
        )
        is_user = role == "user" or str(event_type) == "user.message"

        if is_assistant:
            # SDK event format — check data.tool_requests
            if isinstance(data, dict):
                tool_requests = data.get("tool_requests") or data.get("toolRequests")
                if tool_requests:
                    return True
            # Plain dict format — check content blocks
            content = msg.get("content", "")
            if isinstance(content, list):
                return any(
                    isinstance(block, dict) and block.get("type") == "tool_use"
                    for block in content
                )
            if "tool_use" in str(content):
                return True
            # Check for tool execution events
            if str(event_type).startswith("tool.execution"):
                return True
        if is_user:
            break
    return False


def _normalize_sdk_payload(value: Any) -> Any:
    """Convert SDK event objects into JSON-serializable Python values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _normalize_sdk_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_sdk_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_sdk_payload(item) for item in value]

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _normalize_sdk_payload(to_dict())

    try:
        return _normalize_sdk_payload(vars(value))
    except TypeError:
        return str(value)


def _normalize_sdk_messages(messages: list[Any]) -> list[Any]:
    return [_normalize_sdk_payload(message) for message in messages]


class CopilotCodeSession:
    """Thin wrapper over a Copilot SDK session with memory helpers."""

    def __init__(
        self,
        session: "SDKCopilotSession",
        memory_store: MemoryStore,
        *,
        session_memory_controller: SessionMemoryController | None = None,
        copilot_client: Any = None,
        model: str | None = None,
        task_store: "TaskStore | None" = None,
    ) -> None:
        self._session = session
        self._memory_store = memory_store
        self._smc = session_memory_controller
        self._copilot_client = copilot_client
        self._task_store = task_store
        self._pending_initial_message: str | None = None
        self._instruction_bundle: InstructionBundle | None = None
        self._state = SessionState()
        self._model = model
        self._model_override: str | None = None
        self._cumulative_cost = UsageCost(0.0, 0.0, 0.0, 0.0)
        self._event_bus = EventBus()
        self._subagent_context: SubagentContext | None = None
        self._drain_skills: Callable[[], list[dict[str, Any]]] | None = None
        self._completed_skills: set[str] | None = None

    @property
    def raw_session(self) -> "SDKCopilotSession":
        return self._session

    @property
    def workspace_path(self) -> str | None:
        return self._session.workspace_path

    @property
    def session_id(self) -> str | None:
        return getattr(self._session, "session_id", None)

    @property
    def event_bus(self) -> EventBus:
        """The event bus for this session."""
        return self._event_bus

    @property
    def state(self) -> SessionState:
        """The session state machine tracking lifecycle and token usage."""
        return self._state

    @property
    def cumulative_cost(self) -> UsageCost:
        """Cumulative API cost for this session."""
        return self._cumulative_cost

    @property
    def completed_skills(self) -> set[str]:
        """Skills that have been marked complete via CompleteSkill."""
        return self._completed_skills if self._completed_skills is not None else set()

    @property
    def active_model(self) -> str | None:
        """The currently active model (override or default)."""
        return self._model_override or self._model

    def switch_model(self, model: str | None) -> str | None:
        """Switch to a different model. Pass None to revert to default.

        Returns the previously active model override.
        """
        previous = self._model_override
        self._model_override = model
        # Update model for cost tracking
        if model is not None:
            self._event_bus.emit(_model_switched_event(
                from_model=self._model or "",
                to_model=model,
            ))
            self._model = model
        return previous

    def toggle_fast_mode(self, fast_model: str | None = None) -> bool:
        """Toggle fast mode. Returns True if now in fast mode."""
        if self._model_override is not None:
            # Currently in override (fast) mode — revert
            self._model_override = None
            return False
        # Enter fast mode
        self._model_override = fast_model
        return fast_model is not None

    @property
    def task_store(self) -> "TaskStore | None":
        """The task store for this session, if any."""
        return self._task_store

    @property
    def task_list_id(self) -> str | None:
        """The task list ID for this session's task store."""
        if self._task_store is not None:
            return self._task_store.task_list_id
        return None

    def create_task(
        self,
        subject: str,
        *,
        description: str = "",
        owner: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "TaskRecord":
        """Create a task in this session's task store."""
        if self._task_store is None:
            raise RuntimeError("No task store configured for this session")
        return self._task_store.create(
            subject, description=description, owner=owner, metadata=metadata,
        )

    def update_task(
        self,
        task_id: int,
        **kwargs: Any,
    ) -> "TaskRecord | None":
        """Update a task in this session's task store."""
        if self._task_store is None:
            raise RuntimeError("No task store configured for this session")
        return self._task_store.update(task_id, **kwargs)

    def list_tasks(self) -> "list[TaskRecord]":
        """List all tasks in this session's task store."""
        if self._task_store is None:
            return []
        return self._task_store.list_all()

    def get_task(self, task_id: int) -> "TaskRecord | None":
        """Get a specific task from this session's task store."""
        if self._task_store is None:
            return None
        return self._task_store.get(task_id)

    def get_task_output(self, task_id: int) -> str | None:
        """Read the output artifact for a task."""
        if self._task_store is None:
            return None
        return self._task_store.read_task_output(task_id)

    @property
    def subagent_context(self) -> SubagentContext | None:
        """The subagent context for forking child sessions."""
        return self._subagent_context

    def set_cacheable_prefix(self, prefix: str) -> None:
        """Set the cacheable prefix and initialize the subagent context.

        Called by the client after session creation to share the parent's
        cacheable prompt with future child sessions.
        """
        self._subagent_context = build_subagent_context(
            session_id=self.session_id or "unknown",
            cacheable_prefix=prefix,
        )

    async def fork_child(self, spec: SubagentSpec) -> "EnforcedChildSession":
        """Fork a child session that shares this session's cacheable prefix.

        The child gets the same static system prompt prefix (for cache hits)
        plus the spec's suffix. Returns an :class:`EnforcedChildSession` that
        enforces ``spec.max_turns`` and ``spec.timeout_seconds``.
        """
        from .subagent import ChildSession, EnforcedChildSession

        if self._copilot_client is None:
            raise RuntimeError("No copilot client available for forking")
        if self._subagent_context is None:
            raise RuntimeError("No subagent context — call set_cacheable_prefix first")

        system_message = self._subagent_context.build_child_system_message(spec)
        create_kwargs: dict[str, Any] = {
            "system_message": system_message,
            # Children auto-approve all permissions — the parent session's
            # policy already governs what's allowed.
            "on_permission_request": lambda _: True,
        }
        # Pass tool allowlist if spec restricts tools
        if spec.tools:
            create_kwargs["tools"] = [{"name": t} for t in spec.tools]
        child = await self._copilot_client.create_session(**create_kwargs)
        child_id = getattr(child, "session_id", None) or "child"
        self._subagent_context.register_child(child_id)
        return EnforcedChildSession(ChildSession(session=child, spec=spec, session_id=child_id))

    async def process_pending_skills(self) -> list[str]:
        """Drain and execute any queued InvokeSkill invocations.

        Returns a list of result summaries (one per executed skill).
        Call this between turns to process skills that were queued by
        the ``on_post_tool_use`` hook.
        """
        if self._drain_skills is None:
            return []
        invocations = self._drain_skills()
        if not invocations:
            return []

        results: list[str] = []
        for invocation in invocations:
            skill_name = invocation.get("skill_name", "unknown")
            user_prompt = invocation.get("user_prompt", "")
            try:
                spec = SubagentSpec(
                    role=f"skill:{skill_name}",
                    system_prompt_suffix=user_prompt,
                    max_turns=20,
                    timeout_seconds=300.0,
                )
                child = await self.fork_child(spec)
                result = await child.session.send_and_wait(user_prompt, timeout=300.0)
                await child.session.destroy()
                summary = f"Skill '{skill_name}' completed."
                if isinstance(result, str):
                    summary = result[:500]
                elif isinstance(result, dict):
                    summary = str(result.get("content", summary))[:500]
                results.append(summary)
                if self._completed_skills is not None:
                    self._completed_skills.add(skill_name)
            except Exception as exc:
                results.append(f"Skill '{skill_name}' failed: {exc}")
        return results

    @property
    def pending_initial_message(self) -> str | None:
        """A hook-generated initial message to send as the first user prompt."""
        return self._pending_initial_message

    @property
    def instruction_bundle(self) -> InstructionBundle | None:
        """The instruction bundle loaded at session start, if any."""
        return self._instruction_bundle

    async def send(
        self,
        prompt: str,
        *,
        attachments: list[dict[str, Any]] | None = None,
        mode: str | None = None,
    ) -> str:
        return await self._session.send(prompt, attachments=attachments, mode=mode)

    async def stream(
        self,
        prompt: str,
        *,
        attachments: list[dict[str, Any]] | None = None,
        mode: str | None = None,
    ):
        """Stream response chunks from the model.

        Yields dicts with at least a ``"type"`` key. The underlying SDK
        must expose a ``stream()`` async generator. If it doesn't, this
        falls back to ``send_and_wait()`` and yields a single chunk.
        """
        self._state.start_turn()
        raw_stream = getattr(self._session, "stream", None)
        if raw_stream is not None and callable(raw_stream):
            try:
                async for chunk in raw_stream(prompt, attachments=attachments, mode=mode):
                    yield chunk
                    # Accumulate usage from final chunk if present
                    if isinstance(chunk, dict) and chunk.get("type") == "message_stop":
                        self._accumulate_usage(chunk)
            finally:
                self._state.end_turn()
        else:
            # Fallback: run send_and_wait and yield as single chunk
            try:
                result = await self._session.send_and_wait(
                    prompt, attachments=attachments, mode=mode, timeout=3600.0,
                )
                self._accumulate_usage(result)
                yield {"type": "message_stop", "result": result}
            finally:
                self._state.end_turn()

    async def send_and_wait(
        self,
        prompt: str,
        *,
        attachments: list[dict[str, Any]] | None = None,
        mode: str | None = None,
        timeout: float = 3600.0,
    ) -> Any:
        self._state.start_turn()
        self._event_bus.emit(_turn_started_event(turn_count=self._state.turn_count))
        try:
            result = await self._session.send_and_wait(
                prompt,
                attachments=attachments,
                mode=mode,
                timeout=timeout,
            )
        except Exception:
            self._state.end_turn()
            raise

        # Estimate token usage from result and accumulate cost
        self._accumulate_usage(result)
        self._event_bus.emit(_turn_completed_event(
            turn_count=self._state.turn_count,
            input_tokens=self._state.total_input_tokens,
            output_tokens=self._state.total_output_tokens,
        ))
        self._state.end_turn()

        # Post-turn session memory maintenance
        if self._smc is not None:
            await self._maybe_run_session_memory()
        return result

    async def send_until_complete(
        self,
        prompt: str,
        *,
        completion_check: Callable[[], list[str]] | None = None,
        max_continuations: int = 5,
        continuation_template: str = (
            "You have NOT completed all skills. "
            "The following output directories are still missing or empty: {missing}. "
            "Continue executing skills in dependency order using InvokeSkill. "
            "Do not stop until all skills are complete and all output directories have content."
        ),
        timeout: float = 3600.0,
    ) -> Any:
        """Send a prompt and retry until ``completion_check`` passes.

        After each ``send_and_wait`` call, ``completion_check`` is invoked. If
        it returns a non-empty list of missing items, a continuation prompt is
        sent with those items interpolated via ``{missing}`` in the template.

        Returns the final result once all checks pass or
        ``max_continuations`` is exhausted.
        """
        result = await self.send_and_wait(prompt, timeout=timeout)

        if completion_check is None:
            return result

        for attempt in range(1, max_continuations + 1):
            missing = completion_check()
            if not missing:
                break
            continuation = continuation_template.format(
                missing=", ".join(missing),
            )
            result = await self.send_and_wait(continuation, timeout=timeout)
        return result

    async def get_messages(self) -> list[Any]:
        return await self._session.get_messages()

    async def disconnect(self) -> None:
        await self._session.disconnect()

    async def destroy(self) -> None:
        if self._smc is not None:
            # Force a final extraction pass before promoting, regardless of
            # thresholds.  This matches Claude Code's handleStopHooks behavior
            # where memory extraction fires unconditionally at session end.
            await self._force_final_extraction()
            await self._smc.finalize()
        self._event_bus.emit(_session_destroyed_event(session_id=self.session_id or ""))
        await self._session.destroy()

    async def _force_final_extraction(self) -> None:
        """Run one final session-memory extraction pass at session end.

        Unlike the threshold-gated ``_maybe_run_session_memory``, this fires
        unconditionally so that even short or thin sessions get captured.
        """
        if self._smc is None:
            return
        try:
            raw_messages = await self.get_messages()
        except Exception:
            return
        messages: list[Any] = [
            m.to_dict() if hasattr(m, "to_dict") else m
            for m in raw_messages
        ]
        if not messages:
            return
        context_tokens = count_message_tokens(messages)
        await self._smc.run_extraction(
            self._create_maintenance_session,
            messages,
            context_tokens=context_tokens,
        )

    def _accumulate_usage(self, result: Any) -> None:
        """Extract token usage from SDK result and accumulate cost."""
        # Convert SessionEvent to dict if needed (SDK returns dataclass objects)
        if hasattr(result, "to_dict"):
            result = result.to_dict()
        if not isinstance(result, dict):
            return
        usage = result.get("usage") or result.get("token_usage") or {}
        if not isinstance(usage, dict):
            return
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        cache_read = int(usage.get("cache_read_input_tokens", usage.get("cache_read_tokens", 0)))
        cache_create = int(usage.get("cache_creation_input_tokens", usage.get("cache_creation_tokens", 0)))
        self._state.record_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_create,
        )
        if self._model:
            turn_cost = calculate_cost(
                self._model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read,
                cache_creation_tokens=cache_create,
            )
            self._cumulative_cost = UsageCost(
                input_cost=self._cumulative_cost.input_cost + turn_cost.input_cost,
                output_cost=self._cumulative_cost.output_cost + turn_cost.output_cost,
                cache_read_cost=self._cumulative_cost.cache_read_cost + turn_cost.cache_read_cost,
                cache_creation_cost=self._cumulative_cost.cache_creation_cost + turn_cost.cache_creation_cost,
            )
            self._event_bus.emit(_cost_event(
                total_cost=self._cumulative_cost.total,
                turn_cost=turn_cost.total,
                model=self._model or "",
            ))

    async def _maybe_run_session_memory(self) -> None:
        """Post-turn check: run session memory extraction if thresholds met."""
        if self._smc is None:
            return
        try:
            raw_messages = await self.get_messages()
        except Exception:
            return
        # SDK returns SessionEvent objects; convert to dicts for downstream use
        messages: list[Any] = [
            m.to_dict() if hasattr(m, "to_dict") else m
            for m in raw_messages
        ]
        # Track tool calls for extraction threshold
        has_tool_calls = _last_turn_has_tool_calls(messages)
        if has_tool_calls:
            self._smc.record_tool_call()
        # Count context tokens (uses tiktoken if available, else char/4)
        context_tokens = count_message_tokens(messages)
        if self._smc.should_extract(
            context_tokens, has_tool_calls_in_last_turn=has_tool_calls,
        ):
            await self._smc.run_extraction(
                self._create_maintenance_session,
                messages,
                context_tokens=context_tokens,
            )

    async def _create_maintenance_session(self) -> Any:
        """Create a short-lived session for the maintenance pass.

        When a subagent context exists, the maintenance session shares the
        parent's cacheable prefix for API cache hits.
        """
        if self._copilot_client is None:
            raise RuntimeError("No copilot client available for maintenance session")

        if self._subagent_context is not None:
            system_message = self._subagent_context.build_maintenance_system_message()
        else:
            system_message = {
                "mode": "replace",
                "content": (
                    "You are a session-memory maintenance agent. "
                    "Respond only with the updated notes document."
                ),
            }
        session = await self._copilot_client.create_session(
            system_message=system_message,
            on_permission_request=lambda _: True,  # auto-approve (text-only session)
        )
        if self._subagent_context is not None:
            child_id = getattr(session, "session_id", None) or "maintenance"
            self._subagent_context.register_child(child_id)
        return session

    async def compact_for_handoff(
        self,
        *,
        mode: CompactionMode = "full",
        extra_instructions: str = "",
    ) -> CompactionResult:
        """Run a compaction pass and persist the result for future resume.

        Creates a short-lived maintenance session (no tools), sends the
        compaction prompt, parses the ``<analysis>``/``<summary>`` response,
        and writes the summary to ``<memory_home>/compaction/<session_id>.md``.
        """
        prompt = build_compaction_prompt(
            mode=mode,
            extra_instructions=extra_instructions,
        )
        # Gather the actual conversation transcript so the maintenance
        # agent has real context to summarize (not just the instructions).
        parent_messages = _normalize_sdk_messages(await self.get_messages())
        transcript_block = format_transcript_for_compaction(parent_messages)
        full_prompt = f"{transcript_block}\n\n{prompt}"

        maintenance = await self._create_maintenance_session()
        try:
            event = await maintenance.send_and_wait(full_prompt, timeout=600.0)
            messages = _normalize_sdk_messages(await maintenance.get_messages())
            # Extract the last assistant message text
            response_text = ""
            for msg in reversed(messages):
                role = msg.get("role") or msg.get("type") or ""
                if role == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        response_text = content
                    elif isinstance(content, list):
                        response_text = " ".join(
                            block.get("text", "")
                            for block in content
                            if isinstance(block, dict) and block.get("type") == "text"
                        )
                    break
            if not response_text and event is not None:
                response_text = str(event)
        finally:
            await maintenance.destroy()

        result = parse_compaction_response(response_text)

        # Persist to <memory_home>/compaction/<session_id>.md
        session_id = self.session_id or "unknown"
        compaction_dir = self._memory_store.memory_dir / "compaction"
        compaction_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = compaction_dir / f"{session_id}.md"
        artifact_path.write_text(result.summary, encoding="utf-8")

        return result

    def remember(
        self,
        *,
        title: str,
        description: str,
        memory_type: str,
        content: str,
        slug: str | None = None,
    ) -> Path:
        return self._memory_store.upsert_memory(
            title=title,
            description=description,
            memory_type=memory_type,  # type: ignore[arg-type]
            content=content,
            slug=slug,
        )

    def forget(self, slug_or_path: str | Path) -> None:
        self._memory_store.delete_memory(slug_or_path)

    def relevant_memories(self, query: str, *, limit: int = 5) -> list[Any]:
        return self._memory_store.select_relevant(query, limit=limit)

    def reindex_memories(self) -> str:
        return self._memory_store.reindex()


class CopilotCodeClient:
    """High-level wrapper around the Copilot Python SDK."""

    def __init__(
        self,
        config: CopilotCodeConfig | None = None,
        *,
        copilot_client: "SDKCopilotClient | None" = None,
    ) -> None:
        self.config = config or CopilotCodeConfig()
        self._memory_store = MemoryStore(
            self.config.working_path,
            self.config.memory_home,
            brand=self.config.brand,
        )
        self._task_store: TaskStore | None = None
        if self.config.enable_tasks_v2:
            task_root = (
                Path(self.config.task_root)
                if self.config.task_root is not None
                else self.config.memory_home / "tasks"
            )
            persist_path = task_root / "tasks.json"
            self._task_store = TaskStore(persist_path=persist_path)
        self._client = copilot_client
        self._assembler: PromptAssembler | None = None
        self._smc: SessionMemoryController | None = None
        if self.config.session_memory_auto:
            self._smc = SessionMemoryController(
                self._memory_store,
                min_init_tokens=self.config.session_memory_min_init_tokens,
                min_update_tokens=self.config.session_memory_min_update_tokens,
                tool_calls_between_updates=self.config.session_memory_tool_calls_between_updates,
                timeout_seconds=self.config.session_memory_timeout_seconds,
                promote_on_destroy=self.config.session_memory_promote_on_destroy,
            )

    @property
    def raw_client(self) -> "SDKCopilotClient":
        return self._ensure_client()

    @property
    def memory_store(self) -> MemoryStore:
        return self._memory_store

    @property
    def task_store(self) -> TaskStore | None:
        return self._task_store

    @property
    def assembler(self) -> PromptAssembler | None:
        """The prompt assembler from the most recent session, if any."""
        return self._assembler

    def build_system_message(self) -> str:
        if self._assembler is not None:
            return self._assembler.render()
        return build_system_message(self.config)

    def materialize_workspace_instructions(
        self,
        root: str | Path | None = None,
        *,
        overwrite: bool = False,
    ) -> tuple[Path, Path]:
        return materialize_workspace_instructions(
            root or self.config.working_path,
            self.config,
            overwrite=overwrite,
        )

    def preflight(self, *, require_auth: bool = False) -> PreflightReport:
        return self._build_preflight_report(require_auth=require_auth)

    async def smoke_test(
        self,
        *,
        live: bool = False,
        prompt: str = "Reply with the single word OK.",
        timeout: float = 300.0,
        save_report_path: str | Path | None = None,
        save_transcript_dir: str | Path | None = None,
    ) -> SmokeTestReport:
        preflight = self._build_preflight_report(require_auth=live)
        if not preflight.ok:
            report = SmokeTestReport(
                product_name=self.config.brand.public_name,
                live=live,
                success=False,
                preflight=preflight,
                session_created=False,
                prompt_roundtrip=False,
                prompt=prompt if live else None,
                detail="Preflight failed.",
                error="One or more preflight checks returned an error.",
            )
            return _finalize_smoke_report(report, save_report_path=save_report_path)

        if not live:
            report = SmokeTestReport(
                product_name=self.config.brand.public_name,
                live=False,
                success=True,
                preflight=preflight,
                session_created=False,
                prompt_roundtrip=False,
                detail="Dry run only. No live Copilot session was attempted.",
            )
            return _finalize_smoke_report(report, save_report_path=save_report_path)

        session: CopilotCodeSession | None = None
        session_id = f"{self.config.brand.slug}-smoke-{uuid.uuid4().hex}"
        messages: list[Any] = []
        try:
            session = await self.create_session(session_id=session_id)
            event = await session.send_and_wait(prompt, timeout=timeout)
            messages = await session.get_messages()
            report = SmokeTestReport(
                product_name=self.config.brand.public_name,
                live=True,
                success=event is not None,
                preflight=preflight,
                session_created=True,
                prompt_roundtrip=event is not None,
                session_id=session.session_id or session_id,
                workspace_path=session.workspace_path,
                prompt=prompt,
                detail=f"Received {len(messages)} session event(s).",
            )
            transcript_path = _write_transcript_artifact(
                report,
                messages,
                save_transcript_dir,
            )
            if transcript_path is not None:
                report = replace(report, transcript_path=str(transcript_path))
            return _finalize_smoke_report(report, save_report_path=save_report_path)
        except Exception as exc:  # pragma: no cover - exercised in live mode only
            if session is not None:
                try:
                    messages = await session.get_messages()
                except Exception:
                    messages = []
            report = SmokeTestReport(
                product_name=self.config.brand.public_name,
                live=True,
                success=False,
                preflight=preflight,
                session_created=session is not None,
                prompt_roundtrip=False,
                session_id=session.session_id if session is not None else session_id,
                workspace_path=session.workspace_path if session else None,
                prompt=prompt,
                error=str(exc),
            )
            transcript_path = _write_transcript_artifact(
                report,
                messages,
                save_transcript_dir,
            )
            if transcript_path is not None:
                report = replace(report, transcript_path=str(transcript_path))
            return _finalize_smoke_report(report, save_report_path=save_report_path)
        finally:
            if session is not None:
                await session.disconnect()

    async def create_session(
        self,
        *,
        session_id: str | None = None,
        task_list_id: str | None = None,
        on_event: Any | None = None,
    ) -> CopilotCodeSession:
        task_store = self._resolve_task_store(task_list_id or session_id)
        kwargs = self._session_kwargs(on_event=on_event, source="create", task_store=task_store)
        captured = kwargs.pop("_hook_captured", {})
        session = await self._ensure_client().create_session(
            session_id=session_id,
            **kwargs,
        )
        wrapped = CopilotCodeSession(
            session, self._memory_store,
            session_memory_controller=self._smc,
            copilot_client=self._ensure_client(),
            model=self.config.model,
            task_store=task_store,
        )
        # Share the cacheable prefix with child sessions
        if self._assembler is not None:
            wrapped.set_cacheable_prefix(self._assembler.render(cacheable_only=True))
        # Wire on_event config callback as a global listener on the event bus
        event_callback = on_event or self.config.on_event
        if event_callback is not None:
            wrapped.event_bus.subscribe(lambda e: event_callback(e.to_dict()))
        wrapped.event_bus.emit(_session_started_event(
            session_id=wrapped.session_id or "", source="create",
        ))
        if captured.get("initialUserMessage"):
            wrapped._pending_initial_message = captured["initialUserMessage"]
        if captured.get("instructionsLoaded"):
            wrapped._instruction_bundle = captured["instructionsLoaded"]
        # Wire skill execution plumbing
        raw_hooks = captured.get("_raw_hooks", {})
        drain_fn = raw_hooks.get("drain_pending_skill_invocations")
        if drain_fn is not None:
            wrapped._drain_skills = drain_fn
        wrapped._completed_skills = captured.get("_completed_skills")
        # Wire session into CompleteSkill's verification gate
        session_holder = captured.get("_session_holder")
        if session_holder is not None and isinstance(session_holder, list):
            session_holder.append(wrapped)
        return wrapped

    async def resume_session(
        self,
        session_id: str,
        *,
        task_list_id: str | None = None,
        on_event: Any | None = None,
    ) -> CopilotCodeSession:
        task_store = self._resolve_task_store(task_list_id or session_id)
        kwargs = self._session_kwargs(on_event=on_event, source="resume", session_id=session_id, task_store=task_store)
        captured = kwargs.pop("_hook_captured", {})
        session = await self._ensure_client().resume_session(
            session_id,
            **kwargs,
        )
        wrapped = CopilotCodeSession(
            session, self._memory_store,
            session_memory_controller=self._smc,
            copilot_client=self._ensure_client(),
            model=self.config.model,
            task_store=task_store,
        )
        # Restore cacheable prefix so resumed sessions can fork children
        if self._assembler is not None:
            wrapped.set_cacheable_prefix(self._assembler.render(cacheable_only=True))
        event_callback = on_event or self.config.on_event
        if event_callback is not None:
            wrapped.event_bus.subscribe(lambda e: event_callback(e.to_dict()))
        wrapped.event_bus.emit(_session_started_event(
            session_id=wrapped.session_id or session_id, source="resume",
        ))
        if captured.get("initialUserMessage"):
            wrapped._pending_initial_message = captured["initialUserMessage"]
        if captured.get("instructionsLoaded"):
            wrapped._instruction_bundle = captured["instructionsLoaded"]
        # Wire skill execution plumbing
        raw_hooks = captured.get("_raw_hooks", {})
        drain_fn = raw_hooks.get("drain_pending_skill_invocations")
        if drain_fn is not None:
            wrapped._drain_skills = drain_fn
        wrapped._completed_skills = captured.get("_completed_skills")
        # Wire session into CompleteSkill's verification gate
        session_holder = captured.get("_session_holder")
        if session_holder is not None and isinstance(session_holder, list):
            session_holder.append(wrapped)
        return wrapped

    def _resolve_task_store(self, task_list_id: str | None) -> TaskStore | None:
        """Return a per-session TaskStore when task_list_id is given, else the shared one."""
        if not self.config.enable_tasks_v2:
            return None
        if task_list_id is None:
            return self._task_store
        task_root = (
            Path(self.config.task_root)
            if self.config.task_root is not None
            else self.config.memory_home / "tasks"
        )
        return TaskStore(task_list_id=task_list_id, task_root=task_root)

    def _session_kwargs(
        self,
        *,
        on_event: Any | None = None,
        source: str = "create",
        session_id: str | None = None,
        task_store: TaskStore | None = None,
    ) -> dict[str, Any]:
        # Use provided per-session task store, falling back to client-level shared store
        effective_task_store = task_store if task_store is not None else self._task_store
        _, skill_map = build_skill_catalog(
            self._skill_directories(),
            disabled_skills=self._disabled_skills(),
        )
        self._assembler = build_assembler(
            self.config,
            skill_directories=self._skill_directories(),
            disabled_skills=self._disabled_skills(),
            memory_dir=str(self._memory_store.memory_dir),
        )
        # Single shared set so hooks and InvokeSkill see the same completions
        _shared_completed_skills: set[str] = set()
        raw_hooks = build_default_hooks(
            self.config, self._memory_store,
            skill_map=skill_map,
            task_store=effective_task_store,
            assembler=self._assembler,
            completed_skills=_shared_completed_skills,
            session_memory_controller=self._smc,
        )

        # Wrap on_session_start to inject source/session_id and capture initialUserMessage
        _captured: dict[str, Any] = {
            "_completed_skills": _shared_completed_skills,
            "_raw_hooks": raw_hooks,
        }
        original_session_start = raw_hooks.get("on_session_start")

        def _wrapped_session_start(
            input_data: dict[str, Any],
            env: dict[str, str],
        ) -> dict[str, Any] | None:
            input_data = {**input_data, "source": source}
            if session_id is not None:
                input_data["session_id"] = session_id
            result = original_session_start(input_data, env) if original_session_start else None
            if result and "initialUserMessage" in result:
                _captured["initialUserMessage"] = result.pop("initialUserMessage")
            if result and "_instructionsLoaded" in result:
                _captured["instructionsLoaded"] = result.pop("_instructionsLoaded")
            return result

        hooks = {**raw_hooks, "on_session_start": _wrapped_session_start}

        permission_handler = build_permission_handler(
            policy=self.config.permission_policy,
            permission_handler=self.config.permission_handler,
            allowed_roots=(
                *self.config.allowed_roots,
                self._memory_store.memory_dir,
            ),
            approved_shell_prefixes=self.config.approved_shell_prefixes,
            brand=self.config.brand,
        )

        kwargs: dict[str, Any] = {
            "on_permission_request": permission_handler,
            "client_name": self.config.client_name,
            "reasoning_effort": self.config.reasoning_effort,
            "system_message": {
                "mode": "append",
                "content": self._assembler.render(cacheable_only=True),
            },
            "on_user_input_request": self.config.user_input_handler,
            "hooks": hooks,
            "working_directory": str(self.config.working_path),
            "provider": dict(self.config.provider) if self.config.provider else None,
            "custom_agents": build_default_custom_agents(self.config),
            "agent": self.config.default_agent,
            "config_dir": (
                str(self.config.copilot_config_home)
                if self.config.copilot_config_home is not None
                else None
            ),
            "skill_directories": self._skill_directories(),
            "disabled_skills": self._disabled_skills(),
            "infinite_sessions": self.config.resolved_infinite_session_config(),
            "on_event": on_event or self.config.on_event,
        }
        if self.config.model:
            kwargs["model"] = self.config.model
        # Build tools list: task tools + skill tool
        tools_list: list[Any] = []
        if effective_task_store is not None:
            tools_list.extend(build_task_tools(effective_task_store))
        if skill_map:
            # completed_skills is shared with hooks via the same set reference
            completed_skills = _captured.setdefault("_completed_skills", set())
            # Mutable holder — populated by create_session after the session exists
            session_holder: list[Any] = []
            _captured["_session_holder"] = session_holder
            tools_list.append(build_skill_tool(
                skill_map=skill_map,
                memory_store=self._memory_store,
                working_directory=str(self.config.working_path),
                completed_skills=completed_skills,
                session_holder=session_holder,
            ))
            tools_list.append(build_complete_skill_tool(
                skill_map=skill_map,
                completed_skills=completed_skills,
                working_directory=str(self.config.working_path),
                session_holder=session_holder,
            ))
        if tools_list:
            kwargs["tools"] = tools_list
        # Stash captured hook outputs for the caller (not passed to SDK)
        kwargs["_hook_captured"] = _captured
        cleaned = {key: value for key, value in kwargs.items() if value is not None and not key.startswith("_")}
        cleaned["_hook_captured"] = _captured
        return cleaned

    def _skill_directories(self) -> list[str]:
        packaged_skill_dir = Path(__file__).resolve().parent / "skills"
        directories = [str(packaged_skill_dir)]
        directories.extend(str(Path(path)) for path in self.config.extra_skill_directories)
        return directories

    def _disabled_skills(self) -> list[str]:
        disabled = {
            name
            for name in DEFAULT_SKILL_NAMES
            if name not in self.config.enabled_skills
        }
        disabled.update(self.config.disabled_skills)
        return sorted(disabled)

    def _build_preflight_report(self, *, require_auth: bool) -> PreflightReport:
        checks: list[CheckResult] = []
        cli_path: str | None = None
        package_root: Path | None = None

        try:
            version = importlib_metadata.version("github-copilot-sdk")
            package_root = _copilot_package_root()
            checks.append(
                CheckResult(
                    name="python_sdk",
                    status="ok",
                    message=f"github-copilot-sdk {version} is installed.",
                ),
            )
        except Exception as exc:
            checks.append(
                CheckResult(
                    name="python_sdk",
                    status="error",
                    message="github-copilot-sdk metadata could not be resolved.",
                    detail=str(exc),
                ),
            )

        if self.config.working_path.exists() and self.config.working_path.is_dir():
            checks.append(
                CheckResult(
                    name="working_directory",
                    status="ok",
                    message="Working directory exists.",
                    detail=str(self.config.working_path),
                ),
            )
        else:
            checks.append(
                CheckResult(
                    name="working_directory",
                    status="error",
                    message="Working directory does not exist or is not a directory.",
                    detail=str(self.config.working_path),
                ),
            )

        cli_path = self._resolved_cli_path(package_root)
        if cli_path:
            checks.append(
                CheckResult(
                    name="copilot_cli",
                    status="ok",
                    message="Copilot CLI is available.",
                    detail=cli_path,
                ),
            )
        else:
            checks.append(
                CheckResult(
                    name="copilot_cli",
                    status="error",
                    message="Copilot CLI executable was not found.",
                    detail="Install Copilot CLI or use a platform wheel that bundles the binary.",
                ),
            )

        checks.append(_directory_check("app_config_directory", self.config.app_config_home))
        checks.append(_directory_check("memory_directory", self._memory_store.memory_dir))

        auth_check = self._auth_check(require_auth=require_auth)
        checks.append(auth_check)

        return PreflightReport(
            product_name=self.config.brand.public_name,
            require_auth=require_auth,
            working_directory=str(self.config.working_path),
            app_config_directory=str(self.config.app_config_home),
            memory_directory=str(self._memory_store.memory_dir),
            copilot_config_directory=str(
                self.config.copilot_config_home
                or self.config.brand.copilot_default_config_home()
            ),
            cli_path=cli_path,
            checks=tuple(checks),
        )

    def _auth_check(self, *, require_auth: bool) -> CheckResult:
        if self.config.github_token:
            return CheckResult(
                name="auth",
                status="ok",
                message="GitHub token is configured.",
            )

        config_home = (
            self.config.copilot_config_home
            or self.config.brand.copilot_default_config_home()
        )
        auth_config = config_home / "config.json"
        if auth_config.exists():
            return CheckResult(
                name="auth",
                status="ok",
                message="Copilot CLI config file exists, so auth is likely configured.",
                detail=str(auth_config),
            )

        return CheckResult(
            name="auth",
            status="error" if require_auth else "warning",
            message="No GitHub token was provided and no Copilot CLI config file was found.",
            detail=(
                f"Expected a config file at `{auth_config}` or a configured github_token. "
                f"Run `{self.config.brand.cli_name} preflight` after `copilot login` or pass github_token explicitly."
            ),
        )

    def _resolved_cli_path(self, package_root: Path | None = None) -> str | None:
        if self.config.cli_path:
            resolved = shutil.which(self.config.cli_path) or self.config.cli_path
            return resolved if Path(resolved).exists() else None

        shell_path = shutil.which("copilot")
        if shell_path:
            return shell_path

        if package_root is None:
            return None

        candidates = [
            package_root / "bin" / "copilot.exe",
            package_root / "bin" / "copilot",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    def _ensure_client(self) -> "SDKCopilotClient":
        if self._client is not None:
            return self._client

        CopilotClient, _, SubprocessConfig, _ = _load_copilot_sdk()
        subprocess_config = SubprocessConfig(
            cli_path=self._resolved_cli_path(_copilot_package_root()),
            cli_args=list(self.config.cli_args),
            cwd=str(self.config.working_path),
            use_stdio=self.config.cli_use_stdio,
            port=self.config.cli_port,
            log_level=self.config.cli_log_level,
            env=dict(self.config.cli_env) if self.config.cli_env else None,
            github_token=self.config.github_token,
            use_logged_in_user=self.config.use_logged_in_user,
        )
        self._client = CopilotClient(subprocess_config)
        return self._client

def _directory_check(name: str, path: Path) -> CheckResult:
    if path.exists() and not path.is_dir():
        return CheckResult(
            name=name,
            status="error",
            message="Path exists but is not a directory.",
            detail=str(path),
        )

    anchor = path if path.exists() else _nearest_existing_parent(path)
    try:
        anchor.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=anchor, delete=True):
            pass
    except OSError as exc:
        return CheckResult(
            name=name,
            status="error",
            message="Directory is not writable.",
            detail=f"{path}: {exc}",
        )

    message = (
        "Directory exists and is writable."
        if path.exists()
        else "Parent directory is writable, so this directory can be created."
    )
    return CheckResult(
        name=name,
        status="ok",
        message=message,
        detail=str(path),
    )


def _nearest_existing_parent(path: Path) -> Path:
    current = path.resolve(strict=False)
    while not current.exists():
        if current.parent == current:
            return current
        current = current.parent
    return current


def _copilot_package_root() -> Path | None:
    spec = importlib.util.find_spec("copilot")
    if spec is None:
        return None

    if spec.submodule_search_locations:
        location = next(iter(spec.submodule_search_locations), None)
        if location:
            return Path(location).expanduser().resolve(strict=False)

    if spec.origin:
        return Path(spec.origin).expanduser().resolve(strict=False).parent

    return None


def _write_transcript_artifact(
    report: SmokeTestReport,
    messages: list[Any],
    save_transcript_dir: str | Path | None,
) -> Path | None:
    if save_transcript_dir is None or not messages:
        return None

    directory = Path(save_transcript_dir).expanduser().resolve(strict=False)
    directory.mkdir(parents=True, exist_ok=True)
    session_stub = report.session_id or "copilotcode-smoke"
    transcript_path = directory / f"{session_stub}-transcript.json"
    payload = {
        "product_name": report.product_name,
        "session_id": report.session_id,
        "live": report.live,
        "workspace_path": report.workspace_path,
        "prompt": report.prompt,
        "messages": _normalize_sdk_messages(messages),
    }
    transcript_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return transcript_path


def _finalize_smoke_report(
    report: SmokeTestReport,
    *,
    save_report_path: str | Path | None,
) -> SmokeTestReport:
    if save_report_path is None:
        return report

    path = Path(save_report_path).expanduser().resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = replace(report, report_path=str(path))
    path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report
