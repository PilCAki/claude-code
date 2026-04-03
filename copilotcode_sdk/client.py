from __future__ import annotations

from dataclasses import replace
import importlib.util
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING, Any
import uuid

from .agents import build_default_custom_agents
from .config import CopilotCodeConfig, DEFAULT_SKILL_NAMES
from .hooks import build_default_hooks
from .memory import MemoryStore
from .permissions import build_permission_handler
from .prompt_compiler import (
    build_system_message,
    materialize_workspace_instructions,
)
from .reports import CheckResult, PreflightReport, SmokeTestReport
from .skill_assets import build_skill_catalog
from .tasks import TaskStore
from .task_tools import build_task_tools

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


class CopilotCodeSession:
    """Thin wrapper over a Copilot SDK session with memory helpers."""

    def __init__(
        self,
        session: "SDKCopilotSession",
        memory_store: MemoryStore,
    ) -> None:
        self._session = session
        self._memory_store = memory_store

    @property
    def raw_session(self) -> "SDKCopilotSession":
        return self._session

    @property
    def workspace_path(self) -> str | None:
        return self._session.workspace_path

    @property
    def session_id(self) -> str | None:
        return getattr(self._session, "session_id", None)

    async def send(
        self,
        prompt: str,
        *,
        attachments: list[dict[str, Any]] | None = None,
        mode: str | None = None,
    ) -> str:
        return await self._session.send(prompt, attachments=attachments, mode=mode)

    async def send_and_wait(
        self,
        prompt: str,
        *,
        attachments: list[dict[str, Any]] | None = None,
        mode: str | None = None,
        timeout: float = 60.0,
    ) -> Any:
        return await self._session.send_and_wait(
            prompt,
            attachments=attachments,
            mode=mode,
            timeout=timeout,
        )

    async def get_messages(self) -> list[Any]:
        return await self._session.get_messages()

    async def disconnect(self) -> None:
        await self._session.disconnect()

    async def destroy(self) -> None:
        await self._session.destroy()

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

    @property
    def raw_client(self) -> "SDKCopilotClient":
        return self._ensure_client()

    @property
    def memory_store(self) -> MemoryStore:
        return self._memory_store

    @property
    def task_store(self) -> TaskStore | None:
        return self._task_store

    def build_system_message(self) -> str:
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
        timeout: float = 60.0,
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
        on_event: Any | None = None,
    ) -> CopilotCodeSession:
        kwargs = self._session_kwargs(on_event=on_event)
        session = await self._ensure_client().create_session(
            session_id=session_id,
            **kwargs,
        )
        return CopilotCodeSession(session, self._memory_store)

    async def resume_session(
        self,
        session_id: str,
        *,
        on_event: Any | None = None,
    ) -> CopilotCodeSession:
        kwargs = self._session_kwargs(on_event=on_event)
        session = await self._ensure_client().resume_session(
            session_id,
            **kwargs,
        )
        return CopilotCodeSession(session, self._memory_store)

    def _session_kwargs(self, *, on_event: Any | None = None) -> dict[str, Any]:
        _, skill_map = build_skill_catalog(
            self._skill_directories(),
            disabled_skills=self._disabled_skills(),
        )
        hooks = build_default_hooks(
            self.config, self._memory_store,
            skill_map=skill_map,
            task_store=self._task_store,
        )
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
                "content": build_system_message(
                    self.config,
                    skill_directories=self._skill_directories(),
                    disabled_skills=self._disabled_skills(),
                ),
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
        if self._task_store is not None:
            kwargs["tools"] = build_task_tools(self._task_store)
        return {key: value for key, value in kwargs.items() if value is not None}

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
        "messages": messages,
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
