from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


CheckStatus = Literal["ok", "warning", "error"]


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    status: CheckStatus
    message: str
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.status != "error"


@dataclass(frozen=True, slots=True)
class PreflightReport:
    product_name: str
    require_auth: bool
    working_directory: str
    app_config_directory: str
    memory_directory: str
    copilot_config_directory: str | None
    cli_path: str | None
    checks: tuple[CheckResult, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "product_name": self.product_name,
            "require_auth": self.require_auth,
            "working_directory": self.working_directory,
            "app_config_directory": self.app_config_directory,
            "memory_directory": self.memory_directory,
            "copilot_config_directory": self.copilot_config_directory,
            "cli_path": self.cli_path,
            "ok": self.ok,
            "checks": [asdict(check) for check in self.checks],
        }

    def to_text(self) -> str:
        lines = [f"{self.product_name} preflight", f"ok: {self.ok}"]
        lines.append(f"require_auth: {self.require_auth}")
        lines.append(f"working_directory: {self.working_directory}")
        lines.append(f"app_config_directory: {self.app_config_directory}")
        lines.append(f"memory_directory: {self.memory_directory}")
        if self.copilot_config_directory:
            lines.append(f"copilot_config_directory: {self.copilot_config_directory}")
        if self.cli_path:
            lines.append(f"cli_path: {self.cli_path}")
        lines.append("checks:")
        for check in self.checks:
            lines.append(f"- [{check.status}] {check.name}: {check.message}")
            if check.detail:
                lines.append(f"  {check.detail}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class SmokeTestReport:
    product_name: str
    live: bool
    success: bool
    preflight: PreflightReport
    session_created: bool
    prompt_roundtrip: bool
    session_id: str | None = None
    workspace_path: str | None = None
    prompt: str | None = None
    detail: str | None = None
    error: str | None = None
    report_path: str | None = None
    transcript_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "product_name": self.product_name,
            "live": self.live,
            "success": self.success,
            "session_created": self.session_created,
            "prompt_roundtrip": self.prompt_roundtrip,
            "session_id": self.session_id,
            "workspace_path": self.workspace_path,
            "prompt": self.prompt,
            "detail": self.detail,
            "error": self.error,
            "report_path": self.report_path,
            "transcript_path": self.transcript_path,
            "preflight": self.preflight.to_dict(),
        }

    def to_text(self) -> str:
        lines = [
            f"{self.product_name} smoke test",
            f"live: {self.live}",
            f"success: {self.success}",
            f"session_created: {self.session_created}",
            f"prompt_roundtrip: {self.prompt_roundtrip}",
        ]
        if self.session_id:
            lines.append(f"session_id: {self.session_id}")
        if self.workspace_path:
            lines.append(f"workspace_path: {self.workspace_path}")
        if self.detail:
            lines.append(f"detail: {self.detail}")
        if self.error:
            lines.append(f"error: {self.error}")
        if self.report_path:
            lines.append(f"report_path: {self.report_path}")
        if self.transcript_path:
            lines.append(f"transcript_path: {self.transcript_path}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ValidationPhaseReport:
    name: str
    success: bool
    command: tuple[str, ...] = field(default_factory=tuple)
    returncode: int | None = None
    detail: str | None = None
    artifact_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "success": self.success,
            "command": list(self.command),
            "returncode": self.returncode,
            "detail": self.detail,
            "artifact_path": self.artifact_path,
        }


@dataclass(frozen=True, slots=True)
class ValidationReport:
    product_name: str
    repo_root: str
    phases: tuple[ValidationPhaseReport, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return all(phase.success for phase in self.phases)

    def to_dict(self) -> dict[str, object]:
        return {
            "product_name": self.product_name,
            "repo_root": self.repo_root,
            "ok": self.ok,
            "phases": [phase.to_dict() for phase in self.phases],
        }

    def to_text(self) -> str:
        lines = [
            f"{self.product_name} validate",
            f"repo_root: {self.repo_root}",
            f"ok: {self.ok}",
            "phases:",
        ]
        for phase in self.phases:
            lines.append(f"- [{ 'ok' if phase.success else 'error' }] {phase.name}")
            if phase.detail:
                lines.append(f"  {phase.detail}")
            if phase.artifact_path:
                lines.append(f"  artifact_path: {phase.artifact_path}")
        return "\n".join(lines)
