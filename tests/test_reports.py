from __future__ import annotations

from copilotcode_sdk.reports import (
    CheckResult,
    PreflightReport,
    SmokeTestReport,
    ValidationPhaseReport,
    ValidationReport,
)


def test_preflight_report_to_dict_and_text_include_auth_requirement() -> None:
    report = PreflightReport(
        product_name="CopilotCode",
        require_auth=True,
        working_directory="C:/repo",
        app_config_directory="C:/repo/.copilotcode/config",
        memory_directory="C:/repo/.copilotcode/projects/test/memory",
        copilot_config_directory="C:/Users/test/.copilot",
        cli_path="C:/bin/copilot.exe",
        checks=(
            CheckResult(name="python_sdk", status="ok", message="importable"),
            CheckResult(name="auth", status="warning", message="missing", detail="details"),
        ),
    )

    payload = report.to_dict()
    text = report.to_text()

    assert payload["require_auth"] is True
    assert payload["ok"] is True
    assert "require_auth: True" in text
    assert "[warning] auth: missing" in text


def test_smoke_report_to_dict_and_text_include_artifacts() -> None:
    preflight = PreflightReport(
        product_name="CopilotCode",
        require_auth=False,
        working_directory="C:/repo",
        app_config_directory="cfg",
        memory_directory="mem",
        copilot_config_directory="copilot",
        cli_path=None,
        checks=(CheckResult(name="python_sdk", status="ok", message="ok"),),
    )
    report = SmokeTestReport(
        product_name="CopilotCode",
        live=True,
        success=True,
        preflight=preflight,
        session_created=True,
        prompt_roundtrip=True,
        session_id="session-123",
        workspace_path="workspace",
        prompt="Reply OK",
        detail="detail",
        error=None,
        report_path="C:/repo/report.json",
        transcript_path="C:/repo/transcript.json",
    )

    payload = report.to_dict()
    text = report.to_text()

    assert payload["session_id"] == "session-123"
    assert payload["report_path"] == "C:/repo/report.json"
    assert "session_id: session-123" in text
    assert "transcript_path: C:/repo/transcript.json" in text


def test_validation_report_to_dict_and_text_reflect_phase_outcomes() -> None:
    report = ValidationReport(
        product_name="CopilotCode",
        repo_root="C:/repo",
        phases=(
            ValidationPhaseReport(
                name="deterministic",
                success=True,
                command=("py", "-m", "pytest"),
                returncode=0,
                detail="44 passed",
            ),
            ValidationPhaseReport(
                name="coverage",
                success=False,
                detail="pytest-cov missing",
                artifact_path="C:/repo/artifacts",
            ),
        ),
    )

    payload = report.to_dict()
    text = report.to_text()

    assert payload["ok"] is False
    assert payload["phases"][1]["artifact_path"] == "C:/repo/artifacts"
    assert "[ok] deterministic" in text
    assert "[error] coverage" in text
