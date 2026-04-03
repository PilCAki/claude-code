from __future__ import annotations

import json
from pathlib import Path
import subprocess

from copilotcode_sdk import cli


def test_validate_runs_deterministic_phase_by_default(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def fake_run(command, **kwargs):
        calls.append((tuple(command), kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="12 passed in 0.50s\n",
            stderr="",
        )

    monkeypatch.setattr(cli, "_find_validation_root", lambda start: tmp_path)
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    exit_code = cli.main(["validate", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["ok"] is True
    assert len(calls) == 1
    assert calls[0][0][-2:] == ("-m", "not packaging and not live")


def test_validate_includes_packaging_and_live_with_expected_env(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def fake_run(command, **kwargs):
        calls.append((tuple(command), kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="1 passed in 0.25s\n",
            stderr="",
        )

    monkeypatch.setattr(cli, "_find_validation_root", lambda start: tmp_path)
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.setenv("COPILOTCODE_LIVE_ARTIFACT_DIR", str(tmp_path / "live-artifacts"))

    exit_code = cli.main(
        [
            "validate",
            "--include-packaging",
            "--include-live",
            "--cli-path",
            "copilot",
            "--github-token",
            "token-123",
            "--copilot-config-dir",
            str(tmp_path / ".copilot"),
            "--json",
        ],
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert len(calls) == 3
    assert calls[1][0][-2:] == ("-m", "packaging")
    assert calls[2][0][-2:] == ("-m", "live")
    live_env = calls[2][1]["env"]
    assert live_env["COPILOTCODE_RUN_LIVE"] == "1"
    assert live_env["COPILOTCODE_TEST_GITHUB_TOKEN"] == "token-123"
    assert live_env["COPILOTCODE_TEST_COPILOT_CONFIG_DIR"] == str(tmp_path / ".copilot")
    assert live_env["COPILOTCODE_TEST_CLI_PATH"] == "copilot"
    live_phase = next(phase for phase in payload["phases"] if phase["name"] == "live")
    assert live_phase["artifact_path"] == str((tmp_path / "live-artifacts").resolve())


def test_validate_with_coverage_adds_cov_flags(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def fake_run(command, **kwargs):
        calls.append((tuple(command), kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="12 passed in 0.50s\n",
            stderr="",
        )

    monkeypatch.setattr(cli, "_find_validation_root", lambda start: tmp_path)
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    exit_code = cli.main(["validate", "--coverage", "--json"])
    payload = json.loads(capsys.readouterr().out)
    command = calls[0][0]

    assert exit_code == 0
    assert payload["ok"] is True
    assert "--cov=copilotcode_sdk" in command
    assert "--cov-fail-under=90" in command


def test_validate_coverage_reports_missing_pytest_cov(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    def fake_find_spec(name: str):
        if name == "pytest":
            return object()
        if name == "pytest_cov":
            return None
        return object()

    monkeypatch.setattr(cli, "_find_validation_root", lambda start: tmp_path)
    monkeypatch.setattr(cli.importlib.util, "find_spec", fake_find_spec)

    exit_code = cli.main(["validate", "--coverage", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["phases"][0]["name"] == "coverage"
    assert "pytest-cov" in payload["phases"][0]["detail"]


def test_validate_packaging_reports_missing_build_dependency(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def fake_run(command, **kwargs):
        calls.append((tuple(command), kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="10 passed in 0.40s\n",
            stderr="",
        )

    def fake_find_spec(name: str):
        if name == "pytest":
            return object()
        if name == "build":
            return None
        return object()

    monkeypatch.setattr(cli, "_find_validation_root", lambda start: tmp_path)
    monkeypatch.setattr(cli.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    exit_code = cli.main(["validate", "--include-packaging", "--json"])
    payload = json.loads(capsys.readouterr().out)
    packaging_phase = next(phase for phase in payload["phases"] if phase["name"] == "packaging")

    assert exit_code == 1
    assert len(calls) == 1
    assert payload["ok"] is False
    assert packaging_phase["success"] is False
    assert "pip install build" in packaging_phase["detail"]


def test_validate_reports_missing_pytest_dependency(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    monkeypatch.setattr(cli, "_find_validation_root", lambda start: tmp_path)
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: None)

    exit_code = cli.main(["validate", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["phases"][0]["name"] == "deterministic"
    assert "Pytest is not installed" in payload["phases"][0]["detail"]
