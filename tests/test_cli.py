from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from copilotcode_sdk.memory import MemoryStore


def _run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "copilotcode_sdk", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_cli_preflight_json_reports_ready_state(
    tmp_path,
    python_executable: str,
) -> None:
    result = _run_cli(
        "preflight",
        "--workdir",
        str(tmp_path),
        "--memory-root",
        str(tmp_path / ".mem"),
        "--cli-path",
        python_executable,
        "--github-token",
        "test-token",
        "--json",
        cwd=Path.cwd(),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["product_name"] == "CopilotCode"
    assert payload["ok"] is True


def test_cli_preflight_require_auth_surfaces_auth_error(
    tmp_path,
    python_executable: str,
) -> None:
    result = _run_cli(
        "preflight",
        "--workdir",
        str(tmp_path),
        "--memory-root",
        str(tmp_path / ".mem"),
        "--cli-path",
        python_executable,
        "--copilot-config-dir",
        str(tmp_path / ".missing-copilot-config"),
        "--require-auth",
        "--json",
        cwd=Path.cwd(),
    )

    assert result.returncode == 1, result.stderr
    payload = json.loads(result.stdout)
    auth_check = next(check for check in payload["checks"] if check["name"] == "auth")
    assert payload["require_auth"] is True
    assert auth_check["status"] == "error"


def test_cli_init_writes_workspace_instruction_files(tmp_path) -> None:
    result = _run_cli(
        "init",
        "--workdir",
        str(tmp_path),
        "--memory-root",
        str(tmp_path / ".mem"),
        cwd=Path.cwd(),
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "CLAUDE.md").exists()
    assert (tmp_path / ".github" / "copilot-instructions.md").exists()


def test_cli_init_refuses_to_overwrite_without_flag(tmp_path) -> None:
    first = _run_cli(
        "init",
        "--workdir",
        str(tmp_path),
        "--memory-root",
        str(tmp_path / ".mem"),
        cwd=Path.cwd(),
    )
    second = _run_cli(
        "init",
        "--workdir",
        str(tmp_path),
        "--memory-root",
        str(tmp_path / ".mem"),
        cwd=Path.cwd(),
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 1
    assert "already exist" in second.stdout


def test_cli_memory_list_and_reindex(tmp_path) -> None:
    memory_root = tmp_path / ".mem"
    store = MemoryStore(tmp_path, memory_root)
    store.upsert_memory(
        title="Repo Convention",
        description="Run pytest -q before finishing.",
        memory_type="project",
        content="Run pytest -q before you wrap up.",
    )

    list_result = _run_cli(
        "memory",
        "list",
        "--workdir",
        str(tmp_path),
        "--memory-root",
        str(memory_root),
        "--json",
        cwd=Path.cwd(),
    )
    reindex_result = _run_cli(
        "memory",
        "reindex",
        "--workdir",
        str(tmp_path),
        "--memory-root",
        str(memory_root),
        "--json",
        cwd=Path.cwd(),
    )

    assert list_result.returncode == 0, list_result.stderr
    assert reindex_result.returncode == 0, reindex_result.stderr

    list_payload = json.loads(list_result.stdout)
    reindex_payload = json.loads(reindex_result.stdout)
    assert list_payload["records"][0]["name"] == "Repo Convention"
    assert reindex_payload["entries"]


def test_cli_memory_text_output_handles_empty_store(tmp_path) -> None:
    result = _run_cli(
        "memory",
        "list",
        "--workdir",
        str(tmp_path),
        "--memory-root",
        str(tmp_path / ".mem"),
        cwd=Path.cwd(),
    )

    assert result.returncode == 0, result.stderr
    assert "No durable memories found." in result.stdout


def test_cli_memory_reindex_text_output_handles_empty_store(tmp_path) -> None:
    result = _run_cli(
        "memory",
        "reindex",
        "--workdir",
        str(tmp_path),
        "--memory-root",
        str(tmp_path / ".mem"),
        cwd=Path.cwd(),
    )

    assert result.returncode == 0, result.stderr
    assert "Reindexed" in result.stdout


def test_cli_smoke_dry_run_json(
    tmp_path,
    python_executable: str,
) -> None:
    result = _run_cli(
        "smoke",
        "--workdir",
        str(tmp_path),
        "--memory-root",
        str(tmp_path / ".mem"),
        "--cli-path",
        python_executable,
        "--github-token",
        "test-token",
        "--json",
        cwd=Path.cwd(),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["success"] is True
    assert payload["live"] is False


def test_cli_smoke_dry_run_can_save_report(
    tmp_path,
    python_executable: str,
) -> None:
    report_path = tmp_path / "smoke-report.json"
    result = _run_cli(
        "smoke",
        "--workdir",
        str(tmp_path),
        "--memory-root",
        str(tmp_path / ".mem"),
        "--cli-path",
        python_executable,
        "--github-token",
        "test-token",
        "--save-report",
        str(report_path),
        "--json",
        cwd=Path.cwd(),
    )

    assert result.returncode == 0, result.stderr
    assert report_path.exists()
    payload = json.loads(result.stdout)
    assert payload["report_path"] == str(report_path.resolve())


def test_cli_preflight_text_mode_outputs_summary(
    tmp_path,
    python_executable: str,
) -> None:
    result = _run_cli(
        "preflight",
        "--workdir",
        str(tmp_path),
        "--memory-root",
        str(tmp_path / ".mem"),
        "--cli-path",
        python_executable,
        "--github-token",
        "test-token",
        cwd=Path.cwd(),
    )

    assert result.returncode == 0, result.stderr
    assert "CopilotCode preflight" in result.stdout
