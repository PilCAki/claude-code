from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence

from .brand import DEFAULT_BRAND
from .client import CopilotCodeClient
from .config import CopilotCodeConfig
from .memory import MemoryStore
from .prompt_compiler import materialize_workspace_instructions
from .exercise import run_exercise
from .reports import ValidationPhaseReport, ValidationReport
from .tasks import TaskStore

PYTEST_TIMEOUT_SECONDS = 900


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=DEFAULT_BRAND.cli_name,
        description=(
            f"{DEFAULT_BRAND.public_name} wraps the GitHub Copilot Python SDK with "
            "safer defaults, durable memory, and portable Claude Code-inspired workflows."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser(
        "preflight",
        help="Check whether the local environment is ready for CopilotCode.",
    )
    _add_common_config_arguments(preflight_parser)
    preflight_parser.add_argument(
        "--require-auth",
        action="store_true",
        help="Fail the check if Copilot auth does not appear to be configured.",
    )
    preflight_parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    preflight_parser.set_defaults(func=_run_preflight)

    init_parser = subparsers.add_parser(
        "init",
        help="Write CLAUDE.md and .github/copilot-instructions.md into a workspace.",
    )
    _add_common_config_arguments(init_parser)
    init_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing instruction files if they already exist.",
    )
    init_parser.set_defaults(func=_run_init)

    smoke_parser = subparsers.add_parser(
        "smoke",
        help="Run a dry-run or live smoke test against the Copilot runtime.",
    )
    _add_common_config_arguments(smoke_parser)
    smoke_parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    smoke_parser.add_argument(
        "--live",
        action="store_true",
        help="Attempt a real Copilot session instead of running preflight only.",
    )
    smoke_parser.add_argument(
        "--prompt",
        default="Reply with the single word OK.",
        help="Prompt to use for a live smoke test.",
    )
    smoke_parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for the live smoke prompt to complete.",
    )
    smoke_parser.add_argument(
        "--save-report",
        dest="save_report_path",
        default=None,
        help="Optional JSON file path for the smoke report.",
    )
    smoke_parser.add_argument(
        "--save-transcript",
        dest="save_transcript_dir",
        default=None,
        help="Optional directory for saved live smoke transcripts.",
    )
    smoke_parser.set_defaults(func=_run_smoke)

    exercise_parser = subparsers.add_parser(
        "exercise",
        help="Exercise every SDK subsystem with a real LLM session and report results.",
    )
    _add_common_config_arguments(exercise_parser)
    exercise_parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    exercise_parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Seconds to wait for the exercise session to complete.",
    )
    exercise_parser.add_argument(
        "--save-report",
        dest="save_report_path",
        default=None,
        help="Optional JSON file path for the exercise report.",
    )
    exercise_parser.add_argument(
        "--subsystems",
        nargs="*",
        default=None,
        help="Optional list of specific subsystem names to exercise.",
    )
    exercise_parser.add_argument(
        "--mode",
        choices=("subsystem", "orchestration", "advanced", "cascade", "micro", "chain", "full"),
        default="full",
        help="Exercise mode: subsystem (API checks), orchestration (basic wiring), advanced (reactive behaviors), cascade (multi-subsystem interactions), micro (LLM single-prompt), chain (LLM multi-step), or full (all).",
    )
    exercise_parser.set_defaults(func=_run_exercise)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Run the local confidence checks for this CopilotCode repo checkout.",
    )
    _add_common_config_arguments(validate_parser)
    validate_parser.add_argument(
        "--include-packaging",
        action="store_true",
        help="Also run the packaging build/install validation phase.",
    )
    validate_parser.add_argument(
        "--include-live",
        action="store_true",
        help="Also run the manual authenticated live Copilot validation phase.",
    )
    validate_parser.add_argument(
        "--coverage",
        action="store_true",
        help="Also enforce 90% line coverage for copilotcode_sdk during deterministic validation.",
    )
    validate_parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    validate_parser.set_defaults(func=_run_validate)

    memory_parser = subparsers.add_parser(
        "memory",
        help="Inspect or maintain CopilotCode durable memory.",
    )
    memory_subparsers = memory_parser.add_subparsers(dest="memory_command", required=True)

    memory_list_parser = memory_subparsers.add_parser(
        "list",
        help="List durable memory records for the current workspace.",
    )
    _add_common_config_arguments(memory_list_parser)
    memory_list_parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    memory_list_parser.set_defaults(func=_run_memory_list)

    memory_reindex_parser = memory_subparsers.add_parser(
        "reindex",
        help="Regenerate MEMORY.md from the current durable memory files.",
    )
    _add_common_config_arguments(memory_reindex_parser)
    memory_reindex_parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    memory_reindex_parser.set_defaults(func=_run_memory_reindex)

    # -- tasks subcommand --
    tasks_parser = subparsers.add_parser(
        "tasks",
        help="Inspect or manage CopilotCode task tracking.",
    )
    tasks_subparsers = tasks_parser.add_subparsers(dest="tasks_command", required=True)

    tasks_list_parser = tasks_subparsers.add_parser(
        "list",
        help="List all open tasks for the current workspace.",
    )
    _add_common_config_arguments(tasks_list_parser)
    tasks_list_parser.add_argument("--all", action="store_true", help="Include completed and deleted tasks.")
    tasks_list_parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    tasks_list_parser.set_defaults(func=_run_tasks_list)

    tasks_get_parser = tasks_subparsers.add_parser(
        "get",
        help="Get details for a specific task by ID.",
    )
    _add_common_config_arguments(tasks_get_parser)
    tasks_get_parser.add_argument("task_id", type=int, help="The task ID to retrieve.")
    tasks_get_parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    tasks_get_parser.set_defaults(func=_run_tasks_get)

    tasks_clear_parser = tasks_subparsers.add_parser(
        "clear",
        help="Delete the task store file for the current workspace.",
    )
    _add_common_config_arguments(tasks_clear_parser)
    tasks_clear_parser.set_defaults(func=_run_tasks_clear)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


def _add_common_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workdir",
        default=".",
        help="Workspace directory to operate on. Defaults to the current directory.",
    )
    parser.add_argument(
        "--memory-root",
        default=None,
        help="Override the durable memory root. Defaults to ~/.copilotcode.",
    )
    parser.add_argument(
        "--copilot-config-dir",
        default=None,
        help="Override the Copilot CLI config directory. Defaults to ~/.copilot.",
    )
    parser.add_argument(
        "--cli-path",
        default=None,
        help="Path to the Copilot CLI executable if it is not on PATH.",
    )
    parser.add_argument(
        "--github-token",
        default=None,
        help="Optional GitHub token to use instead of the logged-in Copilot CLI user.",
    )
    parser.add_argument(
        "--permission-policy",
        choices=("safe", "approve_all"),
        default="safe",
        help="Permission policy for live session creation. Defaults to safe.",
    )
    parser.add_argument(
        "--approved-shell-prefix",
        dest="approved_shell_prefixes",
        action="append",
        default=None,
        help=(
            "Add an approved shell-command prefix for the safe permission policy. "
            "Repeat this flag to allow multiple prefixes."
        ),
    )


def _config_from_args(args: argparse.Namespace) -> CopilotCodeConfig:
    kwargs = {
        "working_directory": args.workdir,
        "memory_root": args.memory_root,
        "copilot_config_dir": args.copilot_config_dir,
        "cli_path": args.cli_path,
        "github_token": args.github_token,
        "permission_policy": args.permission_policy,
    }
    if args.approved_shell_prefixes:
        kwargs["approved_shell_prefixes"] = tuple(args.approved_shell_prefixes)
    return CopilotCodeConfig(**kwargs)


def _run_preflight(args: argparse.Namespace) -> int:
    client = CopilotCodeClient(_config_from_args(args))
    report = client.preflight(require_auth=args.require_auth)
    _emit_report(report.to_dict(), report.to_text(), as_json=args.json)
    return 0 if report.ok else 1


def _run_init(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    try:
        claude_path, copilot_path = materialize_workspace_instructions(
            config.working_path,
            config,
            overwrite=args.overwrite,
        )
    except FileExistsError as exc:
        print(str(exc))
        return 1

    print(f"Wrote {claude_path}")
    print(f"Wrote {copilot_path}")
    return 0


def _run_smoke(args: argparse.Namespace) -> int:
    client = CopilotCodeClient(_config_from_args(args))
    report = asyncio.run(
        client.smoke_test(
            live=args.live,
            prompt=args.prompt,
            timeout=args.timeout,
            save_report_path=args.save_report_path,
            save_transcript_dir=args.save_transcript_dir,
        ),
    )
    _emit_report(report.to_dict(), report.to_text(), as_json=args.json)
    return 0 if report.success else 1


def _run_exercise(args: argparse.Namespace) -> int:
    client = CopilotCodeClient(_config_from_args(args))
    report = asyncio.run(
        run_exercise(
            client,
            timeout=args.timeout,
            mode=args.mode,
            subsystems=args.subsystems,
            save_report_path=args.save_report_path,
        ),
    )
    _emit_report(report.to_dict(), report.to_text(), as_json=args.json)
    return 0 if report.ok else 1


def _run_validate(args: argparse.Namespace) -> int:
    report = _build_validation_report(args)
    _emit_report(report.to_dict(), report.to_text(), as_json=args.json)
    return 0 if report.ok else 1


def _run_memory_list(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    store = MemoryStore(
        config.working_path,
        config.memory_home,
        brand=config.brand,
    )
    records = store.list_records()
    if args.json:
        print(
            json.dumps(
                {
                    "memory_directory": str(store.memory_dir),
                    "records": [
                        {
                            "path": str(record.path),
                            "name": record.name,
                            "description": record.description,
                            "type": record.memory_type,
                        }
                        for record in records
                    ],
                },
                indent=2,
            ),
        )
        return 0

    print(f"Memory directory: {store.memory_dir}")
    if not records:
        print("No durable memories found.")
        return 0

    for record in records:
        type_label = record.memory_type or "untyped"
        print(f"- {record.name} [{type_label}]")
        print(f"  {record.path}")
        if record.description:
            print(f"  {record.description}")
    return 0


def _run_memory_reindex(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    store = MemoryStore(
        config.working_path,
        config.memory_home,
        brand=config.brand,
    )
    index_content = store.reindex()
    payload = {
        "memory_directory": str(store.memory_dir),
        "index_path": str(store.index_path),
        "entries": [line for line in index_content.splitlines() if line.strip()],
    }
    _emit_report(payload, f"Reindexed {store.index_path}", as_json=args.json)
    return 0


def _task_store_from_args(args: argparse.Namespace) -> TaskStore:
    config = _config_from_args(args)
    task_root = config.memory_home / "tasks"
    return TaskStore(persist_path=task_root / "tasks.json")


def _run_tasks_list(args: argparse.Namespace) -> int:
    store = _task_store_from_args(args)
    tasks = store.list_all() if args.all else store.list_open()

    if args.json:
        print(json.dumps([t.to_dict() for t in tasks], indent=2))
        return 0

    if not tasks:
        print("No tasks found.")
        return 0

    for t in tasks:
        status = t.status.value.replace("_", " ")
        owner = f" (owner: {t.owner})" if t.owner else ""
        print(f"#{t.id} [{status}]{owner}: {t.subject}")
        if t.description:
            print(f"    {t.description}")
    return 0


def _run_tasks_get(args: argparse.Namespace) -> int:
    store = _task_store_from_args(args)
    task = store.get(args.task_id)
    if task is None:
        print(f"Task #{args.task_id} not found.")
        return 1

    if args.json:
        print(json.dumps(task.to_dict(), indent=2))
        return 0

    status = task.status.value.replace("_", " ")
    print(f"Task #{task.id}: {task.subject}")
    print(f"  Status:      {status}")
    if task.owner:
        print(f"  Owner:       {task.owner}")
    if task.description:
        print(f"  Description: {task.description}")
    if task.metadata:
        print(f"  Metadata:    {json.dumps(task.metadata)}")
    return 0


def _run_tasks_clear(args: argparse.Namespace) -> int:
    store = _task_store_from_args(args)
    if store.persist_path and store.persist_path.exists():
        store.persist_path.unlink()
        print(f"Deleted {store.persist_path}")
    else:
        print("No task store file found.")
    return 0


def _build_validation_report(args: argparse.Namespace) -> ValidationReport:
    repo_root = _find_validation_root(Path.cwd())
    phases: list[ValidationPhaseReport] = []

    if importlib.util.find_spec("pytest") is None:
        phases.append(
            ValidationPhaseReport(
                name="deterministic",
                success=False,
                detail=(
                    "Pytest is not installed. Install it with `pip install pytest` "
                    "or `pip install .[dev]` before running validation."
                ),
            ),
        )
        return ValidationReport(
            product_name=DEFAULT_BRAND.public_name,
            repo_root=str(repo_root),
            phases=tuple(phases),
        )

    if args.coverage and importlib.util.find_spec("pytest_cov") is None:
        phases.append(
            ValidationPhaseReport(
                name="coverage",
                success=False,
                detail=(
                    "Coverage validation requires `pytest-cov`. Install it with "
                    "`pip install pytest-cov` or `pip install .[dev]`."
                ),
            ),
        )
        return ValidationReport(
            product_name=DEFAULT_BRAND.public_name,
            repo_root=str(repo_root),
            phases=tuple(phases),
        )

    phases.append(
        _run_pytest_phase(
            "deterministic",
            repo_root=repo_root,
            marker_expression="not packaging and not live",
            with_coverage=args.coverage,
        ),
    )

    if args.include_packaging:
        if importlib.util.find_spec("build") is None:
            phases.append(
                ValidationPhaseReport(
                    name="packaging",
                    success=False,
                    detail=(
                        "Packaging validation requires `build`. Install it with "
                        "`pip install build` or `pip install .[dev]`."
                    ),
                ),
            )
        else:
            phases.append(
                _run_pytest_phase(
                    "packaging",
                    repo_root=repo_root,
                    marker_expression="packaging",
                ),
            )

    if args.include_live:
        live_env, artifact_path = _live_validation_environment(args)
        phases.append(
            _run_pytest_phase(
                "live",
                repo_root=repo_root,
                marker_expression="live",
                env=live_env,
                artifact_path=artifact_path,
            ),
        )

    return ValidationReport(
        product_name=DEFAULT_BRAND.public_name,
        repo_root=str(repo_root),
        phases=tuple(phases),
    )


def _run_pytest_phase(
    name: str,
    *,
    repo_root: Path,
    marker_expression: str,
    env: dict[str, str] | None = None,
    artifact_path: str | None = None,
    with_coverage: bool = False,
) -> ValidationPhaseReport:
    command_parts = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-m",
        marker_expression,
    ]
    if with_coverage:
        command_parts.extend(
            [
                "--cov=copilotcode_sdk",
                "--cov-report=term-missing",
                "--cov-fail-under=90",
            ],
        )
    command = tuple(command_parts)
    result = subprocess.run(
        command,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
        timeout=PYTEST_TIMEOUT_SECONDS,
        env=env,
    )
    return ValidationPhaseReport(
        name=name,
        success=result.returncode == 0,
        command=command,
        returncode=result.returncode,
        detail=_summarize_pytest_result(name, result),
        artifact_path=artifact_path,
    )


def _summarize_pytest_result(
    name: str,
    result: subprocess.CompletedProcess[str],
) -> str:
    output = "\n".join(
        line.strip()
        for line in (result.stdout or "").splitlines() + (result.stderr or "").splitlines()
        if line.strip()
    )
    if not output:
        return (
            f"{name} validation passed."
            if result.returncode == 0
            else f"{name} validation failed with exit code {result.returncode}."
        )

    lines = output.splitlines()
    tail = lines[-1]
    if result.returncode == 0:
        return tail
    return f"Exit code {result.returncode}. Last output: {tail}"


def _live_validation_environment(
    args: argparse.Namespace,
) -> tuple[dict[str, str], str | None]:
    env = dict(os.environ)
    env["COPILOTCODE_RUN_LIVE"] = "1"
    if args.github_token:
        env["COPILOTCODE_TEST_GITHUB_TOKEN"] = args.github_token
    if args.copilot_config_dir:
        env["COPILOTCODE_TEST_COPILOT_CONFIG_DIR"] = str(args.copilot_config_dir)
    if args.cli_path:
        env["COPILOTCODE_TEST_CLI_PATH"] = str(args.cli_path)
    artifact_dir = env.get("COPILOTCODE_LIVE_ARTIFACT_DIR")
    resolved_artifact_dir = (
        str(Path(artifact_dir).expanduser().resolve(strict=False))
        if artifact_dir
        else None
    )
    return env, resolved_artifact_dir


def _find_validation_root(start: Path) -> Path:
    current = start.resolve(strict=False)
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "tests").is_dir():
            return candidate
    return Path(__file__).resolve().parents[1]


def _emit_report(payload: dict[str, object], text: str, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2))
        return
    print(text)
