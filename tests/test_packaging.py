from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tarfile
import venv
import zipfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.packaging


def test_build_artifacts_include_assets_and_console_entrypoint(tmp_path) -> None:
    if importlib.util.find_spec("build") is None:
        pytest.skip("python -m build is not installed")

    dist_dir = tmp_path / "dist"
    result = subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--wheel", "--outdir", str(dist_dir)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr

    wheel_path = next(dist_dir.glob("copilotcode_sdk-*.whl"))
    sdist_path = next(dist_dir.glob("copilotcode_sdk-*.tar.gz"))

    with zipfile.ZipFile(wheel_path) as wheel:
        names = wheel.namelist()
        assert any(name.endswith("copilotcode_sdk/templates/CLAUDE.md") for name in names)
        assert any(name.endswith("copilotcode_sdk/skills/verify/SKILL.md") for name in names)
        entrypoints = next(name for name in names if name.endswith("entry_points.txt"))
        assert "copilotcode =" in wheel.read(entrypoints).decode("utf-8")

    with tarfile.open(sdist_path, "r:gz") as sdist:
        names = sdist.getnames()
        assert any(name.endswith("README.copilotcode.md") for name in names)
        assert any(name.endswith("copilotcode_sdk/cli.py") for name in names)

    venv_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    venv_python = venv_dir / ("Scripts" if sys.platform == "win32" else "bin") / "python"
    install_result = subprocess.run(
        [str(venv_python), "-m", "pip", "install", "--no-deps", str(wheel_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert install_result.returncode == 0, install_result.stderr

    module_help = subprocess.run(
        [str(venv_python), "-m", "copilotcode_sdk", "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert module_help.returncode == 0, module_help.stderr
    assert "copilotcode" in module_help.stdout.lower()

    script_name = "copilotcode.exe" if sys.platform == "win32" else "copilotcode"
    script_path = venv_dir / ("Scripts" if sys.platform == "win32" else "bin") / script_name
    assert script_path.exists()
