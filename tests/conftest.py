from __future__ import annotations

import shutil
from pathlib import Path
import sys
import uuid

import pytest


@pytest.fixture
def tmp_path() -> Path:
    base = Path.cwd() / ".copilotcode_pytest_tmp"
    base.mkdir(exist_ok=True)
    path = base / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def python_executable() -> str:
    return sys.executable
