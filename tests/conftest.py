from __future__ import annotations

import os
import shutil
from pathlib import Path
import sys
import uuid

import pytest

# Provide a default model for tests so CopilotCodeConfig() doesn't require
# an explicit model= in every test.  Production code still requires it.
os.environ.setdefault("COPILOTCODE_TEST_DEFAULT_MODEL", "claude-sonnet-4.6")


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
