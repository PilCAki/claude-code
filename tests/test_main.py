from __future__ import annotations

import runpy

import pytest

import copilotcode_sdk.cli


def test_module_entrypoint_invokes_cli_main(monkeypatch) -> None:
    called = {"count": 0}

    def fake_main() -> int:
        called["count"] += 1
        return 0

    monkeypatch.setattr(copilotcode_sdk.cli, "main", fake_main)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("copilotcode_sdk.__main__", run_name="__main__")

    assert exc_info.value.code == 0
    assert called["count"] == 1
