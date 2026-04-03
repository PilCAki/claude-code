from __future__ import annotations

from pathlib import Path

import pytest

from copilotcode_sdk import CopilotCodeConfig, DEFAULT_BRAND


def test_default_config_uses_copilotcode_paths(tmp_path: Path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path)

    assert config.permission_policy == "safe"
    assert config.client_name == DEFAULT_BRAND.package_name
    assert ".copilotcode" in str(config.memory_home)
    assert ".copilotcode" in str(config.app_config_home)


def test_config_normalizes_names_paths_and_prefixes(tmp_path: Path) -> None:
    config = CopilotCodeConfig(
        working_directory=tmp_path / "." / "subdir",
        path_allowlist=(tmp_path / "shared", tmp_path / "shared"),
        enabled_skills=("verify", "verify", "remember"),
        disabled_skills=("debug", "debug"),
        enabled_agents=("planner", "planner", "verifier"),
        extra_skill_directories=(tmp_path / "skills",),
        approved_shell_prefixes=("  pytest   -q  ", "pytest -q"),
    )

    assert config.working_path == (tmp_path / "subdir").resolve()
    assert len(config.path_allowlist) == 2
    assert config.enabled_skills == ("verify", "remember")
    assert config.disabled_skills == ("debug",)
    assert config.enabled_agents == ("planner", "verifier")
    assert config.extra_skill_directories == (str((tmp_path / "skills").resolve()),)
    assert config.approved_shell_prefixes == ("pytest -q",)


def test_custom_permission_policy_requires_handler(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        CopilotCodeConfig(
            working_directory=tmp_path,
            permission_policy="custom",
        )


def test_config_rejects_invalid_thresholds(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        CopilotCodeConfig(working_directory=tmp_path, shell_timeout_ms=0)

    with pytest.raises(ValueError):
        CopilotCodeConfig(working_directory=tmp_path, noisy_tool_char_limit=127)


def test_resolved_infinite_session_config_handles_true_false_and_mapping(tmp_path: Path) -> None:
    disabled = CopilotCodeConfig(working_directory=tmp_path, infinite_sessions=False)
    enabled = CopilotCodeConfig(working_directory=tmp_path, infinite_sessions=True)
    custom = CopilotCodeConfig(
        working_directory=tmp_path,
        infinite_sessions={"buffer_exhaustion_threshold": 0.5},
    )

    assert disabled.resolved_infinite_session_config() == {"enabled": False}
    assert enabled.resolved_infinite_session_config()["enabled"] is True
    assert custom.resolved_infinite_session_config()["background_compaction_threshold"] == 0.80
    assert custom.resolved_infinite_session_config()["buffer_exhaustion_threshold"] == 0.5


def test_config_property_style_normalizes_duplicate_names(tmp_path: Path) -> None:
    hypothesis = pytest.importorskip("hypothesis")
    strategies = pytest.importorskip("hypothesis.strategies")

    @hypothesis.given(
        strategies.lists(
            strategies.sampled_from(["verify", "remember", "debug"]),
            min_size=1,
            max_size=8,
        ),
    )
    def run(values: list[str]) -> None:
        config = CopilotCodeConfig(
            working_directory=tmp_path,
            enabled_skills=tuple(values),
        )
        assert len(config.enabled_skills) <= len(values)
        assert len(config.enabled_skills) == len(dict.fromkeys(config.enabled_skills))

    run()


def test_extraction_config_defaults() -> None:
    config = CopilotCodeConfig()
    assert config.extraction_tool_call_interval == 20
    assert config.extraction_char_threshold == 50_000
    assert config.extraction_min_turn_gap == 10
