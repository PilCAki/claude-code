from __future__ import annotations

from copilotcode_sdk.agents import build_default_custom_agents
from copilotcode_sdk.config import CopilotCodeConfig


def test_build_default_custom_agents_respects_enabled_agent_filter(tmp_path) -> None:
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        enabled_agents=("planner", "verifier"),
    )

    agents = build_default_custom_agents(config)

    assert [agent["name"] for agent in agents] == ["planner", "verifier"]
    assert agents[0]["tools"] == ["read", "search", "execute"]
    assert agents[1]["infer"] is False


def test_build_default_custom_agents_skips_unknown_and_appends_extra_agents(tmp_path) -> None:
    config = CopilotCodeConfig(
        working_directory=tmp_path,
        enabled_agents=("planner", "unknown"),
        extra_agents=(
            {
                "name": "custom-reviewer",
                "display_name": "Custom Reviewer",
                "description": "Custom agent",
                "tools": ["read"],
                "prompt": "Inspect and report.",
            },
        ),
    )

    agents = build_default_custom_agents(config)

    assert [agent["name"] for agent in agents] == ["planner", "custom-reviewer"]
    assert "CopilotCode" in agents[0]["prompt"]


def test_researcher_prompt_includes_read_only_rules(tmp_path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path)
    agents = build_default_custom_agents(config)
    researcher = next(a for a in agents if a["name"] == "researcher")

    assert "read-only" in researcher["prompt"].lower()
    assert "file paths" in researcher["prompt"].lower()
    assert "Output contract" in researcher["prompt"]
    assert "Do not modify" in researcher["prompt"]


def test_implementer_prompt_includes_skill_awareness(tmp_path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path)
    agents = build_default_custom_agents(config)
    implementer = next(a for a in agents if a["name"] == "implementer")

    assert "skill" in implementer["prompt"].lower()
    assert "verify" in implementer["prompt"].lower()
    assert "Output contract" in implementer["prompt"]


def test_verifier_prompt_includes_adversarial_posture(tmp_path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path)
    agents = build_default_custom_agents(config)
    verifier = next(a for a in agents if a["name"] == "verifier")

    assert "wrong until proven" in verifier["prompt"].lower() or "adversarial" in verifier["prompt"].lower()
    assert "PASS" in verifier["prompt"]
    assert "FAIL" in verifier["prompt"]
    assert "Do not edit" in verifier["prompt"]


def test_planner_prompt_includes_risk_assessment(tmp_path) -> None:
    config = CopilotCodeConfig(working_directory=tmp_path)
    agents = build_default_custom_agents(config)
    planner = next(a for a in agents if a["name"] == "planner")

    assert "risk" in planner["prompt"].lower()
    assert "Output contract" in planner["prompt"]
    assert "Do not modify" in planner["prompt"]
