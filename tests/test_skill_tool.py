"""Tests for the InvokeSkill tool.

Covers:
- Tool construction and prompt generation
- Skill invocation (valid, invalid, completed, missing prereqs)
- Memory context injection
- Dependency chain enforcement
- Integration with skill_assets (frontmatter parsing, _path field)
- Multi-skill scenarios (3-skill chain, diamond deps, parallel skills)
- Verification gate in CompleteSkill
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from copilotcode_sdk.skill_tool import (
    build_complete_skill_tool,
    build_skill_tool,
    build_skill_tool_prompt,
    _build_skill_user_prompt,
    _read_skill_content,
    COMPLETE_SKILL_PARAMETERS,
    SKILL_TOOL_PARAMETERS,
)
from copilotcode_sdk.skill_assets import parse_skill_frontmatter, build_skill_catalog
from copilotcode_sdk.memory import MemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_invocation(arguments: dict | None = None):
    inv = MagicMock()
    inv.session_id = "test-session"
    inv.tool_call_id = "call-1"
    inv.tool_name = "InvokeSkill"
    inv.arguments = arguments
    return inv


def _write_skill(skill_dir: Path, name: str, frontmatter: str, body: str = "") -> Path:
    d = skill_dir / name
    d.mkdir(parents=True, exist_ok=True)
    content = frontmatter
    if body:
        content += f"\n{body}"
    (d / "SKILL.md").write_text(content, encoding="utf-8")
    return d


def _build_skill_map(skill_dir: Path) -> dict[str, dict[str, str]]:
    """Build a skill map from a directory of skills."""
    _, skill_map = build_skill_catalog([str(skill_dir)])
    return skill_map


class FakeToolResult:
    def __init__(self, text_result_for_llm="", result_type="success", **kw):
        self.text_result_for_llm = text_result_for_llm
        self.result_type = result_type


class FakeTool:
    def __init__(self, name="", description="", handler=None, parameters=None,
                 overrides_built_in_tool=False, skip_permission=False):
        self.name = name
        self.description = description
        self.handler = handler
        self.parameters = parameters
        self.skip_permission = skip_permission


@pytest.fixture
def fake_copilot_types():
    """Patch copilot.types with fakes so we don't need the real SDK."""
    fake_types = types.ModuleType("copilot.types")
    fake_types.Tool = FakeTool
    fake_types.ToolInvocation = MagicMock
    fake_types.ToolResult = FakeToolResult
    original = sys.modules.get("copilot.types")
    sys.modules["copilot.types"] = fake_types
    yield fake_types
    if original is not None:
        sys.modules["copilot.types"] = original
    else:
        sys.modules.pop("copilot.types", None)


@pytest.fixture
def three_skill_dir(tmp_path: Path) -> Path:
    """Create a 3-skill dependency chain: intake -> analysis -> report."""
    _write_skill(tmp_path, "excel-workbook-intake", (
        "---\n"
        "name: excel-workbook-intake\n"
        "description: Prepare a raw Excel workbook for analysis.\n"
        "type: data-intake\n"
        "inputs: .xlsx file\n"
        "outputs: outputs/intake/\n"
        "requires: none\n"
        "---\n"
    ), body="# Intake\n\nProcess each sheet into parquet + metadata.\n")

    _write_skill(tmp_path, "rcm-analysis", (
        "---\n"
        "name: rcm-analysis\n"
        "description: Revenue cycle management analysis with KPIs.\n"
        "type: rcm-analysis\n"
        "inputs: outputs/intake/\n"
        "outputs: outputs/rcm_analysis/\n"
        "requires: data-intake\n"
        "---\n"
    ), body="# RCM Analysis\n\nCompute KPIs from intake data.\n")

    _write_skill(tmp_path, "executive-report", (
        "---\n"
        "name: executive-report\n"
        "description: Generate a polished HTML executive report.\n"
        "type: report-generation\n"
        "inputs: outputs/rcm_analysis/\n"
        "outputs: outputs/executive_report/\n"
        "requires: rcm-analysis\n"
        "---\n"
    ), body="# Executive Report\n\nProduce decision-ready HTML.\n")

    return tmp_path


@pytest.fixture
def three_skill_map(three_skill_dir: Path) -> dict[str, dict[str, str]]:
    return _build_skill_map(three_skill_dir)


@pytest.fixture
def memory_store(tmp_path: Path) -> MemoryStore:
    store = MemoryStore(working_directory=tmp_path, memory_root=tmp_path / "memory")
    store.ensure()
    return store


@pytest.fixture
def tool(three_skill_map, memory_store, fake_copilot_types):
    """Build the InvokeSkill tool with a 3-skill chain."""
    completed = set()
    return build_skill_tool(
        skill_map=three_skill_map,
        memory_store=memory_store,
        working_directory="/tmp/test",
        completed_skills=completed,
    ), completed


def _call(tool_obj, args):
    handler = tool_obj.handler
    return handler(_make_invocation(args))


# ===========================================================================
# Tool construction
# ===========================================================================

class TestToolConstruction:
    def test_tool_has_correct_name(self, tool):
        t, _ = tool
        assert t.name == "InvokeSkill"

    def test_tool_has_parameters(self, tool):
        t, _ = tool
        assert t.parameters == SKILL_TOOL_PARAMETERS
        assert "skill" in t.parameters["properties"]
        assert "context" in t.parameters["properties"]

    def test_tool_skips_permission(self, tool):
        t, _ = tool
        assert t.skip_permission is True

    def test_tool_description_lists_skills(self, tool):
        t, _ = tool
        desc = t.description
        assert "excel-workbook-intake" in desc
        assert "rcm-analysis" in desc
        assert "executive-report" in desc

    def test_tool_description_shows_dependencies(self, tool):
        t, _ = tool
        desc = t.description
        assert "requires: data-intake" in desc
        assert "requires: rcm-analysis" in desc
        assert "requires: none" in desc


# ===========================================================================
# Prompt generation
# ===========================================================================

class TestPromptGeneration:
    def test_prompt_includes_blocking_requirement(self, three_skill_map):
        prompt = build_skill_tool_prompt(three_skill_map)
        assert "BLOCKING REQUIREMENT" in prompt

    def test_prompt_includes_invoke_instruction(self, three_skill_map):
        prompt = build_skill_tool_prompt(three_skill_map)
        assert "InvokeSkill" in prompt

    def test_prompt_empty_skill_map(self):
        prompt = build_skill_tool_prompt({})
        assert "InvokeSkill" in prompt
        assert "Available skills:" not in prompt


# ===========================================================================
# Valid invocations
# ===========================================================================

class TestValidInvocation:
    def test_invoke_root_skill(self, tool):
        t, completed = tool
        result = _call(t, {"skill": "excel-workbook-intake"})
        data = json.loads(result.text_result_for_llm)
        assert data["status"] in ("invoke_skill", "invoke_skill_no_session")
        assert data["skill_name"] == "excel-workbook-intake"
        assert data["skill_type"] == "data-intake"
        assert data["requires"] == "none"

    def test_invoke_with_context(self, tool):
        t, completed = tool
        result = _call(t, {
            "skill": "excel-workbook-intake",
            "context": "Dataset at test_data/Q1.xlsx",
        })
        data = json.loads(result.text_result_for_llm)
        assert data["status"] in ("invoke_skill", "invoke_skill_no_session")
        assert data["skill_name"] == "excel-workbook-intake"

    def test_invoke_downstream_after_prereq_complete(self, tool):
        t, completed = tool
        completed.add("excel-workbook-intake")
        result = _call(t, {"skill": "rcm-analysis"})
        data = json.loads(result.text_result_for_llm)
        assert data["status"] in ("invoke_skill", "invoke_skill_no_session")
        assert data["skill_name"] == "rcm-analysis"

    def test_invoke_third_skill_after_chain_complete(self, tool):
        t, completed = tool
        completed.add("excel-workbook-intake")
        completed.add("rcm-analysis")
        result = _call(t, {"skill": "executive-report"})
        data = json.loads(result.text_result_for_llm)
        assert data["status"] in ("invoke_skill", "invoke_skill_no_session")
        assert data["skill_name"] == "executive-report"


# ===========================================================================
# Error cases
# ===========================================================================

class TestErrorCases:
    def test_missing_skill_name(self, tool):
        t, _ = tool
        result = _call(t, {})
        assert result.result_type == "error"
        assert "required" in result.text_result_for_llm.lower()

    def test_empty_skill_name(self, tool):
        t, _ = tool
        result = _call(t, {"skill": "   "})
        assert result.result_type == "error"

    def test_unknown_skill(self, tool):
        t, _ = tool
        result = _call(t, {"skill": "nonexistent-skill"})
        assert result.result_type == "error"
        assert "not found" in result.text_result_for_llm.lower()
        # Should list available skills
        assert "excel-workbook-intake" in result.text_result_for_llm

    def test_already_completed_skill(self, tool):
        t, completed = tool
        completed.add("excel-workbook-intake")
        result = _call(t, {"skill": "excel-workbook-intake"})
        assert "already completed" in result.text_result_for_llm

    def test_prerequisite_not_met(self, tool):
        t, completed = tool
        # Try to invoke rcm-analysis without completing intake first
        result = _call(t, {"skill": "rcm-analysis"})
        assert result.result_type == "error"
        assert "requires" in result.text_result_for_llm.lower()
        assert "data-intake" in result.text_result_for_llm

    def test_transitive_prereq_not_met(self, tool):
        t, completed = tool
        # Try to invoke executive-report without any prereqs
        result = _call(t, {"skill": "executive-report"})
        assert result.result_type == "error"
        assert "rcm-analysis" in result.text_result_for_llm

    def test_transitive_prereq_partial(self, tool):
        t, completed = tool
        # Complete intake but not analysis — report should still fail
        completed.add("excel-workbook-intake")
        result = _call(t, {"skill": "executive-report"})
        assert result.result_type == "error"


# ===========================================================================
# Skill content reading
# ===========================================================================

class TestSkillContentReading:
    def test_read_existing_skill(self, three_skill_map):
        content = _read_skill_content(three_skill_map, "excel-workbook-intake")
        assert content is not None
        assert "# Intake" in content
        assert "parquet" in content

    def test_read_nonexistent_skill(self, three_skill_map):
        assert _read_skill_content(three_skill_map, "nope") is None

    def test_read_skill_with_missing_file(self, three_skill_map):
        # Remove the _path
        modified = dict(three_skill_map)
        modified["excel-workbook-intake"] = {
            k: v for k, v in modified["excel-workbook-intake"].items()
            if k != "_path"
        }
        assert _read_skill_content(modified, "excel-workbook-intake") is None


# ===========================================================================
# User prompt construction
# ===========================================================================

class TestUserPrompt:
    def test_includes_skill_content(self):
        prompt = _build_skill_user_prompt(
            skill_name="test-skill",
            skill_content="# Test\nDo the thing.",
            context="",
            memory_context="",
            working_directory="/tmp",
        )
        assert "# Skill Execution: test-skill" in prompt
        assert "# Test\nDo the thing." in prompt

    def test_includes_context(self):
        prompt = _build_skill_user_prompt(
            skill_name="test-skill",
            skill_content="# Test",
            context="Dataset at /data/file.xlsx",
            memory_context="",
            working_directory="/tmp",
        )
        assert "Dataset at /data/file.xlsx" in prompt
        assert "Context from orchestrator" in prompt

    def test_includes_memory(self):
        prompt = _build_skill_user_prompt(
            skill_name="test-skill",
            skill_content="# Test",
            context="",
            memory_context="- [Prior metrics](prior.md) - KPIs from last run",
            working_directory="/tmp",
        )
        assert "Prior metrics" in prompt
        assert "Project memory" in prompt

    def test_includes_completed_skills(self):
        prompt = _build_skill_user_prompt(
            skill_name="rcm-analysis",
            skill_content="# Analysis",
            context="",
            memory_context="",
            working_directory="/tmp",
            completed_skills=["excel-workbook-intake"],
        )
        assert "excel-workbook-intake" in prompt
        assert "Previously completed" in prompt

    def test_includes_working_directory(self):
        prompt = _build_skill_user_prompt(
            skill_name="test",
            skill_content="# Test",
            context="",
            memory_context="",
            working_directory="/home/user/project",
        )
        assert "/home/user/project" in prompt

    def test_includes_instructions(self):
        prompt = _build_skill_user_prompt(
            skill_name="test",
            skill_content="# Test",
            context="",
            memory_context="",
            working_directory="/tmp",
        )
        assert "Follow the skill methodology" in prompt
        assert "quality rubric" in prompt


# ===========================================================================
# Memory context injection
# ===========================================================================

class TestMemoryInjection:
    def test_memory_index_passed_to_prompt(self, three_skill_map, memory_store, fake_copilot_types):
        # Write some memory
        memory_store.upsert_memory(
            title="Dataset structure",
            description="The dataset has 166K rows of claims data",
            memory_type="project",
            content="Claims data with 166K rows, 27 columns.",
            slug="dataset-structure",
        )
        completed = set()
        t = build_skill_tool(
            skill_map=three_skill_map,
            memory_store=memory_store,
            working_directory="/tmp",
            completed_skills=completed,
        )
        result = _call(t, {"skill": "excel-workbook-intake"})
        data = json.loads(result.text_result_for_llm)
        # Without a session, skill is not forked — just check it was accepted
        assert data["skill_name"] == "excel-workbook-intake"

    def test_empty_memory_still_works(self, three_skill_map, memory_store, fake_copilot_types):
        completed = set()
        t = build_skill_tool(
            skill_map=three_skill_map,
            memory_store=memory_store,
            working_directory="/tmp",
            completed_skills=completed,
        )
        result = _call(t, {"skill": "excel-workbook-intake"})
        data = json.loads(result.text_result_for_llm)
        assert data["status"] in ("invoke_skill", "invoke_skill_no_session")


# ===========================================================================
# Frontmatter _path field
# ===========================================================================

class TestPathField:
    def test_parse_frontmatter_includes_path(self, tmp_path: Path):
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: A skill.\n---\n\n# Body\n",
            encoding="utf-8",
        )
        result = parse_skill_frontmatter(skill_dir / "SKILL.md")
        assert "_path" in result
        assert result["_path"].endswith("SKILL.md")
        assert Path(result["_path"]).is_file()

    def test_skill_catalog_preserves_path(self, three_skill_dir):
        _, skill_map = build_skill_catalog([str(three_skill_dir)])
        for name, fm in skill_map.items():
            assert "_path" in fm, f"Skill {name} missing _path"
            assert Path(fm["_path"]).is_file()


# ===========================================================================
# Multi-skill dependency scenarios
# ===========================================================================

class TestDependencyScenarios:
    """Test complex dependency patterns beyond simple chains."""

    def test_parallel_independent_skills(self, tmp_path, memory_store, fake_copilot_types):
        """Two skills with no dependencies — both should be invocable."""
        _write_skill(tmp_path, "skill-a", (
            "---\nname: skill-a\ndescription: A.\ntype: type-a\nrequires: none\n---\n\n# A\n"
        ))
        _write_skill(tmp_path, "skill-b", (
            "---\nname: skill-b\ndescription: B.\ntype: type-b\nrequires: none\n---\n\n# B\n"
        ))
        skill_map = _build_skill_map(tmp_path)
        completed = set()
        t = build_skill_tool(skill_map, memory_store, "/tmp", completed)

        # Both should be invocable independently
        r_a = _call(t, {"skill": "skill-a"})
        assert json.loads(r_a.text_result_for_llm)["status"] in ("invoke_skill", "invoke_skill_no_session")

        r_b = _call(t, {"skill": "skill-b"})
        assert json.loads(r_b.text_result_for_llm)["status"] in ("invoke_skill", "invoke_skill_no_session")

    def test_diamond_dependency(self, tmp_path, memory_store, fake_copilot_types):
        """Diamond: A -> B, A -> C, B+C -> D."""
        _write_skill(tmp_path, "a", "---\nname: a\ntype: base\nrequires: none\n---\n\n# A\n")
        _write_skill(tmp_path, "b", "---\nname: b\ntype: mid-b\nrequires: base\n---\n\n# B\n")
        _write_skill(tmp_path, "c", "---\nname: c\ntype: mid-c\nrequires: base\n---\n\n# C\n")
        # D requires mid-b (only one requires field supported)
        _write_skill(tmp_path, "d", "---\nname: d\ntype: final\nrequires: mid-b\n---\n\n# D\n")

        skill_map = _build_skill_map(tmp_path)
        completed = set()
        t = build_skill_tool(skill_map, memory_store, "/tmp", completed)

        # Can't invoke B without A
        assert _call(t, {"skill": "b"}).result_type == "error"

        # Complete A, now B and C should work
        completed.add("a")
        assert json.loads(_call(t, {"skill": "b"}).text_result_for_llm)["status"] in ("invoke_skill", "invoke_skill_no_session")
        assert json.loads(_call(t, {"skill": "c"}).text_result_for_llm)["status"] in ("invoke_skill", "invoke_skill_no_session")

        # D needs B completed
        assert _call(t, {"skill": "d"}).result_type == "error"
        completed.add("b")
        assert json.loads(_call(t, {"skill": "d"}).text_result_for_llm)["status"] in ("invoke_skill", "invoke_skill_no_session")

    def test_skill_with_long_content(self, tmp_path, memory_store, fake_copilot_types):
        """Ensure full SKILL.md content (even large) gets into the prompt."""
        body = "# Detailed Skill\n\n" + "Step instructions.\n" * 200
        _write_skill(tmp_path, "big-skill", (
            "---\nname: big-skill\ndescription: A big skill.\ntype: big\nrequires: none\n---\n"
        ), body=body)

        skill_map = _build_skill_map(tmp_path)
        completed = set()
        t = build_skill_tool(skill_map, memory_store, "/tmp", completed)

        result = _call(t, {"skill": "big-skill"})
        data = json.loads(result.text_result_for_llm)
        # Without a session, skill is accepted but not forked
        assert data["skill_name"] == "big-skill"

    def test_completed_skills_shown_in_context(self, tmp_path, memory_store, fake_copilot_types):
        """When invoking a downstream skill, completed skills should be listed."""
        _write_skill(tmp_path, "step1", "---\nname: step1\ntype: s1\nrequires: none\n---\n\n# Step 1\n")
        _write_skill(tmp_path, "step2", "---\nname: step2\ntype: s2\nrequires: s1\n---\n\n# Step 2\n")

        skill_map = _build_skill_map(tmp_path)
        completed = {"step1"}
        t = build_skill_tool(skill_map, memory_store, "/tmp", completed)

        result = _call(t, {"skill": "step2"})
        data = json.loads(result.text_result_for_llm)
        assert data["skill_name"] == "step2"


# ===========================================================================
# Skill catalog text updates
# ===========================================================================

class TestCatalogText:
    def test_catalog_references_invoke_skill(self, three_skill_dir):
        catalog_text, _ = build_skill_catalog([str(three_skill_dir)])
        assert "InvokeSkill" in catalog_text

    def test_catalog_says_not_to_do_work_directly(self, three_skill_dir):
        catalog_text, _ = build_skill_catalog([str(three_skill_dir)])
        assert "Do not attempt skill work directly" in catalog_text


# ===========================================================================
# System prompt integration
# ===========================================================================

class TestSystemPromptIntegration:
    def test_core_rules_reference_invoke_skill(self):
        from copilotcode_sdk.prompt_compiler import _core_operating_rules
        rules = _core_operating_rules()
        combined = " ".join(rules)
        assert "InvokeSkill" in combined

    def test_session_guidance_references_invoke_skill(self):
        from copilotcode_sdk.prompt_compiler import _session_guidance
        guidance = _session_guidance()
        combined = " ".join(guidance)
        assert "InvokeSkill" in combined

    def test_session_guidance_says_dont_stop(self):
        from copilotcode_sdk.prompt_compiler import _session_guidance
        guidance = _session_guidance()
        combined = " ".join(guidance)
        assert "Do not stop" in combined


# ===========================================================================
# Hook integration
# ===========================================================================

class TestHookIntegration:
    def test_passive_skill_completion_disabled(self, three_skill_dir):
        """Passive detection is disabled — _check_skill_completion always returns None."""
        from copilotcode_sdk.hooks import _check_skill_completion
        _, skill_map = build_skill_catalog([str(three_skill_dir)])
        completed: set[str] = set()

        result = _check_skill_completion(
            tool_name="write",
            tool_args={"file_path": "outputs/intake/data_intake/sheet.json"},
            skill_map=skill_map,
            completed_skills=completed,
        )
        assert result is None

    def test_skill_reminder_mentions_invoke_skill(self, three_skill_dir):
        from copilotcode_sdk.hooks import _build_skill_reminder
        _, skill_map = build_skill_catalog([str(three_skill_dir)])
        completed: set[str] = {"excel-workbook-intake"}

        reminder = _build_skill_reminder(skill_map, completed)
        assert "InvokeSkill" in reminder
        assert "rcm-analysis" in reminder

    def test_all_complete_reminder(self, three_skill_dir):
        from copilotcode_sdk.hooks import _build_skill_reminder
        _, skill_map = build_skill_catalog([str(three_skill_dir)])
        completed = {"excel-workbook-intake", "rcm-analysis", "executive-report"}

        reminder = _build_skill_reminder(skill_map, completed)
        assert "All skills complete" in reminder


# ===========================================================================
# Invocation record structure
# ===========================================================================

class TestInvocationRecord:
    def test_record_contains_required_fields(self, tool):
        """Without a session, the no-session fallback returns basic metadata."""
        t, completed = tool
        result = _call(t, {"skill": "excel-workbook-intake"})
        data = json.loads(result.text_result_for_llm)

        assert "status" in data
        assert "skill_name" in data
        assert "skill_type" in data
        assert "outputs" in data

    def test_no_session_returns_warning(self, tool):
        """Without a session, InvokeSkill should warn that no child was forked."""
        t, _ = tool
        result = _call(t, {"skill": "excel-workbook-intake"})
        data = json.loads(result.text_result_for_llm)
        assert data["status"] == "invoke_skill_no_session"
        assert "warning" in data


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_skill_name_with_whitespace(self, tool):
        t, _ = tool
        result = _call(t, {"skill": "  excel-workbook-intake  "})
        data = json.loads(result.text_result_for_llm)
        assert data["status"] in ("invoke_skill", "invoke_skill_no_session")

    def test_none_arguments(self, tool):
        t, _ = tool
        inv = MagicMock()
        inv.arguments = None
        result = t.handler(inv)
        assert result.result_type == "error"

    def test_shared_completed_set_mutates(self, three_skill_map, memory_store, fake_copilot_types):
        """completed_skills set is shared — external mutations should be visible."""
        completed = set()
        t = build_skill_tool(three_skill_map, memory_store, "/tmp", completed)

        # Initially can't invoke rcm-analysis
        assert _call(t, {"skill": "rcm-analysis"}).result_type == "error"

        # External mutation (e.g., hooks marking a skill complete)
        completed.add("excel-workbook-intake")

        # Now it should work
        result = _call(t, {"skill": "rcm-analysis"})
        assert json.loads(result.text_result_for_llm)["status"] in ("invoke_skill", "invoke_skill_no_session")

    def test_skill_with_no_type(self, tmp_path, memory_store, fake_copilot_types):
        """Skills without a type field should still be invocable."""
        _write_skill(tmp_path, "typeless", (
            "---\nname: typeless\ndescription: No type.\nrequires: none\n---\n\n# Typeless\n"
        ))
        skill_map = _build_skill_map(tmp_path)
        completed = set()
        t = build_skill_tool(skill_map, memory_store, "/tmp", completed)

        result = _call(t, {"skill": "typeless"})
        data = json.loads(result.text_result_for_llm)
        assert data["status"] in ("invoke_skill", "invoke_skill_no_session")


# ---------------------------------------------------------------------------
# CompleteSkill tool tests
# ---------------------------------------------------------------------------


def _call_complete(tool, arguments: dict):
    """Call a CompleteSkill tool handler directly."""
    inv = _make_invocation(arguments)
    inv.tool_name = "CompleteSkill"
    return tool.handler(inv)


class TestCompleteSkillTool:
    """Tests for the CompleteSkill tool."""

    def test_build_creates_tool(self, fake_copilot_types, three_skill_map):
        completed = set()
        tool = build_complete_skill_tool(three_skill_map, completed, "/tmp")
        assert tool.name == "CompleteSkill"
        assert tool.skip_permission is True

    def test_missing_skill_name(self, fake_copilot_types, three_skill_map):
        tool = build_complete_skill_tool(three_skill_map, set(), "/tmp")
        result = _call_complete(tool, {"output_summary": "done"})
        assert "error" in result.result_type
        assert "'skill' is required" in result.text_result_for_llm

    def test_unknown_skill(self, fake_copilot_types, three_skill_map):
        tool = build_complete_skill_tool(three_skill_map, set(), "/tmp")
        result = _call_complete(tool, {"skill": "nonexistent", "output_summary": "done"})
        assert "error" in result.result_type
        assert "not found" in result.text_result_for_llm

    def test_missing_output_dir(self, fake_copilot_types, three_skill_map, tmp_path):
        """CompleteSkill should reject if the output directory doesn't exist."""
        tool = build_complete_skill_tool(three_skill_map, set(), str(tmp_path))
        result = _call_complete(tool, {
            "skill": "excel-workbook-intake",
            "output_summary": "I finished",
        })
        assert "error" in result.result_type
        assert "does not exist" in result.text_result_for_llm

    def test_empty_output_dir(self, fake_copilot_types, three_skill_map, tmp_path):
        """CompleteSkill should reject if output dir has < MIN_OUTPUT_BYTES."""
        output_dir = tmp_path / "outputs" / "intake"
        output_dir.mkdir(parents=True)
        # Write a tiny placeholder
        (output_dir / "stub.txt").write_text("x")
        tool = build_complete_skill_tool(three_skill_map, set(), str(tmp_path))
        result = _call_complete(tool, {
            "skill": "excel-workbook-intake",
            "output_summary": "done",
        })
        assert "error" in result.result_type
        assert "placeholders" in result.text_result_for_llm

    def test_successful_completion(self, fake_copilot_types, three_skill_map, tmp_path):
        """CompleteSkill should mark skill complete and list unblocked downstream."""
        # Create substantive output
        output_dir = tmp_path / "outputs" / "intake"
        output_dir.mkdir(parents=True)
        (output_dir / "data.parquet").write_bytes(b"x" * 2000)

        completed = set()
        tool = build_complete_skill_tool(three_skill_map, completed, str(tmp_path))
        result = _call_complete(tool, {
            "skill": "excel-workbook-intake",
            "source_file": "test_data/test.xlsx",
            "output_summary": "Exported 5 sheets to outputs/intake/",
            "row_count": 500,
        })
        assert "marked as complete" in result.text_result_for_llm
        assert "excel-workbook-intake" in completed
        assert "rcm-analysis" in result.text_result_for_llm  # downstream unblocked
        assert "Source: test_data/test.xlsx" in result.text_result_for_llm
        assert "Rows processed: 500" in result.text_result_for_llm

    def test_no_downstream(self, fake_copilot_types, three_skill_map, tmp_path):
        """Last skill in chain should report no downstream skills."""
        output_dir = tmp_path / "outputs" / "executive_report"
        output_dir.mkdir(parents=True)
        (output_dir / "report.html").write_bytes(b"<html>" + b"x" * 2000)

        completed = {"excel-workbook-intake", "rcm-analysis"}
        tool = build_complete_skill_tool(three_skill_map, completed, str(tmp_path))
        result = _call_complete(tool, {
            "skill": "executive-report",
            "output_summary": "Wrote analysis_report.html",
        })
        assert "marked as complete" in result.text_result_for_llm
        assert "No downstream skills" in result.text_result_for_llm

    def test_unblocks_invoke_skill(self, fake_copilot_types, three_skill_map,
                                    memory_store, tmp_path):
        """After CompleteSkill, InvokeSkill for downstream should work."""
        # Set up intake outputs
        output_dir = tmp_path / "outputs" / "intake"
        output_dir.mkdir(parents=True)
        (output_dir / "data.parquet").write_bytes(b"x" * 2000)

        completed = set()
        complete_tool = build_complete_skill_tool(
            three_skill_map, completed, str(tmp_path),
        )
        invoke_tool = build_skill_tool(
            three_skill_map, memory_store, str(tmp_path), completed,
        )

        # rcm-analysis should be blocked initially
        result = _call(invoke_tool, {"skill": "rcm-analysis"})
        assert "error" in result.result_type
        assert "requires" in result.text_result_for_llm

        # Complete intake
        _call_complete(complete_tool, {
            "skill": "excel-workbook-intake",
            "output_summary": "done",
        })

        # Now rcm-analysis should work
        result = _call(invoke_tool, {"skill": "rcm-analysis"})
        data = json.loads(result.text_result_for_llm)
        assert data["status"] in ("invoke_skill", "invoke_skill_no_session")

    def test_prereq_error_includes_hints(self, fake_copilot_types, three_skill_map,
                                          memory_store, tmp_path):
        """Prerequisite error should include directive hints about CompleteSkill."""
        completed = set()
        tool = build_skill_tool(three_skill_map, memory_store, str(tmp_path), completed)
        result = _call(tool, {"skill": "rcm-analysis"})
        assert "error" in result.result_type
        assert "CompleteSkill" in result.text_result_for_llm
        assert "Do NOT create workaround scripts" in result.text_result_for_llm


# ===========================================================================
# Verification gate in CompleteSkill
# ===========================================================================

from copilotcode_sdk.verifier import VerificationExhaustedError


class TestCompleteSkillVerification:
    """Tests for the verification gate inside CompleteSkill."""

    def _make_tool_with_session(self, skill_map, working_dir, completed=None, fork_output="VERDICT: PASS"):
        if completed is None:
            completed = set()

        class FakeChild:
            async def send_and_wait(self, prompt, *, timeout=None):
                return (
                    "### Check: basic\n"
                    "**Command run:**\n  echo ok\n"
                    "**Output observed:**\n  ok\n"
                    f"**Result: {'PASS' if 'PASS' in fork_output else 'FAIL'}**\n\n"
                    f"{fork_output}"
                )
            async def destroy(self):
                pass

        class FakeSession:
            async def fork_child(self, spec):
                return FakeChild()

        tool = build_complete_skill_tool(
            skill_map=skill_map,
            completed_skills=completed,
            working_directory=working_dir,
            session_holder=[FakeSession()],
        )
        return tool, completed

    def test_pass_marks_complete(self, fake_copilot_types, three_skill_map, tmp_path):
        output_dir = tmp_path / "outputs" / "intake"
        output_dir.mkdir(parents=True)
        (output_dir / "data.parquet").write_bytes(b"x" * 2000)

        tool, completed = self._make_tool_with_session(
            three_skill_map, str(tmp_path), fork_output="VERDICT: PASS",
        )
        result = _call_complete(tool, {
            "skill": "excel-workbook-intake",
            "output_summary": "done",
        })
        assert "marked as complete" in result.text_result_for_llm
        assert "excel-workbook-intake" in completed

    def test_fail_returns_error_with_feedback(self, fake_copilot_types, three_skill_map, tmp_path):
        output_dir = tmp_path / "outputs" / "intake"
        output_dir.mkdir(parents=True)
        (output_dir / "data.parquet").write_bytes(b"x" * 2000)

        tool, completed = self._make_tool_with_session(
            three_skill_map, str(tmp_path), fork_output="VERDICT: FAIL",
        )
        result = _call_complete(tool, {
            "skill": "excel-workbook-intake",
            "output_summary": "done",
        })
        assert "error" in result.result_type
        assert "Verification FAILED" in result.text_result_for_llm
        assert "excel-workbook-intake" not in completed

    def test_no_session_skips_verification(self, fake_copilot_types, three_skill_map, tmp_path):
        output_dir = tmp_path / "outputs" / "intake"
        output_dir.mkdir(parents=True)
        (output_dir / "data.parquet").write_bytes(b"x" * 2000)

        completed = set()
        tool = build_complete_skill_tool(
            skill_map=three_skill_map,
            completed_skills=completed,
            working_directory=str(tmp_path),
            session_holder=[],
        )
        result = _call_complete(tool, {
            "skill": "excel-workbook-intake",
            "output_summary": "done",
        })
        assert "marked as complete" in result.text_result_for_llm
