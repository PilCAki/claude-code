"""Tests for the verification gate module."""
from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from copilotcode_sdk.verifier import run_verification
from copilotcode_sdk.subagent import SubagentSpec
from copilotcode_sdk.verifier import (
    snapshot_output_hashes,
    compare_output_hashes,
    VerificationExhaustedError,
    VerificationResult,
    MAX_VERIFICATION_ATTEMPTS,
    MAX_VERIFIER_MALFUNCTIONS,
)
from copilotcode_sdk.verifier import (
    parse_verdict,
    extract_failed_checks,
    build_verifier_prompt,
    format_fail_feedback,
    VERIFIER_SYSTEM_PROMPT,
)
from copilotcode_sdk.verifier import write_failure_trace


# ---------------------------------------------------------------------------
# Fake copilot.types fixture (mirrors test_skill_tool.py)
# ---------------------------------------------------------------------------

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


def _call_complete(tool, arguments: dict):
    """Call a CompleteSkill tool handler directly."""
    import asyncio
    inv = MagicMock()
    inv.arguments = arguments
    result = tool.handler(inv)
    if asyncio.iscoroutine(result):
        return asyncio.run(result)
    return result


class TestHashSnapshotting:
    def test_snapshot_empty_dir(self, tmp_path: Path):
        snapshot = snapshot_output_hashes(tmp_path)
        assert snapshot == {}

    def test_snapshot_files(self, tmp_path: Path):
        (tmp_path / "a.json").write_text('{"key": "value"}')
        (tmp_path / "b.parquet").write_bytes(b"x" * 100)
        snapshot = snapshot_output_hashes(tmp_path)
        assert "a.json" in snapshot
        assert "b.parquet" in snapshot
        expected = hashlib.sha256(b'{"key": "value"}').hexdigest()
        assert snapshot["a.json"] == expected

    def test_snapshot_nested_files(self, tmp_path: Path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.csv").write_text("a,b\n1,2")
        snapshot = snapshot_output_hashes(tmp_path)
        assert "sub/c.csv" in snapshot

    def test_compare_no_changes(self, tmp_path: Path):
        (tmp_path / "a.json").write_text("data")
        before = snapshot_output_hashes(tmp_path)
        changed = compare_output_hashes(tmp_path, before)
        assert changed == []

    def test_compare_detects_modification(self, tmp_path: Path):
        (tmp_path / "a.json").write_text("original")
        before = snapshot_output_hashes(tmp_path)
        (tmp_path / "a.json").write_text("tampered")
        changed = compare_output_hashes(tmp_path, before)
        assert "a.json" in changed

    def test_compare_detects_new_file(self, tmp_path: Path):
        (tmp_path / "a.json").write_text("data")
        before = snapshot_output_hashes(tmp_path)
        (tmp_path / "new.txt").write_text("new")
        changed = compare_output_hashes(tmp_path, before)
        assert "new.txt" in changed

    def test_compare_detects_deletion(self, tmp_path: Path):
        (tmp_path / "a.json").write_text("data")
        before = snapshot_output_hashes(tmp_path)
        (tmp_path / "a.json").unlink()
        changed = compare_output_hashes(tmp_path, before)
        assert "a.json" in changed


class TestVerificationExhaustedError:
    def test_has_trace_path(self, tmp_path: Path):
        err = VerificationExhaustedError("intake", str(tmp_path / "trace.json"))
        assert err.skill_name == "intake"
        assert "trace.json" in err.trace_path
        assert "intake" in str(err)


class TestVerificationResult:
    def test_pass_result(self):
        r = VerificationResult(verdict="PASS", raw_output="all good", failed_checks=[])
        assert r.passed
        assert not r.failed

    def test_fail_result(self):
        r = VerificationResult(
            verdict="FAIL",
            raw_output="stuff broke",
            failed_checks=[{"check": "row count", "detail": "0 rows"}],
        )
        assert not r.passed
        assert r.failed

    def test_malfunction_result(self):
        r = VerificationResult(verdict="MALFUNCTION", raw_output="no verdict found", failed_checks=[])
        assert not r.passed
        assert not r.failed
        assert r.is_malfunction


class TestConstants:
    def test_max_attempts_is_5(self):
        assert MAX_VERIFICATION_ATTEMPTS == 5

    def test_max_malfunctions_is_2(self):
        assert MAX_VERIFIER_MALFUNCTIONS == 2


class TestParseVerdict:
    def test_parse_pass(self):
        output = "### Check: row count\n**Result: PASS**\n\nVERDICT: PASS"
        assert parse_verdict(output) == "PASS"

    def test_parse_fail(self):
        output = "### Check: schema\n**Result: FAIL**\n\nVERDICT: FAIL"
        assert parse_verdict(output) == "FAIL"

    def test_parse_partial(self):
        output = "VERDICT: PARTIAL"
        assert parse_verdict(output) == "PARTIAL"

    def test_no_verdict_returns_none(self):
        output = "I checked some stuff and it looks fine."
        assert parse_verdict(output) is None

    def test_ignores_verdict_in_middle_of_line(self):
        output = "The VERDICT: PASS was issued.\nVERDICT: FAIL"
        assert parse_verdict(output) == "FAIL"

    def test_verdict_with_trailing_whitespace(self):
        output = "VERDICT: PASS   \n"
        assert parse_verdict(output) == "PASS"


class TestExtractFailedChecks:
    def test_extract_single_fail(self):
        output = (
            "### Check: Row count matches source\n"
            "**Command run:**\n"
            "  duckdb -c \"SELECT count(*) FROM 'data.parquet'\"\n"
            "**Output observed:**\n"
            "  0\n"
            "**Result: FAIL**\n"
            "Expected: ~15000 rows. Actual: 0 rows.\n"
            "\nVERDICT: FAIL"
        )
        checks = extract_failed_checks(output)
        assert len(checks) == 1
        assert checks[0]["check"] == "Row count matches source"
        assert "duckdb" in checks[0]["command"]
        assert "0" in checks[0]["observed"]

    def test_extract_multiple_fails(self):
        output = (
            "### Check: File exists\n"
            "**Command run:**\n  ls data.parquet\n"
            "**Output observed:**\n  No such file\n"
            "**Result: FAIL**\n"
            "\n"
            "### Check: Schema valid\n"
            "**Command run:**\n  duckdb -c \"DESCRIBE...\"\n"
            "**Output observed:**\n  charge VARCHAR\n"
            "**Result: FAIL**\n"
            "\n"
            "### Check: Has manifest\n"
            "**Command run:**\n  cat manifest.json\n"
            "**Output observed:**\n  {\"source\": \"test.xlsx\"}\n"
            "**Result: PASS**\n"
            "\nVERDICT: FAIL"
        )
        checks = extract_failed_checks(output)
        assert len(checks) == 2
        assert checks[0]["check"] == "File exists"
        assert checks[1]["check"] == "Schema valid"

    def test_extract_no_fails(self):
        output = (
            "### Check: Everything\n"
            "**Command run:**\n  echo ok\n"
            "**Output observed:**\n  ok\n"
            "**Result: PASS**\n"
            "\nVERDICT: PASS"
        )
        checks = extract_failed_checks(output)
        assert checks == []


class TestBuildVerifierPrompt:
    def test_contains_skill_content(self):
        prompt = build_verifier_prompt(
            skill_content="# Intake\nProcess sheets.",
            output_dir="/work/outputs/intake",
            prior_metrics=None,
        )
        assert "# Intake" in prompt
        assert "Process sheets." in prompt

    def test_contains_output_dir(self):
        prompt = build_verifier_prompt(
            skill_content="# Skill",
            output_dir="/work/outputs/intake",
            prior_metrics=None,
        )
        assert "/work/outputs/intake" in prompt

    def test_contains_prior_metrics(self):
        prompt = build_verifier_prompt(
            skill_content="# Skill",
            output_dir="/out",
            prior_metrics='{"row_count": 15000}',
        )
        assert "15000" in prompt

    def test_no_prior_metrics(self):
        prompt = build_verifier_prompt(
            skill_content="# Skill",
            output_dir="/out",
            prior_metrics=None,
        )
        assert "None available" in prompt

    def test_contains_verification_only_preface(self):
        prompt = build_verifier_prompt(
            skill_content="# Skill",
            output_dir="/out",
            prior_metrics=None,
        )
        assert "VERIFICATION ONLY" in prompt
        assert "Do NOT implement" in prompt


class TestFormatFailFeedback:
    def test_format_with_checks(self):
        checks = [
            {"check": "Row count", "command": "duckdb -c 'SELECT...'", "observed": "0"},
            {"check": "Schema", "command": "duckdb -c 'DESCRIBE...'", "observed": "VARCHAR"},
        ]
        feedback = format_fail_feedback(checks, attempt=2, max_attempts=5)
        assert "attempt 2 of 5" in feedback
        assert "Row count" in feedback
        assert "Schema" in feedback
        assert "duckdb" in feedback

    def test_format_empty_checks(self):
        feedback = format_fail_feedback([], attempt=1, max_attempts=5)
        assert "attempt 1 of 5" in feedback


class TestVerifierSystemPrompt:
    def test_prompt_requires_commands(self):
        assert "MUST run commands" in VERIFIER_SYSTEM_PROMPT

    def test_prompt_forbids_modifications(self):
        assert "Do NOT create, modify, or fix" in VERIFIER_SYSTEM_PROMPT

    def test_prompt_has_verdict_format(self):
        assert "VERDICT: PASS" in VERIFIER_SYSTEM_PROMPT
        assert "VERDICT: FAIL" in VERIFIER_SYSTEM_PROMPT
        assert "VERDICT: PARTIAL" in VERIFIER_SYSTEM_PROMPT

    def test_prompt_has_rationalization_warnings(self):
        assert "reading is not verification" in VERIFIER_SYSTEM_PROMPT


class TestWriteFailureTrace:
    def test_writes_valid_json(self, tmp_path: Path):
        history = [
            {
                "attempt": 1,
                "failed_checks": [
                    {"check": "row count", "command": "duckdb ...", "observed": "0"}
                ],
            },
            {
                "attempt": 2,
                "failed_checks": [
                    {"check": "schema", "command": "duckdb ...", "observed": "VARCHAR"}
                ],
            },
        ]
        output_dir = tmp_path / "outputs" / "intake"
        output_dir.mkdir(parents=True)
        (output_dir / "data.parquet").write_bytes(b"x" * 2000)

        trace_path = write_failure_trace(
            skill_name="excel-workbook-intake",
            history=history,
            output_dir=output_dir,
            workspace=tmp_path,
        )
        assert Path(trace_path).exists()
        data = json.loads(Path(trace_path).read_text())
        assert data["skill"] == "excel-workbook-intake"
        assert data["attempts"] == 2
        assert len(data["history"]) == 2
        assert "output_snapshot" in data
        assert "data.parquet" in data["output_snapshot"]["files"]

    def test_creates_verification_failures_dir(self, tmp_path: Path):
        trace_path = write_failure_trace(
            skill_name="intake",
            history=[],
            output_dir=tmp_path,
            workspace=tmp_path,
        )
        assert "verification_failures" in trace_path
        assert Path(trace_path).parent.exists()

    def test_trace_has_timestamp(self, tmp_path: Path):
        trace_path = write_failure_trace(
            skill_name="intake",
            history=[],
            output_dir=tmp_path,
            workspace=tmp_path,
        )
        data = json.loads(Path(trace_path).read_text())
        assert "timestamp" in data


class FakeEnforcedChild:
    """Fake child session that returns canned output."""
    def __init__(self, output: str):
        self._output = output
        self.spec = SubagentSpec(role="skill-verifier", system_prompt_suffix="test")

    async def send_and_wait(self, prompt, *, timeout=None):
        return self._output

    async def get_last_response_text(self):
        return self._output

    async def destroy(self):
        pass


class TestRunVerification:
    def test_pass_returns_success(self, tmp_path: Path):
        output_dir = tmp_path / "outputs" / "intake"
        output_dir.mkdir(parents=True)
        (output_dir / "data.parquet").write_bytes(b"x" * 2000)

        verifier_output = (
            "### Check: Files exist\n"
            "**Command run:**\n  ls outputs/intake/\n"
            "**Output observed:**\n  data.parquet\n"
            "**Result: PASS**\n\n"
            "VERDICT: PASS"
        )

        async def fake_fork_child(spec):
            return FakeEnforcedChild(verifier_output)

        result = asyncio.run(run_verification(
            skill_name="excel-workbook-intake",
            skill_content="# Intake\nProcess sheets.",
            output_dir=output_dir,
            workspace=tmp_path,
            fork_child=fake_fork_child,
            prior_metrics=None,
        ))
        assert result.passed

    def test_fail_returns_failed_checks(self, tmp_path: Path):
        output_dir = tmp_path / "outputs" / "intake"
        output_dir.mkdir(parents=True)
        (output_dir / "data.parquet").write_bytes(b"x" * 2000)

        verifier_output = (
            "### Check: Row count\n"
            "**Command run:**\n  duckdb -c \"SELECT count(*)\"\n"
            "**Output observed:**\n  0\n"
            "**Result: FAIL**\n\n"
            "VERDICT: FAIL"
        )

        async def fake_fork_child(spec):
            return FakeEnforcedChild(verifier_output)

        result = asyncio.run(run_verification(
            skill_name="excel-workbook-intake",
            skill_content="# Intake",
            output_dir=output_dir,
            workspace=tmp_path,
            fork_child=fake_fork_child,
            prior_metrics=None,
        ))
        assert result.failed
        assert len(result.failed_checks) == 1
        assert result.failed_checks[0]["check"] == "Row count"

    def test_no_verdict_is_malfunction(self, tmp_path: Path):
        output_dir = tmp_path / "outputs" / "intake"
        output_dir.mkdir(parents=True)
        (output_dir / "data.parquet").write_bytes(b"x" * 2000)

        async def fake_fork_child(spec):
            return FakeEnforcedChild("I looked at the files and they seem fine.")

        result = asyncio.run(run_verification(
            skill_name="excel-workbook-intake",
            skill_content="# Intake",
            output_dir=output_dir,
            workspace=tmp_path,
            fork_child=fake_fork_child,
            prior_metrics=None,
        ))
        assert result.is_malfunction

    def test_no_commands_is_malfunction(self, tmp_path: Path):
        output_dir = tmp_path / "outputs" / "intake"
        output_dir.mkdir(parents=True)
        (output_dir / "data.parquet").write_bytes(b"x" * 2000)

        async def fake_fork_child(spec):
            return FakeEnforcedChild("VERDICT: PASS")

        result = asyncio.run(run_verification(
            skill_name="excel-workbook-intake",
            skill_content="# Intake",
            output_dir=output_dir,
            workspace=tmp_path,
            fork_child=fake_fork_child,
            prior_metrics=None,
        ))
        assert result.is_malfunction

    def test_tampered_outputs_is_malfunction(self, tmp_path: Path):
        output_dir = tmp_path / "outputs" / "intake"
        output_dir.mkdir(parents=True)
        data_file = output_dir / "data.parquet"
        data_file.write_bytes(b"x" * 2000)

        async def fake_fork_child(spec):
            data_file.write_bytes(b"TAMPERED" * 500)
            return FakeEnforcedChild(
                "### Check: test\n**Command run:**\n  echo ok\n"
                "**Output observed:**\n  ok\n**Result: PASS**\n\n"
                "VERDICT: PASS"
            )

        result = asyncio.run(run_verification(
            skill_name="excel-workbook-intake",
            skill_content="# Intake",
            output_dir=output_dir,
            workspace=tmp_path,
            fork_child=fake_fork_child,
            prior_metrics=None,
        ))
        assert result.is_malfunction
        assert "tamper" in result.raw_output.lower()

    def test_fork_child_receives_correct_spec(self, tmp_path: Path):
        output_dir = tmp_path / "outputs" / "intake"
        output_dir.mkdir(parents=True)
        (output_dir / "data.parquet").write_bytes(b"x" * 2000)

        captured_spec = None

        async def fake_fork_child(spec):
            nonlocal captured_spec
            captured_spec = spec
            return FakeEnforcedChild(
                "### Check: ok\n**Command run:**\n  echo ok\n"
                "**Output observed:**\n  ok\n**Result: PASS**\n\n"
                "VERDICT: PASS"
            )

        asyncio.run(run_verification(
            skill_name="test",
            skill_content="# Test",
            output_dir=output_dir,
            workspace=tmp_path,
            fork_child=fake_fork_child,
            prior_metrics=None,
        ))
        assert captured_spec is not None
        assert captured_spec.role == "skill-verifier"
        assert captured_spec.max_turns == 20
        assert captured_spec.timeout_seconds == 3600.0


# ---------------------------------------------------------------------------
# Full integration tests
# ---------------------------------------------------------------------------

class TestFullRetryLoop:
    """Simulate the full CompleteSkill → verify → retry → exhaust loop."""

    def test_exhaustion_writes_trace_and_raises(self, tmp_path, fake_copilot_types):
        """Simulate 5 consecutive FAILs and verify trace is written."""
        from copilotcode_sdk.skill_tool import build_complete_skill_tool

        skill_map = {
            "test-skill": {
                "name": "test-skill",
                "type": "test",
                "outputs": "outputs/test",
                "requires": "none",
                "_path": str(tmp_path / "SKILL.md"),
            },
        }
        (tmp_path / "SKILL.md").write_text("---\nname: test-skill\n---\n# Test\n")
        output_dir = tmp_path / "outputs" / "test"
        output_dir.mkdir(parents=True)
        (output_dir / "data.json").write_bytes(b"x" * 2000)

        fail_output = (
            "### Check: broken\n"
            "**Command run:**\n  echo fail\n"
            "**Output observed:**\n  fail\n"
            "**Result: FAIL**\n\n"
            "VERDICT: FAIL"
        )

        class FakeChild:
            def __init__(self):
                self._last = ""
            async def send_and_wait(self, prompt, *, timeout=None):
                self._last = fail_output
                return self._last
            async def get_last_response_text(self):
                return self._last
            async def destroy(self):
                pass

        class FakeSession:
            async def fork_child(self, spec, **kw):
                return FakeChild()

        completed = set()
        tool = build_complete_skill_tool(
            skill_map=skill_map,
            completed_skills=completed,
            working_directory=str(tmp_path),
            session_holder=[FakeSession()],
        )

        import asyncio

        inv = MagicMock()
        inv.arguments = {"skill": "test-skill", "output_summary": "done"}

        def _run(inv):
            r = tool.handler(inv)
            return asyncio.run(r) if asyncio.iscoroutine(r) else r

        # Call 4 times — should get errors
        for i in range(4):
            result = _run(inv)
            assert "error" in result.result_type
            assert "Verification FAILED" in result.text_result_for_llm

        # 5th call should raise VerificationExhaustedError
        with pytest.raises(VerificationExhaustedError) as exc_info:
            _run(inv)

        assert exc_info.value.skill_name == "test-skill"
        assert Path(exc_info.value.trace_path).exists()

        # Verify trace content
        trace = json.loads(Path(exc_info.value.trace_path).read_text())
        assert trace["skill"] == "test-skill"
        assert trace["attempts"] == 5
        assert len(trace["history"]) == 5

    def test_pass_after_retries(self, tmp_path, fake_copilot_types):
        """Fail twice, then pass on third attempt."""
        from copilotcode_sdk.skill_tool import build_complete_skill_tool

        skill_map = {
            "test-skill": {
                "name": "test-skill",
                "type": "test",
                "outputs": "outputs/test",
                "requires": "none",
                "_path": str(tmp_path / "SKILL.md"),
            },
        }
        (tmp_path / "SKILL.md").write_text("---\nname: test-skill\n---\n# Test\n")
        output_dir = tmp_path / "outputs" / "test"
        output_dir.mkdir(parents=True)
        (output_dir / "data.json").write_bytes(b"x" * 2000)

        call_count = [0]

        class FakeChild:
            def __init__(self):
                self._last = ""
            async def send_and_wait(self, prompt, *, timeout=None):
                call_count[0] += 1
                if call_count[0] <= 2:
                    self._last = (
                        "### Check: broken\n"
                        "**Command run:**\n  echo fail\n"
                        "**Output observed:**\n  fail\n"
                        "**Result: FAIL**\n\n"
                        "VERDICT: FAIL"
                    )
                else:
                    self._last = (
                        "### Check: fixed\n"
                        "**Command run:**\n  echo ok\n"
                        "**Output observed:**\n  ok\n"
                        "**Result: PASS**\n\n"
                        "VERDICT: PASS"
                    )
                return self._last
            async def get_last_response_text(self):
                return self._last
            async def destroy(self):
                pass

        class FakeSession:
            async def fork_child(self, spec, **kw):
                return FakeChild()

        completed = set()
        tool = build_complete_skill_tool(
            skill_map=skill_map,
            completed_skills=completed,
            working_directory=str(tmp_path),
            session_holder=[FakeSession()],
        )

        import asyncio

        inv = MagicMock()
        inv.arguments = {"skill": "test-skill", "output_summary": "done"}

        def _run(inv):
            r = tool.handler(inv)
            return asyncio.run(r) if asyncio.iscoroutine(r) else r

        # First 2 calls fail
        for _ in range(2):
            result = _run(inv)
            assert "error" in result.result_type

        # Third call passes
        result = _run(inv)
        assert "marked as complete" in result.text_result_for_llm
        assert "test-skill" in completed

    def test_malfunction_does_not_count_as_attempt(self, tmp_path, fake_copilot_types):
        """Verifier malfunctions shouldn't burn implementer's attempts."""
        import asyncio
        from copilotcode_sdk.skill_tool import build_complete_skill_tool

        skill_map = {
            "test-skill": {
                "name": "test-skill",
                "type": "test",
                "outputs": "outputs/test",
                "requires": "none",
                "_path": str(tmp_path / "SKILL.md"),
            },
        }
        (tmp_path / "SKILL.md").write_text("---\nname: test-skill\n---\n# Test\n")
        output_dir = tmp_path / "outputs" / "test"
        output_dir.mkdir(parents=True)
        (output_dir / "data.json").write_bytes(b"x" * 2000)

        class FakeChild:
            def __init__(self):
                self._last = ""
            async def send_and_wait(self, prompt, *, timeout=None):
                self._last = "I looked at the files and they seem fine."
                return self._last
            async def get_last_response_text(self):
                return self._last
            async def destroy(self):
                pass

        class FakeSession:
            async def fork_child(self, spec, **kw):
                return FakeChild()

        completed = set()
        tool = build_complete_skill_tool(
            skill_map=skill_map,
            completed_skills=completed,
            working_directory=str(tmp_path),
            session_holder=[FakeSession()],
        )

        inv = MagicMock()
        inv.arguments = {"skill": "test-skill", "output_summary": "done"}

        def _run(inv):
            r = tool.handler(inv)
            return asyncio.run(r) if asyncio.iscoroutine(r) else r

        # First 2 malfunctions — should not count as attempts
        for _ in range(2):
            result = _run(inv)
            assert "error" in result.result_type
            assert "MALFUNCTION" in result.text_result_for_llm
