"""Tests for the holistic verification module."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from copilotcode_sdk.holistic_verifier import (
    HOLISTIC_VERIFIER_SYSTEM_PROMPT,
    _find_carry_forward_artifacts,
    build_holistic_checker,
    build_holistic_verifier_prompt,
    format_holistic_feedback,
    run_holistic_verification,
)
from copilotcode_sdk.subagent import SubagentSpec
from copilotcode_sdk.verifier import VerificationResult


# ---------------------------------------------------------------------------
# Fake child session (mirrors test_verifier.py pattern)
# ---------------------------------------------------------------------------

class FakeEnforcedChild:
    """Fake child session that returns canned output."""
    def __init__(self, output: str, tamper_file: Path | None = None):
        self._output = output
        self._tamper_file = tamper_file
        self.spec = SubagentSpec(role="holistic-verifier", system_prompt_suffix="test")

    async def send_and_wait(self, prompt, *, timeout=None):
        # Optionally tamper with a file during "verification"
        if self._tamper_file is not None:
            self._tamper_file.write_text("TAMPERED")
        return self._output

    async def get_last_response_text(self):
        return self._output

    async def destroy(self):
        pass


# ---------------------------------------------------------------------------
# System prompt tests
# ---------------------------------------------------------------------------

class TestHolisticVerifierSystemPrompt:
    def test_is_domain_agnostic(self):
        """Prompt must NOT contain RCM-specific terms."""
        rcm_terms = [
            "charges", "payments", "payer", "denial", "aging",
            "realization rate", "RCM", "billing",
        ]
        lower_prompt = HOLISTIC_VERIFIER_SYSTEM_PROMPT.lower()
        for term in rcm_terms:
            assert term.lower() not in lower_prompt, (
                f"System prompt contains domain-specific term: '{term}'"
            )

    def test_has_generic_categories(self):
        assert "CROSS-SKILL CONSISTENCY" in HOLISTIC_VERIFIER_SYSTEM_PROMPT
        assert "PIPELINE COMPLETENESS" in HOLISTIC_VERIFIER_SYSTEM_PROMPT
        assert "CARRY-FORWARD INTEGRITY" in HOLISTIC_VERIFIER_SYSTEM_PROMPT
        assert "AGGREGATE QUALITY" in HOLISTIC_VERIFIER_SYSTEM_PROMPT
        assert "ANTI-GAMING" in HOLISTIC_VERIFIER_SYSTEM_PROMPT

    def test_requires_derived_check_plan(self):
        assert "DERIVE YOUR CHECK PLAN" in HOLISTIC_VERIFIER_SYSTEM_PROMPT

    def test_requires_commands(self):
        assert "MUST run commands" in HOLISTIC_VERIFIER_SYSTEM_PROMPT

    def test_forbids_modifications(self):
        assert "Do NOT create, modify, or fix" in HOLISTIC_VERIFIER_SYSTEM_PROMPT

    def test_has_verdict_format(self):
        assert "VERDICT: PASS" in HOLISTIC_VERIFIER_SYSTEM_PROMPT
        assert "VERDICT: FAIL" in HOLISTIC_VERIFIER_SYSTEM_PROMPT
        assert "VERDICT: PARTIAL" in HOLISTIC_VERIFIER_SYSTEM_PROMPT

    def test_instructs_to_read_skill_definitions(self):
        prompt_lower = HOLISTIC_VERIFIER_SYSTEM_PROMPT.lower()
        assert "skill definition" in prompt_lower


# ---------------------------------------------------------------------------
# Prompt builder tests
# ---------------------------------------------------------------------------

class TestBuildHolisticVerifierPrompt:
    def test_contains_all_skill_definitions(self):
        prompt = build_holistic_verifier_prompt(
            skill_definitions={
                "intake": "# Intake\nProcess workbooks.",
                "analysis": "# Analysis\nCompute metrics.",
            },
            output_dirs={
                "intake": "/work/outputs/intake",
                "analysis": "/work/outputs/analysis",
            },
            workspace="/work",
        )
        assert "# Intake" in prompt
        assert "Process workbooks." in prompt
        assert "# Analysis" in prompt
        assert "Compute metrics." in prompt

    def test_contains_output_dirs(self):
        prompt = build_holistic_verifier_prompt(
            skill_definitions={"intake": "# Intake"},
            output_dirs={"intake": "/work/outputs/intake"},
            workspace="/work",
        )
        assert "/work/outputs/intake" in prompt

    def test_contains_workspace(self):
        prompt = build_holistic_verifier_prompt(
            skill_definitions={"intake": "# Intake"},
            output_dirs={"intake": "/out"},
            workspace="/my/workspace",
        )
        assert "/my/workspace" in prompt

    def test_contains_carry_forward_artifacts(self):
        prompt = build_holistic_verifier_prompt(
            skill_definitions={"intake": "# Intake"},
            output_dirs={"intake": "/out"},
            workspace="/work",
            carry_forward_artifacts=[
                "/work/outputs/prior_run_metrics.json",
                "/work/outputs/column_mapping.json",
            ],
        )
        assert "prior_run_metrics.json" in prompt
        assert "column_mapping.json" in prompt
        assert "CARRY-FORWARD ARTIFACTS FOUND" in prompt

    def test_no_carry_forward_says_none(self):
        prompt = build_holistic_verifier_prompt(
            skill_definitions={"intake": "# Intake"},
            output_dirs={"intake": "/out"},
            workspace="/work",
            carry_forward_artifacts=[],
        )
        assert "None found" in prompt

    def test_holistic_verification_only_preface(self):
        prompt = build_holistic_verifier_prompt(
            skill_definitions={"intake": "# Intake"},
            output_dirs={"intake": "/out"},
            workspace="/work",
        )
        assert "HOLISTIC VERIFICATION ONLY" in prompt
        assert "Do NOT implement" in prompt


# ---------------------------------------------------------------------------
# Carry-forward artifact discovery
# ---------------------------------------------------------------------------

class TestFindCarryForwardArtifacts:
    def test_finds_prior_run_metrics(self, tmp_path: Path):
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        (outputs / "prior_run_metrics.json").write_text('{"key": 1}')
        artifacts = _find_carry_forward_artifacts(tmp_path)
        assert any("prior_run_metrics.json" in a for a in artifacts)

    def test_finds_column_mapping(self, tmp_path: Path):
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        (outputs / "column_mapping.json").write_text('{}')
        artifacts = _find_carry_forward_artifacts(tmp_path)
        assert any("column_mapping.json" in a for a in artifacts)

    def test_finds_memory_md(self, tmp_path: Path):
        (tmp_path / "MEMORY.md").write_text("# Memory")
        artifacts = _find_carry_forward_artifacts(tmp_path)
        assert any("MEMORY.md" in a for a in artifacts)

    def test_finds_files_with_prior_in_name(self, tmp_path: Path):
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        (outputs / "prior_learnings.json").write_text("[]")
        artifacts = _find_carry_forward_artifacts(tmp_path)
        assert any("prior_learnings.json" in a for a in artifacts)

    def test_empty_workspace(self, tmp_path: Path):
        artifacts = _find_carry_forward_artifacts(tmp_path)
        assert artifacts == []

    def test_no_duplicates(self, tmp_path: Path):
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        (outputs / "prior_run_metrics.json").write_text('{}')
        artifacts = _find_carry_forward_artifacts(tmp_path)
        assert len(artifacts) == len(set(artifacts))


# ---------------------------------------------------------------------------
# run_holistic_verification tests
# ---------------------------------------------------------------------------

class TestRunHolisticVerification:
    def test_pass_returns_success(self, tmp_path: Path):
        output_dir = tmp_path / "outputs" / "intake"
        output_dir.mkdir(parents=True)
        (output_dir / "data.parquet").write_bytes(b"x" * 100)

        verifier_output = (
            "### Check: Cross-skill consistency\n"
            "**Command run:**\n  cat outputs/intake/data.parquet | wc -c\n"
            "**Output observed:**\n  100\n"
            "**Result: PASS**\n\n"
            "VERDICT: PASS"
        )

        async def fake_fork(spec, **kwargs):
            return FakeEnforcedChild(verifier_output)

        result = asyncio.run(run_holistic_verification(
            skill_definitions={"intake": "# Intake"},
            output_dirs={"intake": str(output_dir)},
            workspace=str(tmp_path),
            fork_child=fake_fork,
        ))
        assert result.passed

    def test_fail_returns_failed_checks(self, tmp_path: Path):
        output_dir = tmp_path / "outputs" / "intake"
        output_dir.mkdir(parents=True)
        (output_dir / "data.parquet").write_bytes(b"x" * 100)

        verifier_output = (
            "### Check: Numbers match\n"
            "**Command run:**\n  diff a.json b.json\n"
            "**Output observed:**\n  mismatch on line 3\n"
            "**Result: FAIL**\n\n"
            "VERDICT: FAIL"
        )

        async def fake_fork(spec, **kwargs):
            return FakeEnforcedChild(verifier_output)

        result = asyncio.run(run_holistic_verification(
            skill_definitions={"intake": "# Intake"},
            output_dirs={"intake": str(output_dir)},
            workspace=str(tmp_path),
            fork_child=fake_fork,
        ))
        assert result.failed
        assert len(result.failed_checks) == 1
        assert result.failed_checks[0]["check"] == "Numbers match"

    def test_no_verdict_is_malfunction(self, tmp_path: Path):
        async def fake_fork(spec, **kwargs):
            return FakeEnforcedChild("Looks good to me!")

        result = asyncio.run(run_holistic_verification(
            skill_definitions={"intake": "# Intake"},
            output_dirs={},
            workspace=str(tmp_path),
            fork_child=fake_fork,
        ))
        assert result.is_malfunction

    def test_no_commands_is_malfunction(self, tmp_path: Path):
        async def fake_fork(spec, **kwargs):
            return FakeEnforcedChild("VERDICT: PASS")

        result = asyncio.run(run_holistic_verification(
            skill_definitions={"intake": "# Intake"},
            output_dirs={},
            workspace=str(tmp_path),
            fork_child=fake_fork,
        ))
        assert result.is_malfunction

    def test_tampered_output_is_malfunction(self, tmp_path: Path):
        output_dir = tmp_path / "outputs" / "intake"
        output_dir.mkdir(parents=True)
        data_file = output_dir / "data.json"
        data_file.write_text('{"original": true}')

        verifier_output = (
            "### Check: test\n"
            "**Command run:**\n  echo ok\n"
            "**Output observed:**\n  ok\n"
            "**Result: PASS**\n\n"
            "VERDICT: PASS"
        )

        async def fake_fork(spec, **kwargs):
            return FakeEnforcedChild(verifier_output, tamper_file=data_file)

        result = asyncio.run(run_holistic_verification(
            skill_definitions={"intake": "# Intake"},
            output_dirs={"intake": str(output_dir)},
            workspace=str(tmp_path),
            fork_child=fake_fork,
        ))
        assert result.is_malfunction
        assert "modified" in result.raw_output.lower()

    def test_pass_overridden_when_checks_fail(self, tmp_path: Path):
        """If verifier says PASS but individual checks say FAIL, override."""
        verifier_output = (
            "### Check: consistency\n"
            "**Command run:**\n  diff a b\n"
            "**Output observed:**\n  mismatch\n"
            "**Result: FAIL**\n\n"
            "### Check: completeness\n"
            "**Command run:**\n  ls -la\n"
            "**Output observed:**\n  ok\n"
            "**Result: PASS**\n\n"
            "VERDICT: PASS"
        )

        async def fake_fork(spec, **kwargs):
            return FakeEnforcedChild(verifier_output)

        result = asyncio.run(run_holistic_verification(
            skill_definitions={"intake": "# Intake"},
            output_dirs={},
            workspace=str(tmp_path),
            fork_child=fake_fork,
        ))
        assert result.failed  # Overridden from PASS to FAIL
        assert len(result.failed_checks) == 1

    def test_fork_child_receives_correct_spec(self, tmp_path: Path):
        captured_spec = None

        async def fake_fork(spec, **kwargs):
            nonlocal captured_spec
            captured_spec = spec
            return FakeEnforcedChild(
                "### Check: ok\n**Command run:**\n  echo ok\n"
                "**Output observed:**\n  ok\n**Result: PASS**\n\n"
                "VERDICT: PASS"
            )

        asyncio.run(run_holistic_verification(
            skill_definitions={"intake": "# Intake"},
            output_dirs={},
            workspace=str(tmp_path),
            fork_child=fake_fork,
        ))
        assert captured_spec is not None
        assert captured_spec.role == "holistic-verifier"
        assert captured_spec.max_turns == 30
        assert captured_spec.timeout_seconds == 3600.0

    def test_fork_crash_is_malfunction(self, tmp_path: Path):
        async def crashing_fork(spec, **kwargs):
            raise RuntimeError("connection lost")

        result = asyncio.run(run_holistic_verification(
            skill_definitions={"intake": "# Intake"},
            output_dirs={},
            workspace=str(tmp_path),
            fork_child=crashing_fork,
        ))
        assert result.is_malfunction
        assert "crashed" in result.raw_output.lower()

    def test_carry_forward_auto_discovered(self, tmp_path: Path):
        """When carry_forward_artifacts is None, they're auto-discovered."""
        outputs = tmp_path / "outputs"
        outputs.mkdir()
        (outputs / "prior_run_metrics.json").write_text('{"total": 100}')

        captured_prompt = None

        async def fake_fork(spec, **kwargs):
            return FakeEnforcedChild(
                "### Check: ok\n**Command run:**\n  echo ok\n"
                "**Output observed:**\n  ok\n**Result: PASS**\n\n"
                "VERDICT: PASS"
            )

        # Patch build_holistic_verifier_prompt to capture what it receives
        original_build = build_holistic_verifier_prompt
        def capturing_build(**kwargs):
            nonlocal captured_prompt
            captured_prompt = kwargs.get("carry_forward_artifacts")
            return original_build(**kwargs)

        with patch(
            "copilotcode_sdk.holistic_verifier.build_holistic_verifier_prompt",
            side_effect=capturing_build,
        ):
            asyncio.run(run_holistic_verification(
                skill_definitions={"intake": "# Intake"},
                output_dirs={},
                workspace=str(tmp_path),
                fork_child=fake_fork,
                carry_forward_artifacts=None,
            ))

        assert captured_prompt is not None
        assert any("prior_run_metrics.json" in a for a in captured_prompt)


# ---------------------------------------------------------------------------
# format_holistic_feedback tests
# ---------------------------------------------------------------------------

class TestFormatHolisticFeedback:
    def test_format_with_checks(self):
        result = VerificationResult(
            verdict="FAIL",
            raw_output="...",
            failed_checks=[
                {"check": "Numbers match", "command": "diff a b", "observed": "mismatch"},
                {"check": "Schema aligned", "command": "cat schema", "observed": "wrong type"},
            ],
        )
        feedback = format_holistic_feedback(result)
        assert "Numbers match" in feedback
        assert "Schema aligned" in feedback
        assert "force=true" in feedback

    def test_format_empty_checks(self):
        result = VerificationResult(
            verdict="FAIL",
            raw_output="generic failure",
            failed_checks=[],
        )
        feedback = format_holistic_feedback(result)
        assert "FAIL" in feedback
        assert "no specific checks were extracted" in feedback.lower()


# ---------------------------------------------------------------------------
# build_holistic_checker tests
# ---------------------------------------------------------------------------

class TestBuildHolisticChecker:
    def test_builds_callable(self, tmp_path: Path):
        skill_map = {
            "intake": {
                "name": "intake",
                "type": "data-intake",
                "outputs": "outputs/intake",
                "_path": str(tmp_path / "SKILL.md"),
            },
        }
        (tmp_path / "SKILL.md").write_text("# Intake\nProcess workbooks.")
        (tmp_path / "outputs" / "intake").mkdir(parents=True)

        async def fake_fork(spec, **kwargs):
            return FakeEnforcedChild(
                "### Check: ok\n**Command run:**\n  echo ok\n"
                "**Output observed:**\n  ok\n**Result: PASS**\n\n"
                "VERDICT: PASS"
            )

        checker = build_holistic_checker(
            skill_map=skill_map,
            workspace=tmp_path,
            fork_child=fake_fork,
        )
        result = asyncio.run(checker())
        assert result.passed

    def test_reads_skill_content(self, tmp_path: Path):
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("# My Custom Skill\nDoes things.")
        skill_map = {
            "custom": {
                "name": "custom",
                "type": "custom",
                "outputs": "outputs/custom",
                "_path": str(skill_md),
            },
        }
        (tmp_path / "outputs" / "custom").mkdir(parents=True)

        captured_definitions = None

        original_run = run_holistic_verification
        async def capturing_run(**kwargs):
            nonlocal captured_definitions
            captured_definitions = kwargs.get("skill_definitions")
            return VerificationResult(verdict="PASS", raw_output="ok", failed_checks=[])

        with patch(
            "copilotcode_sdk.holistic_verifier.run_holistic_verification",
            side_effect=capturing_run,
        ):
            checker = build_holistic_checker(
                skill_map=skill_map,
                workspace=tmp_path,
                fork_child=AsyncMock(),
            )
            asyncio.run(checker())

        assert captured_definitions is not None
        assert "custom" in captured_definitions
        assert "My Custom Skill" in captured_definitions["custom"]

    def test_uses_custom_read_skill_content(self, tmp_path: Path):
        skill_map = {
            "test": {
                "name": "test",
                "type": "test",
                "outputs": "outputs/test",
                "_path": str(tmp_path / "nonexistent.md"),
            },
        }

        def custom_reader(smap, name):
            return "# Custom Reader Output"

        captured_definitions = None

        async def capturing_run(**kwargs):
            nonlocal captured_definitions
            captured_definitions = kwargs.get("skill_definitions")
            return VerificationResult(verdict="PASS", raw_output="ok", failed_checks=[])

        with patch(
            "copilotcode_sdk.holistic_verifier.run_holistic_verification",
            side_effect=capturing_run,
        ):
            checker = build_holistic_checker(
                skill_map=skill_map,
                workspace=tmp_path,
                fork_child=AsyncMock(),
                read_skill_content=custom_reader,
            )
            asyncio.run(checker())

        assert captured_definitions is not None
        assert captured_definitions["test"] == "# Custom Reader Output"


# ---------------------------------------------------------------------------
# Verification log tests
# ---------------------------------------------------------------------------

class TestVerificationLogging:
    def test_pass_creates_log_file(self, tmp_path: Path):
        verifier_output = (
            "### Check: ok\n**Command run:**\n  echo ok\n"
            "**Output observed:**\n  ok\n**Result: PASS**\n\n"
            "VERDICT: PASS"
        )

        async def fake_fork(spec, **kwargs):
            return FakeEnforcedChild(verifier_output)

        asyncio.run(run_holistic_verification(
            skill_definitions={"intake": "# Intake"},
            output_dirs={},
            workspace=str(tmp_path),
            fork_child=fake_fork,
        ))

        log_dir = tmp_path / "outputs" / "verification_logs"
        assert log_dir.exists()
        logs = list(log_dir.glob("holistic_*.md"))
        assert len(logs) >= 1

    def test_fail_creates_log_file(self, tmp_path: Path):
        verifier_output = (
            "### Check: broken\n**Command run:**\n  echo fail\n"
            "**Output observed:**\n  fail\n**Result: FAIL**\n\n"
            "VERDICT: FAIL"
        )

        async def fake_fork(spec, **kwargs):
            return FakeEnforcedChild(verifier_output)

        asyncio.run(run_holistic_verification(
            skill_definitions={"intake": "# Intake"},
            output_dirs={},
            workspace=str(tmp_path),
            fork_child=fake_fork,
        ))

        log_dir = tmp_path / "outputs" / "verification_logs"
        logs = list(log_dir.glob("holistic_*.md"))
        assert len(logs) >= 1
        content = logs[0].read_text()
        assert "FAIL" in content


# ---------------------------------------------------------------------------
# send_until_complete integration (holistic_check parameter)
# ---------------------------------------------------------------------------

class TestSendUntilCompleteHolisticParam:
    """Test that send_until_complete accepts and calls holistic_check."""

    def test_holistic_pass_no_extra_turns(self):
        """When holistic check passes, no feedback is sent."""
        from copilotcode_sdk.client import CopilotCodeSession

        send_count = [0]
        class FakeRawSession:
            workspace_path = "/tmp/test"
            session_id = "test-session"
            def on(self, callback): return lambda: None
            async def send_and_wait(self, prompt, **kw):
                send_count[0] += 1
                return {"role": "assistant", "content": "done"}
            async def get_messages(self):
                return []

        session = CopilotCodeSession(
            FakeRawSession(), MagicMock(),
        )

        async def holistic_pass():
            return VerificationResult(verdict="PASS", raw_output="ok", failed_checks=[])

        asyncio.run(session.send_until_complete(
            "do the work",
            holistic_check=holistic_pass,
        ))
        # Only 1 send: the initial prompt. No holistic feedback sent.
        assert send_count[0] == 1

    def test_holistic_fail_sends_feedback(self):
        """When holistic check fails, feedback is sent to orchestrator."""
        from copilotcode_sdk.client import CopilotCodeSession

        send_prompts = []
        class FakeRawSession:
            workspace_path = "/tmp/test"
            session_id = "test-session"
            def on(self, callback): return lambda: None
            async def send_and_wait(self, prompt, **kw):
                send_prompts.append(prompt)
                return {"role": "assistant", "content": "fixing..."}
            async def get_messages(self):
                return []

        session = CopilotCodeSession(
            FakeRawSession(), MagicMock(),
        )

        call_count = [0]
        async def holistic_fail_then_pass():
            call_count[0] += 1
            if call_count[0] == 1:
                return VerificationResult(
                    verdict="FAIL",
                    raw_output="...",
                    failed_checks=[
                        {"check": "cross-skill mismatch", "command": "diff", "observed": "bad"},
                    ],
                )
            return VerificationResult(verdict="PASS", raw_output="ok", failed_checks=[])

        asyncio.run(session.send_until_complete(
            "do the work",
            holistic_check=holistic_fail_then_pass,
        ))
        # 2 sends: initial prompt + holistic feedback
        assert len(send_prompts) == 2
        assert "HOLISTIC VERIFICATION FAILED" in send_prompts[1]
        assert "cross-skill mismatch" in send_prompts[1]

    def test_holistic_malfunction_no_feedback(self):
        """Malfunction means the verifier broke — don't send feedback."""
        from copilotcode_sdk.client import CopilotCodeSession

        send_count = [0]
        class FakeRawSession:
            workspace_path = "/tmp/test"
            session_id = "test-session"
            def on(self, callback): return lambda: None
            async def send_and_wait(self, prompt, **kw):
                send_count[0] += 1
                return {"role": "assistant", "content": "done"}
            async def get_messages(self):
                return []

        session = CopilotCodeSession(
            FakeRawSession(), MagicMock(),
        )

        async def holistic_malfunction():
            return VerificationResult(
                verdict="MALFUNCTION",
                raw_output="verifier crashed",
                failed_checks=[],
            )

        asyncio.run(session.send_until_complete(
            "do the work",
            holistic_check=holistic_malfunction,
        ))
        # Only 1 send: the initial prompt. No feedback on malfunction.
        assert send_count[0] == 1
