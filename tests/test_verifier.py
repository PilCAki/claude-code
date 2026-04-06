"""Tests for the verification gate module."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

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
