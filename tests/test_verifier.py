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
