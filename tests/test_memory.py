from __future__ import annotations

from pathlib import Path

import pytest

from copilotcode_sdk.brand import BrandSpec
from copilotcode_sdk.memory import (
    ENTRYPOINT_NAME,
    MAX_ENTRYPOINT_BYTES,
    MAX_ENTRYPOINT_LINES,
    MAX_SESSION_MEMORY_ENTRIES,
    SESSION_MEMORY_NAME,
    MemoryStore,
    parse_frontmatter_document,
    sanitize_project_key,
    serialize_frontmatter,
    slugify,
    truncate_entrypoint_content,
)


def test_sanitize_project_key_is_stable(tmp_path: Path) -> None:
    key_a = sanitize_project_key(tmp_path)
    key_b = sanitize_project_key(tmp_path)

    assert key_a == key_b
    assert len(key_a.split("-")[-1]) == 10


def test_memory_store_upsert_reindex_and_select_relevant(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    store.upsert_memory(
        title="Payments API",
        description="Authentication and retry rules for payment requests.",
        memory_type="project",
        content="# Payments\nUse idempotency keys for retried charges.",
    )
    store.upsert_memory(
        title="Testing Preference",
        description="Prefer pytest -q for local verification.",
        memory_type="user",
        content="# Tests\nRun pytest -q before finishing.",
    )

    index_text = store.index_path.read_text(encoding="utf-8")
    relevant = store.select_relevant("payment retry auth", limit=2)

    assert ENTRYPOINT_NAME == store.index_path.name
    assert "Payments API" in index_text
    assert relevant
    assert relevant[0].name == "Payments API"


def test_memory_store_delete_updates_index(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    path = store.upsert_memory(
        title="Delete Me",
        description="Temporary durable memory for deletion test.",
        memory_type="reference",
        content="This record should be removed.",
    )

    store.delete_memory(path)

    assert not path.exists()
    assert "Delete Me" not in store.index_path.read_text(encoding="utf-8")


def test_truncate_entrypoint_content_warns_when_limits_are_exceeded() -> None:
    overlong_lines = "\n".join(
        f"- [Item {index}](item-{index}.md) - {'x' * 160}"
        for index in range(MAX_ENTRYPOINT_LINES + 5)
    )
    overlong_bytes = overlong_lines + ("\n" + ("y" * (MAX_ENTRYPOINT_BYTES + 10)))

    truncated = truncate_entrypoint_content(overlong_bytes)

    assert truncated.was_line_truncated is True
    assert truncated.was_byte_truncated is True
    assert "WARNING" in truncated.content


def test_memory_store_uses_brand_default_root_when_memory_root_is_omitted(
    tmp_path: Path,
) -> None:
    brand = BrandSpec(
        public_name="FutureName",
        slug="futurename",
        package_name="futurename_sdk",
        distribution_name="futurename-sdk",
        cli_name="futurename",
        app_dirname=".futurename",
    )

    store = MemoryStore(tmp_path, brand=brand)

    assert ".futurename" in str(store.memory_root)


def test_slugify_and_frontmatter_round_trip() -> None:
    assert slugify("  Hello, World!  ") == "hello-world"
    assert slugify("###") == "memory"

    raw = serialize_frontmatter(
        {
            "name": "Launch Token",
            "description": "Useful context",
            "type": "reference",
            "ignored": "value",
        },
        "Body text",
    )
    metadata, body = parse_frontmatter_document(raw)

    assert metadata == {
        "name": "Launch Token",
        "description": "Useful context",
        "type": "reference",
    }
    assert body == "Body text"


def test_parse_frontmatter_returns_original_text_when_missing_closing_delimiter() -> None:
    raw = "---\nname: Missing End\nBody"

    metadata, body = parse_frontmatter_document(raw)

    assert metadata == {}
    assert body == raw


def test_memory_store_relevance_ranking_is_stable_for_exact_match(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    store.upsert_memory(
        title="Primary Token",
        description="Contains the alpha token.",
        memory_type="reference",
        content="# Token\nalpha-token-123",
    )
    store.upsert_memory(
        title="Secondary Note",
        description="Contains unrelated context.",
        memory_type="reference",
        content="# Note\nbeta-token-999",
    )

    relevant = store.select_relevant("alpha-token-123", limit=2)

    assert [record.name for record in relevant][:1] == ["Primary Token"]


def test_memory_store_delete_accepts_slug_without_extension(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, tmp_path / ".mem")
    store.upsert_memory(
        title="Temporary Record",
        description="Will be removed by slug.",
        memory_type="reference",
        content="temporary",
    )

    store.delete_memory("temporary-record")

    assert "Temporary Record" not in store.index_path.read_text(encoding="utf-8")


def test_memory_property_style_slugify_and_sanitize(tmp_path: Path) -> None:
    hypothesis = pytest.importorskip("hypothesis")
    strategies = pytest.importorskip("hypothesis.strategies")

    @hypothesis.given(strategies.text(min_size=1, max_size=40))
    def run_slugify(value: str) -> None:
        slug = slugify(value)
        assert slug
        assert slug == slug.lower()
        assert "." in slug or "_" in slug or "-" in slug or slug.isalnum()

    @hypothesis.given(strategies.text(min_size=1, max_size=40))
    def run_sanitize(value: str) -> None:
        path = tmp_path / value
        key = sanitize_project_key(path)
        assert key
        assert len(key.split("-")[-1]) == 10

    run_slugify()
    run_sanitize()


# ---------------------------------------------------------------------------
# Wave 2: Session-scoped memory tests
# ---------------------------------------------------------------------------


class TestSessionMemory:
    def test_session_memory_path(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path, tmp_path / ".mem")

        assert store.session_memory_path.name == SESSION_MEMORY_NAME

    def test_append_creates_file_and_returns_path(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path, tmp_path / ".mem")

        path = store.append_session_memory("# First\nLearned X.")

        assert path.exists()
        assert "Learned X." in path.read_text(encoding="utf-8")

    def test_append_multiple_entries_separated_by_delimiter(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path, tmp_path / ".mem")
        store.append_session_memory("# Entry 1\nFirst learning")
        store.append_session_memory("# Entry 2\nSecond learning")

        content = store.read_session_memory()

        assert "First learning" in content
        assert "Second learning" in content
        assert "\n---\n" in content

    def test_read_returns_empty_when_no_session_file(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path, tmp_path / ".mem")

        assert store.read_session_memory() == ""

    def test_clear_removes_session_file(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path, tmp_path / ".mem")
        store.append_session_memory("# Something\nData")

        store.clear_session_memory()

        assert not store.session_memory_path.exists()
        assert store.read_session_memory() == ""

    def test_clear_is_noop_when_no_file(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path, tmp_path / ".mem")
        store.clear_session_memory()  # should not raise

    def test_append_bounds_to_max_entries(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path, tmp_path / ".mem")

        for i in range(MAX_SESSION_MEMORY_ENTRIES + 5):
            store.append_session_memory(f"# Entry {i}\nContent {i}")

        content = store.read_session_memory()
        entries = [e for e in content.strip().split("\n---\n") if e.strip()]

        assert len(entries) == MAX_SESSION_MEMORY_ENTRIES
        # Oldest entries should be dropped — the last entry should be present
        assert f"Content {MAX_SESSION_MEMORY_ENTRIES + 4}" in content
        # The very first entry should have been evicted
        assert "Content 0" not in content

    def test_promote_creates_durable_memories(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path, tmp_path / ".mem")
        store.append_session_memory("# Schema Discovery\nTable has 10 columns")
        store.append_session_memory("# Data Quality\nNull rate is 5%")

        created = store.promote_session_memory()

        assert len(created) == 2
        assert all(p.exists() for p in created)
        # Session file should be cleared after promotion
        assert store.read_session_memory() == ""
        # Durable memory index should contain the promoted entries
        index_text = store.index_path.read_text(encoding="utf-8")
        assert "Schema Discovery" in index_text
        assert "Data Quality" in index_text

    def test_promote_empty_session_returns_empty(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path, tmp_path / ".mem")

        created = store.promote_session_memory()

        assert created == []

    def test_promote_uses_first_line_as_title(self, tmp_path: Path) -> None:
        store = MemoryStore(tmp_path, tmp_path / ".mem")
        store.append_session_memory("# My Title\nBody content here")

        created = store.promote_session_memory()

        assert len(created) == 1
        content = created[0].read_text(encoding="utf-8")
        assert "name: My Title" in content
