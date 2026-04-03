from __future__ import annotations

from pathlib import Path

import pytest

from copilotcode_sdk.brand import BrandSpec
from copilotcode_sdk.memory import (
    ENTRYPOINT_NAME,
    MAX_ENTRYPOINT_BYTES,
    MAX_ENTRYPOINT_LINES,
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
