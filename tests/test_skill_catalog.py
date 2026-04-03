from __future__ import annotations

from pathlib import Path

from copilotcode_sdk.skill_assets import parse_skill_frontmatter, build_skill_catalog


def _write_skill(skill_dir: Path, name: str, frontmatter: str) -> Path:
    d = skill_dir / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(frontmatter, encoding="utf-8")
    return d


def test_parse_skill_frontmatter_extracts_fields(tmp_path: Path) -> None:
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: my-skill\n"
        "description: Does a thing.\n"
        "type: data-intake\n"
        "inputs: .xlsx file\n"
        "outputs: outputs/intake/\n"
        "requires: none\n"
        "---\n"
        "\n# My Skill\nBody text.\n",
        encoding="utf-8",
    )

    result = parse_skill_frontmatter(skill_dir / "SKILL.md")

    assert result["name"] == "my-skill"
    assert result["description"] == "Does a thing."
    assert result["type"] == "data-intake"
    assert result["inputs"] == ".xlsx file"
    assert result["outputs"] == "outputs/intake/"
    assert result["requires"] == "none"


def test_parse_skill_frontmatter_handles_missing_fields(tmp_path: Path) -> None:
    skill_dir = tmp_path / "minimal"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: minimal\ndescription: Bare minimum.\n---\n\n# Minimal\n",
        encoding="utf-8",
    )

    result = parse_skill_frontmatter(skill_dir / "SKILL.md")

    assert result["name"] == "minimal"
    assert result.get("type") is None
    assert result.get("requires") is None


def test_build_skill_catalog_formats_table_and_deps(tmp_path: Path) -> None:
    _write_skill(tmp_path, "intake", (
        "---\nname: intake\ndescription: Ingest data.\n"
        "type: data-intake\nrequires: none\n---\n\n# Intake\n"
    ))
    _write_skill(tmp_path, "analysis", (
        "---\nname: analysis\ndescription: Analyze data.\n"
        "type: business-analysis\nrequires: data-intake\n---\n\n# Analysis\n"
    ))
    _write_skill(tmp_path, "report", (
        "---\nname: report\ndescription: Write report.\n"
        "type: report-generation\nrequires: business-analysis\n---\n\n# Report\n"
    ))

    catalog, skill_map = build_skill_catalog([str(tmp_path)])

    assert "## Available Skills" in catalog
    assert "intake" in catalog
    assert "analysis" in catalog
    assert "report" in catalog
    assert "intake" in catalog.split("analysis")[0]  # intake listed before analysis
    assert "data-intake" in catalog
    assert "Skill Dependencies" in catalog
    assert len(skill_map) == 3
    assert skill_map["intake"]["type"] == "data-intake"
    assert skill_map["analysis"]["requires"] == "data-intake"


def test_build_skill_catalog_filters_disabled_skills(tmp_path: Path) -> None:
    _write_skill(tmp_path, "keep", (
        "---\nname: keep\ndescription: Keep this.\ntype: keeper\nrequires: none\n---\n\n# Keep\n"
    ))
    _write_skill(tmp_path, "drop", (
        "---\nname: drop\ndescription: Drop this.\ntype: dropper\nrequires: none\n---\n\n# Drop\n"
    ))

    catalog, skill_map = build_skill_catalog(
        [str(tmp_path)], disabled_skills=["drop"],
    )

    assert "keep" in catalog
    assert "drop" not in catalog
    assert "drop" not in skill_map


def test_build_skill_catalog_returns_empty_for_no_skills(tmp_path: Path) -> None:
    catalog, skill_map = build_skill_catalog([str(tmp_path)])

    assert catalog == ""
    assert skill_map == {}
