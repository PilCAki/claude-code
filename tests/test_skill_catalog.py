from __future__ import annotations

from pathlib import Path

from copilotcode_sdk.skill_assets import parse_skill_frontmatter, build_skill_catalog, SkillTracker


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


# ---------------------------------------------------------------------------
# Wave 2: SkillTracker tests
# ---------------------------------------------------------------------------


def _sample_skill_map() -> dict[str, dict[str, str]]:
    """A minimal skill map with a dependency chain: intake → analysis → report."""
    return {
        "intake": {"name": "intake", "type": "data-intake", "requires": "none"},
        "analysis": {"name": "analysis", "type": "business-analysis", "requires": "data-intake"},
        "report": {"name": "report", "type": "report-generation", "requires": "business-analysis"},
    }


class TestSkillTracker:
    def test_initial_state(self) -> None:
        tracker = SkillTracker()

        assert tracker.current_turn == 0
        assert tracker.invocation_count("x") == 0
        assert tracker.completion_count("x") == 0
        assert tracker.last_used_turn("x") == -1

    def test_record_invocation(self) -> None:
        tracker = SkillTracker()
        tracker.record_invocation("intake")

        assert tracker.invocation_count("intake") == 1
        assert tracker.last_used_turn("intake") == 0

    def test_record_completion(self) -> None:
        tracker = SkillTracker()
        tracker.record_completion("intake")

        assert tracker.completion_count("intake") == 1

    def test_advance_turn(self) -> None:
        tracker = SkillTracker()
        tracker.advance_turn()
        tracker.advance_turn()

        assert tracker.current_turn == 2

    def test_score_no_prereqs_not_completed_never_invoked(self) -> None:
        """A fresh skill with no prereqs should get the maximum positive score."""
        tracker = SkillTracker()
        skill_map = _sample_skill_map()

        score = tracker.score("intake", skill_map, completed_skills=set())

        # +10 (no prereqs) + 5 (not completed) + 3 (never invoked) = 18
        assert score == 18.0

    def test_score_prereqs_not_met(self) -> None:
        """A skill whose prereqs are not met should be strongly deprioritized."""
        tracker = SkillTracker()
        skill_map = _sample_skill_map()

        score = tracker.score("analysis", skill_map, completed_skills=set())

        # -20 (prereqs not met) + 5 (not completed) + 3 (never invoked) = -12
        assert score == -12.0

    def test_score_prereqs_met(self) -> None:
        """A skill whose prereqs are met should get the prereqs bonus."""
        tracker = SkillTracker()
        skill_map = _sample_skill_map()

        score = tracker.score("analysis", skill_map, completed_skills={"intake"})

        # +10 (prereqs met) + 5 (not completed) + 3 (never invoked) = 18
        assert score == 18.0

    def test_score_completed_skill_loses_bonus(self) -> None:
        tracker = SkillTracker()
        skill_map = _sample_skill_map()

        score = tracker.score("intake", skill_map, completed_skills={"intake"})

        # +10 (no prereqs) + 0 (completed) + 3 (never invoked) = 13
        assert score == 13.0

    def test_score_recency_penalty_very_recent(self) -> None:
        tracker = SkillTracker()
        tracker.record_invocation("intake")
        tracker.advance_turn()
        tracker.advance_turn()  # 2 turns ago

        skill_map = _sample_skill_map()
        score = tracker.score("intake", skill_map, completed_skills=set())

        # +10 + 5 + 0 (was invoked) - 3 (very recent) = 12
        assert score == 12.0

    def test_score_recency_penalty_somewhat_recent(self) -> None:
        tracker = SkillTracker()
        tracker.record_invocation("intake")
        for _ in range(8):
            tracker.advance_turn()  # 8 turns ago

        skill_map = _sample_skill_map()
        score = tracker.score("intake", skill_map, completed_skills=set())

        # +10 + 5 + 0 (was invoked) - 1 (somewhat recent) = 14
        assert score == 14.0

    def test_score_no_recency_penalty_after_15_turns(self) -> None:
        tracker = SkillTracker()
        tracker.record_invocation("intake")
        for _ in range(20):
            tracker.advance_turn()

        skill_map = _sample_skill_map()
        score = tracker.score("intake", skill_map, completed_skills=set())

        # +10 + 5 + 0 (was invoked) + 0 (no recency penalty) = 15
        assert score == 15.0

    def test_rank_returns_sorted_by_score(self) -> None:
        tracker = SkillTracker()
        skill_map = _sample_skill_map()

        ranked = tracker.rank(skill_map, completed_skills=set())

        # intake (no prereqs) should be first, analysis/report (unmet prereqs) lower
        assert ranked[0][0] == "intake"
        assert ranked[0][1] > ranked[-1][1]

    def test_rank_with_partial_completion(self) -> None:
        tracker = SkillTracker()
        skill_map = _sample_skill_map()

        ranked = tracker.rank(skill_map, completed_skills={"intake"})

        names = [name for name, _ in ranked]
        # analysis should now rank high since its prereq is met
        assert "analysis" in names[:2]

    def test_top_surfaceable_excludes_completed(self) -> None:
        tracker = SkillTracker()
        skill_map = _sample_skill_map()

        top = tracker.top_surfaceable(skill_map, completed_skills={"intake"})

        assert "intake" not in top

    def test_top_surfaceable_respects_limit(self) -> None:
        tracker = SkillTracker()
        skill_map = _sample_skill_map()

        top = tracker.top_surfaceable(skill_map, completed_skills=set(), limit=1)

        assert len(top) <= 1

    def test_top_surfaceable_respects_min_score(self) -> None:
        tracker = SkillTracker()
        skill_map = _sample_skill_map()

        # With min_score=100, nothing should qualify
        top = tracker.top_surfaceable(
            skill_map, completed_skills=set(), min_score=100.0,
        )

        assert top == []

    def test_multiple_invocations_counted(self) -> None:
        tracker = SkillTracker()
        tracker.record_invocation("intake")
        tracker.record_invocation("intake")
        tracker.record_invocation("intake")

        assert tracker.invocation_count("intake") == 3
