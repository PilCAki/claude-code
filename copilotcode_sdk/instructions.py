from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass(slots=True)
class InstructionBundle:
    """Assembled workspace instructions with provenance."""

    content: str = ""
    sources: list[Path] = field(default_factory=list)
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = hashlib.sha256(
                self.content.encode("utf-8"),
            ).hexdigest()[:16]


def load_workspace_instructions(
    root: Path,
    *,
    active_paths: Sequence[Path] | None = None,
    max_include_depth: int = 3,
) -> InstructionBundle:
    """Load layered workspace instructions from conventional file locations.

    Loading order (highest priority first):
    1. ``.claude/rules/*.md`` — local overrides
    2. ``CLAUDE.md``, ``AGENTS.md`` — project instructions
    3. ``.github/copilot-instructions.md`` — GitHub instructions

    Features: ``@include`` expansion, HTML comment stripping, MEMORY.md truncation.
    """
    root = root.resolve(strict=False)
    sections: list[str] = []
    sources: list[Path] = []

    # Layer 1: .claude/rules/*.md
    rules_dir = root / ".claude" / "rules"
    if rules_dir.is_dir():
        for rule_file in sorted(rules_dir.glob("*.md")):
            text = _load_file(rule_file, root, max_include_depth)
            if text:
                sections.append(
                    f"## Local Rules: `{rule_file.name}`\n{text}",
                )
                sources.append(rule_file)

    # Layer 2: CLAUDE.md, AGENTS.md
    for name in ("CLAUDE.md", "AGENTS.md"):
        candidate = root / name
        if candidate.is_file():
            text = _load_file(candidate, root, max_include_depth)
            if text:
                sections.append(
                    f"## Workspace Instructions: `{name}`\n{text}",
                )
                sources.append(candidate)

    # Layer 3: .github/copilot-instructions.md
    copilot_path = root / ".github" / "copilot-instructions.md"
    if copilot_path.is_file():
        text = _load_file(copilot_path, root, max_include_depth)
        if text:
            sections.append(
                f"## Workspace Instructions: `.github/copilot-instructions.md`\n{text}",
            )
            sources.append(copilot_path)

    content = "\n\n".join(sections)
    return InstructionBundle(content=content, sources=sources)


def _load_file(
    path: Path,
    root: Path,
    max_include_depth: int,
    *,
    _current_depth: int = 0,
) -> str:
    """Read a file, expand @include directives, strip HTML comments."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""

    # Truncate MEMORY.md at 200 lines
    if path.name == "MEMORY.md":
        lines = text.splitlines()
        if len(lines) > 200:
            text = "\n".join(lines[:200])

    # Expand @include directives
    if _current_depth < max_include_depth:
        text = _expand_includes(text, root, max_include_depth, _current_depth)

    # Strip HTML comments
    text = _strip_html_comments(text)

    # Truncate to 6000 chars per file
    text = text[:6_000].rstrip()

    return text


def _expand_includes(
    text: str,
    root: Path,
    max_include_depth: int,
    current_depth: int,
) -> str:
    """Replace @include directives with file contents."""
    lines: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^@include\s+(.+)$", line.strip())
        if match:
            include_path = root / match.group(1).strip()
            if include_path.is_file():
                included = _load_file(
                    include_path,
                    root,
                    max_include_depth,
                    _current_depth=current_depth + 1,
                )
                lines.append(included)
            else:
                lines.append(line)
        else:
            lines.append(line)
    return "\n".join(lines)


def _strip_html_comments(text: str) -> str:
    """Remove <!-- ... --> blocks from text."""
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL).strip()
