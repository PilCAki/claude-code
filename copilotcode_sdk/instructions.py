from __future__ import annotations

import fnmatch
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence


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

    @property
    def loaded_paths(self) -> list[str]:
        """Convenience accessor returning source paths as strings."""
        return [str(s) for s in self.sources]


def load_workspace_instructions(
    root: Path,
    *,
    active_paths: Sequence[Path] | None = None,
    max_include_depth: int = 3,
    max_traversal_depth: int = 5,
    on_loaded: Callable[[InstructionBundle], None] | None = None,
    user_config_dir: Path | None = None,
) -> InstructionBundle:
    """Load layered workspace instructions from conventional file locations.

    Loading order (highest priority first):
    0. ``~/.copilotcode/CLAUDE.md`` or ``~/.claude/CLAUDE.md`` — user-level instructions
    1. ``.claude/rules/*.md`` — local overrides (with optional path-conditional frontmatter)
    1.5. ``.claude/CLAUDE.md`` — project dot-directory instructions
    2. ``CLAUDE.md``, ``AGENTS.md`` — project instructions
    3. ``.github/copilot-instructions.md`` — GitHub instructions
    3.5. ``CLAUDE.local.md`` — git-ignored local overrides
    4. Parent directory ``CLAUDE.md`` files (upward traversal, stops at ``.git`` boundary)

    Features: ``@include`` expansion (with external-include warnings), HTML comment
    stripping, MEMORY.md truncation, path-conditional rules, upward traversal.
    """
    root = root.resolve(strict=False)
    sections: list[str] = []
    sources: list[Path] = []

    # Layer 0: User-level instructions (~/.copilotcode/CLAUDE.md or ~/.claude/CLAUDE.md)
    if user_config_dir is not None:
        user_claude = Path(user_config_dir) / "CLAUDE.md"
        if user_claude.is_file():
            text = _load_file(user_claude, user_claude.parent, max_include_depth)
            if text:
                sections.append(f"## User Instructions: `{user_claude}`\n{text}")
                sources.append(user_claude)
    else:
        for dirname in (".copilotcode", ".claude"):
            user_claude = Path.home() / dirname / "CLAUDE.md"
            if user_claude.is_file():
                text = _load_file(user_claude, user_claude.parent, max_include_depth)
                if text:
                    sections.append(
                        f"## User Instructions: `~/{dirname}/CLAUDE.md`\n{text}",
                    )
                    sources.append(user_claude)
                break  # first match wins

    # Layer 1: .claude/rules/*.md (with path-conditional filtering)
    rules_dir = root / ".claude" / "rules"
    if rules_dir.is_dir():
        for rule_file in sorted(rules_dir.glob("*.md")):
            text_raw = _load_file_raw(rule_file)
            if not text_raw:
                continue
            frontmatter, body = _split_frontmatter(text_raw)
            # Path-conditional filtering
            if frontmatter and active_paths is not None:
                globs = _extract_globs(frontmatter)
                if globs and not _any_path_matches(active_paths, globs, root):
                    continue
            text = _process_file_body(body, rule_file, root, max_include_depth)
            if text:
                sections.append(
                    f"## Local Rules: `{rule_file.name}`\n{text}",
                )
                sources.append(rule_file)

    # Layer 1.5: .claude/CLAUDE.md (project dot-directory instructions)
    dot_claude_md = root / ".claude" / "CLAUDE.md"
    if dot_claude_md.is_file():
        text = _load_file(dot_claude_md, root, max_include_depth)
        if text:
            sections.append(
                f"## Project Instructions: `.claude/CLAUDE.md`\n{text}",
            )
            sources.append(dot_claude_md)

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

    # Layer 3.5: CLAUDE.local.md (git-ignored local overrides)
    local_md = root / "CLAUDE.local.md"
    if local_md.is_file():
        text = _load_file(local_md, root, max_include_depth)
        if text:
            sections.append(
                f"## Local Overrides: `CLAUDE.local.md`\n{text}",
            )
            sources.append(local_md)

    # Layer 4: Upward directory traversal for parent CLAUDE.md files
    parent_sections = _load_parent_instructions(root, max_include_depth, max_traversal_depth)
    for parent_path, text in parent_sections:
        sections.append(text)
        sources.append(parent_path)

    content = "\n\n".join(sections)
    bundle = InstructionBundle(content=content, sources=sources)
    if on_loaded is not None:
        on_loaded(bundle)
    return bundle


def _load_parent_instructions(
    root: Path,
    max_include_depth: int,
    max_traversal_depth: int,
) -> list[tuple[Path, str]]:
    """Walk parent directories for CLAUDE.md files, stopping at .git boundary."""
    results: list[tuple[Path, str]] = []
    current = root.parent
    for _ in range(max_traversal_depth):
        if current == current.parent:
            break  # filesystem root
        candidate = current / "CLAUDE.md"
        if candidate.is_file():
            text = _load_file(candidate, current, max_include_depth)
            if text:
                try:
                    rel = candidate.relative_to(root)
                except ValueError:
                    rel = candidate
                results.append((
                    candidate,
                    f"## Parent Instructions: `{rel}`\n{text}",
                ))
        # Stop at .git boundary
        if (current / ".git").exists():
            break
        current = current.parent
    return results


def _load_file(
    path: Path,
    root: Path,
    max_include_depth: int,
    *,
    _current_depth: int = 0,
    _visited: set[Path] | None = None,
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
        if _visited is None:
            _visited = set()
        # Add this file to visited so self-includes are caught
        resolved = path.resolve(strict=False)
        _visited.add(resolved)
        text = _expand_includes(text, root, max_include_depth, _current_depth, _visited=_visited)

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
    *,
    _visited: set[Path] | None = None,
) -> str:
    """Replace @include directives with file contents.

    Includes that resolve outside the project root are replaced with a warning
    comment. Circular includes (A->B->A) are detected via *_visited* and skipped.
    """
    if _visited is None:
        _visited = set()

    lines: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^@include\s+(.+)$", line.strip())
        if match:
            raw_path = match.group(1).strip()
            include_path = (root / raw_path).resolve(strict=False)
            # External-include warning
            try:
                include_path.relative_to(root)
            except ValueError:
                lines.append(
                    f"<!-- WARNING: external include skipped: {raw_path} -->"
                )
                continue
            # Circular include protection
            if include_path in _visited:
                lines.append(
                    f"<!-- circular include skipped: {raw_path} -->"
                )
                continue
            if include_path.is_file():
                _visited.add(include_path)
                included = _load_file(
                    include_path,
                    root,
                    max_include_depth,
                    _current_depth=current_depth + 1,
                    _visited=_visited,
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


# ---------------------------------------------------------------------------
# Path-conditional rule helpers
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split YAML-ish frontmatter from body text.

    Returns (frontmatter_dict, body). If no frontmatter, returns ({}, text).
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    fm_text = match.group(1)
    body = text[match.end():]
    fm: dict[str, str] = {}
    for line in fm_text.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fm[key.strip().lower()] = value.strip().strip('"').strip("'")
    return fm, body


def _extract_globs(frontmatter: dict[str, str]) -> list[str]:
    """Extract glob patterns from frontmatter ``applies_to`` or ``globs`` fields."""
    patterns: list[str] = []
    for key in ("applies_to", "globs"):
        raw = frontmatter.get(key, "")
        if not raw:
            continue
        # Handle comma-separated or bracket-list syntax
        raw = raw.strip("[]")
        for part in raw.split(","):
            part = part.strip().strip('"').strip("'")
            if part:
                patterns.append(part)
    return patterns


def _any_path_matches(
    active_paths: Sequence[Path],
    globs: list[str],
    root: Path,
) -> bool:
    """Check if any active path matches any of the glob patterns."""
    for path in active_paths:
        try:
            rel = str(path.resolve(strict=False).relative_to(root))
        except ValueError:
            rel = str(path)
        # Normalize separators for matching
        rel = rel.replace("\\", "/")
        for pattern in globs:
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(path.name, pattern):
                return True
    return False


def _load_file_raw(path: Path) -> str:
    """Read a file's raw text, returning empty string on failure."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _process_file_body(
    body: str,
    path: Path,
    root: Path,
    max_include_depth: int,
    *,
    _current_depth: int = 0,
) -> str:
    """Process file body (after frontmatter removal): includes, comments, truncation."""
    # Truncate MEMORY.md at 200 lines
    if path.name == "MEMORY.md":
        lines = body.splitlines()
        if len(lines) > 200:
            body = "\n".join(lines[:200])

    # Expand @include directives
    if _current_depth < max_include_depth:
        body = _expand_includes(body, root, max_include_depth, _current_depth)

    # Strip HTML comments
    body = _strip_html_comments(body)

    # Truncate to 6000 chars per file
    body = body[:6_000].rstrip()

    return body
