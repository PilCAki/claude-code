from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
import re
import subprocess
from typing import Literal

from .brand import BrandSpec, DEFAULT_BRAND

ENTRYPOINT_NAME = "MEMORY.md"
SESSION_MEMORY_NAME = "session_memory.md"
MAX_ENTRYPOINT_LINES = 200
MAX_ENTRYPOINT_BYTES = 25_000
MAX_RELEVANT_MEMORIES = 5
MAX_SESSION_MEMORY_ENTRIES = 20

MemoryType = Literal["user", "feedback", "project", "reference"]
VALID_MEMORY_TYPES: set[str] = {"user", "feedback", "project", "reference"}
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "please",
    "the",
    "this",
    "to",
    "we",
    "with",
}


@dataclass(frozen=True, slots=True)
class EntrypointTruncation:
    content: str
    line_count: int
    byte_count: int
    was_line_truncated: bool
    was_byte_truncated: bool


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    path: Path
    name: str
    description: str
    memory_type: MemoryType | None
    content: str
    headings: tuple[str, ...]
    mtime_ms: float


def discover_project_root(start: str | Path) -> Path:
    base = Path(start).expanduser().resolve(strict=False)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(base),
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except OSError:
        return base

    if result.returncode == 0 and result.stdout.strip():
        return Path(result.stdout.strip()).expanduser().resolve(strict=False)
    return base


def sanitize_project_key(path: str | Path) -> str:
    path_str = str(Path(path).expanduser().resolve(strict=False)).lower()
    sanitized = re.sub(r"[^a-z0-9._-]+", "-", path_str).strip("-")
    sanitized = sanitized[:64] or "project"
    digest = sha1(path_str.encode("utf-8")).hexdigest()[:10]
    return f"{sanitized}-{digest}"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", value.lower()).strip("-")
    return slug or "memory"


def truncate_entrypoint_content(raw: str) -> EntrypointTruncation:
    trimmed = raw.strip()
    lines = trimmed.splitlines() if trimmed else []
    line_count = len(lines)
    byte_count = len(trimmed.encode("utf-8"))
    was_line_truncated = line_count > MAX_ENTRYPOINT_LINES
    was_byte_truncated = byte_count > MAX_ENTRYPOINT_BYTES

    if not was_line_truncated and not was_byte_truncated:
        return EntrypointTruncation(
            content=trimmed,
            line_count=line_count,
            byte_count=byte_count,
            was_line_truncated=False,
            was_byte_truncated=False,
        )

    content = "\n".join(lines[:MAX_ENTRYPOINT_LINES]) if was_line_truncated else trimmed
    encoded = content.encode("utf-8")
    if len(encoded) > MAX_ENTRYPOINT_BYTES:
        encoded = encoded[:MAX_ENTRYPOINT_BYTES]
        content = encoded.decode("utf-8", errors="ignore")
        if "\n" in content:
            content = content.rsplit("\n", 1)[0]

    reason_parts: list[str] = []
    if was_line_truncated:
        reason_parts.append(
            f"{line_count} lines (limit: {MAX_ENTRYPOINT_LINES})",
        )
    if was_byte_truncated:
        reason_parts.append(
            f"{byte_count} bytes (limit: {MAX_ENTRYPOINT_BYTES})",
        )
    warning = (
        f"> WARNING: {ENTRYPOINT_NAME} exceeded "
        + " and ".join(reason_parts)
        + ". Only part of the index was loaded."
    )
    full_content = content.strip()
    full_content = f"{full_content}\n\n{warning}" if full_content else warning
    return EntrypointTruncation(
        content=full_content,
        line_count=line_count,
        byte_count=byte_count,
        was_line_truncated=was_line_truncated,
        was_byte_truncated=was_byte_truncated,
    )


def parse_frontmatter_document(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    metadata: dict[str, str] = {}
    end_index = None
    for index, raw_line in enumerate(lines[1:], start=1):
        line = raw_line.rstrip()
        if line.strip() == "---":
            end_index = index
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"').strip("'")

    if end_index is None:
        return {}, text

    body = "\n".join(lines[end_index + 1 :]).lstrip()
    return metadata, body


def serialize_frontmatter(metadata: dict[str, str], body: str) -> str:
    frontmatter_lines = ["---"]
    for key in ("name", "description", "type"):
        value = metadata.get(key, "").strip()
        if value:
            frontmatter_lines.append(f"{key}: {value}")
    frontmatter_lines.append("---")
    frontmatter = "\n".join(frontmatter_lines)
    clean_body = body.strip() + "\n"
    return f"{frontmatter}\n\n{clean_body}"


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9._-]+", text.lower())
        if token not in STOP_WORDS
    }


def _extract_headings(body: str) -> tuple[str, ...]:
    headings = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            headings.append(stripped.lstrip("#").strip())
    return tuple(headings)


class MemoryStore:
    """Manage Claude Code-style durable memory on disk."""

    def __init__(
        self,
        working_directory: str | Path,
        memory_root: str | Path | None = None,
        brand: BrandSpec = DEFAULT_BRAND,
    ) -> None:
        self.working_directory = Path(working_directory).expanduser().resolve(strict=False)
        self.project_root = discover_project_root(self.working_directory)
        base_root = Path(memory_root or brand.memory_home()).expanduser().resolve(strict=False)
        self.memory_root = base_root
        self.project_key = sanitize_project_key(self.project_root)
        self.memory_dir = (
            self.memory_root / "projects" / self.project_key / "memory"
        )
        self.index_path = self.memory_dir / ENTRYPOINT_NAME

    def ensure(self) -> Path:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self.index_path.write_text("", encoding="utf-8")
        return self.memory_dir

    def load_index(self) -> EntrypointTruncation:
        self.ensure()
        raw = self.index_path.read_text(encoding="utf-8")
        return truncate_entrypoint_content(raw)

    def build_index_context(self) -> str:
        truncated = self.load_index()
        if not truncated.content:
            return (
                f"## Durable Memory Index\n"
                f"Memory directory: `{self.memory_dir}`\n"
                "The durable memory index is currently empty."
            )
        return (
            f"## Durable Memory Index\n"
            f"Memory directory: `{self.memory_dir}`\n"
            f"{truncated.content}"
        )

    def list_records(self) -> list[MemoryRecord]:
        self.ensure()
        records: list[MemoryRecord] = []
        for path in sorted(self.memory_dir.rglob("*.md")):
            if path.name == ENTRYPOINT_NAME:
                continue
            metadata, body = parse_frontmatter_document(
                path.read_text(encoding="utf-8"),
            )
            stat = path.stat()
            memory_type = metadata.get("type")
            typed_value: MemoryType | None
            if memory_type in VALID_MEMORY_TYPES:
                typed_value = memory_type  # type: ignore[assignment]
            else:
                typed_value = None
            name = metadata.get("name") or path.stem.replace("-", " ").title()
            description = metadata.get("description", "").strip()
            records.append(
                MemoryRecord(
                    path=path,
                    name=name.strip(),
                    description=description,
                    memory_type=typed_value,
                    content=body.strip(),
                    headings=_extract_headings(body),
                    mtime_ms=stat.st_mtime * 1000,
                ),
            )
        records.sort(key=lambda record: (-record.mtime_ms, record.path.name))
        return records

    def select_relevant(
        self,
        query: str,
        *,
        limit: int = MAX_RELEVANT_MEMORIES,
    ) -> list[MemoryRecord]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored: list[tuple[int, float, str, MemoryRecord]] = []
        for record in self.list_records():
            score = self._score_record(query, query_tokens, record)
            if score <= 0:
                continue
            scored.append((score, record.mtime_ms, record.path.name, record))

        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [record for _, _, _, record in scored[:limit]]

    def build_relevant_context(
        self,
        query: str,
        *,
        limit: int = MAX_RELEVANT_MEMORIES,
    ) -> str:
        records = self.select_relevant(query, limit=limit)
        if not records:
            return ""

        chunks = ["## Relevant Durable Memory"]
        for record in records:
            type_label = record.memory_type or "untyped"
            preview = record.content.strip()
            if len(preview) > 900:
                preview = preview[:900].rstrip() + "..."
            chunks.extend(
                [
                    f"### {record.name} ({type_label})",
                    f"Path: `{record.path}`",
                    f"Description: {record.description or 'None'}",
                    preview or "No additional body content.",
                ],
            )
        return "\n".join(chunks)

    def upsert_memory(
        self,
        *,
        title: str,
        description: str,
        memory_type: MemoryType,
        content: str,
        slug: str | None = None,
    ) -> Path:
        self.ensure()
        filename = f"{slugify(slug or title)}.md"
        path = self.memory_dir / filename
        document = serialize_frontmatter(
            {
                "name": title.strip(),
                "description": description.strip(),
                "type": memory_type,
            },
            content,
        )
        path.write_text(document, encoding="utf-8")
        self.reindex()
        return path

    def delete_memory(self, slug_or_path: str | Path) -> None:
        self.ensure()
        path = self._resolve_memory_path(slug_or_path)
        if path.exists():
            path.unlink()
        self.reindex()

    def reindex(self) -> str:
        entries: list[str] = []
        for record in sorted(
            self.list_records(),
            key=lambda item: (item.name.lower(), item.path.name.lower()),
        ):
            summary = record.description or "durable memory"
            summary = re.sub(r"\s+", " ", summary).strip()
            if len(summary) > 120:
                summary = summary[:117].rstrip() + "..."
            entries.append(f"- [{record.name}]({record.path.name}) - {summary}")

        index_content = "\n".join(entries)
        self.index_path.write_text(index_content + ("\n" if entries else ""), encoding="utf-8")
        return index_content

    # -- Session-scoped memory --

    @property
    def session_memory_path(self) -> Path:
        """Path to the session-scoped memory file."""
        return self.memory_dir / SESSION_MEMORY_NAME

    def append_session_memory(self, entry: str) -> Path:
        """Append a timestamped entry to the session memory file.

        Session memory captures within-session learnings that haven't been
        promoted to durable memory yet.  Bounded to
        :data:`MAX_SESSION_MEMORY_ENTRIES` entries.
        """
        self.ensure()
        path = self.session_memory_path
        existing = ""
        if path.exists():
            existing = path.read_text(encoding="utf-8")

        entries = [e for e in existing.strip().split("\n---\n") if e.strip()]
        entries.append(entry.strip())
        # Bound: keep only the most recent entries
        if len(entries) > MAX_SESSION_MEMORY_ENTRIES:
            entries = entries[-MAX_SESSION_MEMORY_ENTRIES:]

        path.write_text("\n---\n".join(entries) + "\n", encoding="utf-8")
        return path

    def write_session_memory(self, content: str) -> Path:
        """Overwrite the session memory file with complete updated content.

        Unlike :meth:`append_session_memory`, this replaces the entire file.
        Used by the maintenance pass which produces a complete updated notes
        document each time.
        """
        self.ensure()
        path = self.session_memory_path
        path.write_text(content.strip() + "\n", encoding="utf-8")
        return path

    def read_session_memory(self) -> str:
        """Read the current session memory contents."""
        path = self.session_memory_path
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def clear_session_memory(self) -> None:
        """Remove the session memory file (e.g. at session end after promotion)."""
        path = self.session_memory_path
        if path.exists():
            path.unlink()

    def promote_session_memory(self) -> list[Path]:
        """Promote session memory entries to durable project memories.

        Parses the session memory file for structured entries and creates
        proper durable memories.  Returns paths of created memory files.
        Clears the session memory afterward.
        """
        content = self.read_session_memory()
        if not content.strip():
            return []

        entries = [e.strip() for e in content.split("\n---\n") if e.strip()]
        created: list[Path] = []
        for entry in entries:
            # Try to extract a title from the first line
            lines = entry.strip().splitlines()
            title = lines[0].lstrip("#").strip() if lines else "Session learning"
            body = "\n".join(lines[1:]).strip() if len(lines) > 1 else entry
            if not body:
                body = title

            path = self.upsert_memory(
                title=title,
                description=f"Promoted from session memory",
                memory_type="project",
                content=body,
                slug=f"session-{slugify(title)}",
            )
            created.append(path)

        self.clear_session_memory()
        return created

    def _resolve_memory_path(self, slug_or_path: str | Path) -> Path:
        candidate = Path(slug_or_path)
        if candidate.is_absolute():
            return candidate
        filename = candidate.name
        if candidate.suffix != ".md":
            filename = f"{slugify(filename)}.md"
        return self.memory_dir / filename

    @staticmethod
    def _score_record(
        query: str,
        query_tokens: set[str],
        record: MemoryRecord,
    ) -> int:
        lowered_query = query.lower()
        score = 0

        title_tokens = _tokenize(record.name)
        description_tokens = _tokenize(record.description)
        heading_tokens = _tokenize(" ".join(record.headings))
        stem_tokens = _tokenize(record.path.stem.replace("-", " "))
        body_tokens = _tokenize(record.content[:2_000])

        score += len(query_tokens & title_tokens) * 5
        score += len(query_tokens & description_tokens) * 4
        score += len(query_tokens & heading_tokens) * 3
        score += len(query_tokens & stem_tokens) * 2
        score += len(query_tokens & body_tokens)

        haystacks = (
            record.name.lower(),
            record.description.lower(),
            " ".join(record.headings).lower(),
            record.content[:2_000].lower(),
        )
        if any(lowered_query and lowered_query in haystack for haystack in haystacks):
            score += max(3, len(query_tokens))

        return score
