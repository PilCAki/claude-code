from __future__ import annotations


def build_compaction_prompt() -> str:
    return (
        "The conversation context is being compacted. Produce a structured summary "
        "that preserves everything a fresh context window would need to continue "
        "this work seamlessly. Cover these points:\n\n"
        "1. **Primary request and intent** — what the user originally asked for and why\n"
        "2. **Key findings and discoveries** — important facts, data patterns, or decisions made\n"
        "3. **Current state of progress** — what has been completed, what files were created or modified\n"
        "4. **Remaining work** — what still needs to be done, in priority order\n"
        "5. **Active assumptions and caveats** — interpretive decisions, data quality issues, open questions\n"
        "6. **Critical context** — anything else that would be lost without this summary "
        "(column mappings, metric values, error patterns, user preferences expressed during the session)\n\n"
        "Be specific and concrete. Include file paths, metric values, and exact names. "
        "A vague summary is worse than no summary."
    )


def build_handoff_context(
    *,
    compaction_summary: str,
    skill_catalog_text: str = "",
    instruction_content: str = "",
    memory_index: str = "",
) -> str:
    sections = [
        "## Session Continuation\n",
        "The previous context was compacted. Below is the preserved context.\n",
        "### Compaction Summary\n",
        compaction_summary,
    ]

    if skill_catalog_text:
        sections.append("\n### Skill Catalog\n")
        sections.append(skill_catalog_text)

    if instruction_content:
        sections.append("\n### Workspace Instructions\n")
        sections.append(instruction_content)

    if memory_index:
        sections.append("\n### Memory Index\n")
        sections.append(memory_index)

    return "\n".join(sections)
