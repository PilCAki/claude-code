---
name: remember
description: Review and maintain durable project or user memory in the CopilotCode memory store.
allowed-tools:
  - read
  - edit
  - search
when_to_use: Use when the user wants CopilotCode to remember, forget, or clean up durable context. Examples: "remember this preference", "forget that note", "organize project memory".
argument-hint: "[memory request or cleanup goal]"
---

# Remember

## Goal
Keep durable memory clean, useful, and limited to information that will matter in future sessions.

## Steps

### 1. Read the memory index
Inspect `MEMORY.md` first so you understand what is already stored and can avoid duplicates.

Success criteria:
- You know whether an existing memory should be updated instead of creating a new file.

### 2. Classify the information
Decide whether the request belongs in one of these durable buckets:
- `user`: long-lived user preferences or collaboration patterns
- `feedback`: do or do not repeat specific behaviors
- `project`: stable repo conventions, commands, or architecture notes
- `reference`: durable supporting information worth resurfacing later

Success criteria:
- The memory type is explicit.
- Ephemeral task-specific notes are rejected instead of stored.

### 3. Apply the change
Create, update, or remove the relevant memory file, then update the index so it remains a concise pointer list.

Success criteria:
- The memory file and index agree.
- The index entry is one concise line.

### 4. Confirm the durable value
Briefly explain what was stored or removed and why it will matter in future work.

Success criteria:
- The user can tell what changed without rereading every file.

## Rules
- Do not store transient TODOs, one-off debugging facts, or information that can be re-derived cheaply from the repo.
- Prefer updating the best existing memory over creating near-duplicates.
- If the user asks to forget something, remove both the topic file and the index reference.
