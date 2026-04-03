---
name: skillify
description: Convert a successful repeatable workflow into a reusable Copilot skill.
allowed-tools:
  - read
  - edit
  - search
  - execute
  - ask_user
when_to_use: Use near the end of a successful workflow when the same process is likely to repeat. Examples: "turn this into a skill", "capture this workflow", "make a reusable skill from this session".
argument-hint: "[optional description of the workflow]"
---

# Skillify

## Goal
Capture a repeatable workflow as a clear `SKILL.md` that another session can execute reliably.

## Steps

### 1. Analyze the workflow
Review what was done, what inputs were required, what artifacts were produced, and where the user corrected the process.

Success criteria:
- You can describe the workflow as ordered steps with explicit success criteria.

### 2. Interview for missing details
Use `ask_user` only for the details that are still ambiguous: naming, storage location, trigger phrases, required arguments, and hard constraints.

Success criteria:
- You asked only the questions needed to make the skill reusable.

### 3. Draft the skill
Write a `SKILL.md` with frontmatter, goal, inputs, steps, and success criteria for each major step.

Success criteria:
- The draft is self-contained.
- Someone new could run the process without access to the original conversation.

### 4. Confirm and save
Show the proposed skill, get confirmation if appropriate, then save it in the chosen skill directory.

Success criteria:
- The user knows where the skill lives and how to invoke it.

## Rules
- Keep simple workflows simple.
- Encode the user's corrections and preferences as explicit rules.
- Prefer clear triggers and concrete artifacts over vague motivational language.
