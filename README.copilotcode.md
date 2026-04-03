# CopilotCode SDK

`copilotcode_sdk` is a pure-Python layer on top of the [GitHub Copilot Python SDK](https://github.com/github/copilot-sdk). It ports the most useful portable behaviors from Claude Code into a Copilot-friendly runtime: a compiled system prompt, scoped custom agents, reusable `SKILL.md` workflows, durable file-backed memory, safer permission defaults, and repo instruction materialization.

## Quickstart

Install the package and the Copilot SDK:

```bash
pip install github-copilot-sdk copilotcode-sdk
```

Run a local environment check:

```bash
copilotcode preflight
```

Materialize repo instructions into the current workspace:

```bash
copilotcode init
```

Run the default local confidence flow:

```bash
copilotcode validate
```

Run the deterministic flow with a coverage gate:

```bash
copilotcode validate --coverage
```

Run a dry-run smoke test:

```bash
copilotcode smoke
```

## Python Usage

```python
from copilotcode_sdk import CopilotCodeClient, CopilotCodeConfig

config = CopilotCodeConfig(working_directory=".")
client = CopilotCodeClient(config)

preflight = client.preflight()
print(preflight.to_text())

session = client.create_session()
result = session.send_and_wait(
    "Inspect this repository and summarize the expected verification flow.",
    timeout=120.0,
)
print(result)
```

## CLI Commands

- `copilotcode preflight`: verify SDK importability, Copilot CLI availability, writable config and memory paths, and likely auth state.
- `copilotcode init`: write `CLAUDE.md` and `.github/copilot-instructions.md` into a workspace.
- `copilotcode smoke`: run a dry-run or live smoke test, optionally saving a report or transcript artifact.
- `copilotcode validate`: run the canonical local confidence ladder for this repo checkout.
- `copilotcode memory list`: show durable memory records for the workspace.
- `copilotcode memory reindex`: regenerate `MEMORY.md` from the durable memory files.

## Local Validation Ladder

CopilotCode does not currently ship GitHub Actions or other repository CI. Confidence is intentionally local-first.

1. Check the environment:

```bash
copilotcode preflight
```

2. Run the deterministic validation subset:

```bash
copilotcode validate
```

3. Add packaging validation:

```bash
copilotcode validate --include-packaging
```

4. Add the coverage gate for `copilotcode_sdk`:

```bash
copilotcode validate --coverage
```

5. Confirm that live Copilot prerequisites are present:

```bash
copilotcode preflight --require-auth
```

6. Run the manual live validation subset:

```bash
copilotcode validate --include-live
```

See [docs/manual-validation.md](docs/manual-validation.md) for the live-validation environment variables, artifact capture, and failure-triage workflow.

## Safety Defaults

The default permission policy is `safe`.

- Read and search tools are approved automatically.
- Writes are limited to the workspace, CopilotCode config directory, and memory roots.
- Shell commands are denied unless they match an approved prefix.

Use `permission_policy="approve_all"` in Python or `--permission-policy approve_all` in the CLI only for explicit smoke runs or tightly controlled automation.

## Durable Memory

CopilotCode stores durable memory under `~/.copilotcode/projects/<project>/memory/`.

- `MEMORY.md` is a concise index.
- Topic files hold the actual durable memory content.
- Query-time memory injection uses deterministic header and keyword scoring.

## Manual Live Validation

The manual live path stays opt-in and local-only.

- `COPILOTCODE_RUN_LIVE=1` enables the live pytest subset when run directly.
- `COPILOTCODE_TEST_MODEL` optionally overrides the model used by the live suite.
- `COPILOTCODE_TEST_COPILOT_CONFIG_DIR` points the live suite at a specific Copilot CLI config directory.
- `COPILOTCODE_TEST_GITHUB_TOKEN` optionally supplies auth without relying on the logged-in Copilot CLI user.
- `COPILOTCODE_TEST_CLI_PATH` overrides the Copilot CLI executable path for the live suite.
- `COPILOTCODE_LIVE_ARTIFACT_DIR` preserves transcripts and normalized live artifacts for debugging.
- `COPILOTCODE_RUN_VALIDATE_LIVE_E2E=1` opts into the slowest live check, which runs `copilotcode validate --include-live` from inside the live suite to confirm artifact preservation end to end.

## Test Depth

The repo expects local confidence from several layers, not just a raw test count:

- deterministic unit and contract tests across the CLI, client, hooks, permissions, memory, branding, and reports
- packaging validation for the built wheel and sdist
- a 90% line-coverage floor for `copilotcode_sdk` when running `copilotcode validate --coverage`
- manual live Copilot validation with saved artifacts when real-runtime confidence matters

## Notes

- The package is pure Python, but the runtime still depends on the Copilot CLI that the SDK uses underneath.
- `CLAUDE.md` remains the generated repo instruction filename because that is an interoperability surface, not product branding.
