# Manual Validation

CopilotCode keeps real Copilot runtime validation manual-only. This repo currently does not ship CI, so the intended confidence ladder is local.

## Prerequisites

- `github-copilot-sdk` installed
- `pytest` installed
- `pytest-cov` installed if you want coverage validation
- `hypothesis` installed if you want the full property-based deterministic suite
- `build` installed if you want packaging validation
- Copilot CLI installed or available through the SDK bundle
- Copilot authentication configured, either through `copilot login` or `COPILOTCODE_TEST_GITHUB_TOKEN`

## Recommended Flow

1. Environment readiness:

```bash
copilotcode preflight
```

2. Deterministic suite:

```bash
copilotcode validate
```

3. Packaging verification:

```bash
copilotcode validate --include-packaging
```

4. Coverage-gated deterministic verification:

```bash
copilotcode validate --coverage
```

5. Live-readiness check:

```bash
copilotcode preflight --require-auth
```

6. Manual live validation:

```bash
copilotcode validate --include-live
```

## Useful Environment Variables

- `COPILOTCODE_RUN_LIVE=1`
  - Enables the live pytest subset when you run it directly instead of through `copilotcode validate --include-live`.
- `COPILOTCODE_TEST_MODEL`
  - Optional model override for the live suite.
- `COPILOTCODE_TEST_COPILOT_CONFIG_DIR`
  - Optional Copilot CLI config directory override.
- `COPILOTCODE_TEST_GITHUB_TOKEN`
  - Optional GitHub token for live validation.
- `COPILOTCODE_TEST_CLI_PATH`
  - Optional Copilot CLI executable override.
- `COPILOTCODE_LIVE_ARTIFACT_DIR`
  - Directory where live transcript artifacts will be written.
- `COPILOTCODE_RUN_VALIDATE_LIVE_E2E=1`
  - Opts into the slowest live validation case, which runs `copilotcode validate --include-live` from inside the live suite to verify end-to-end artifact preservation.

Example:

```powershell
$env:COPILOTCODE_LIVE_ARTIFACT_DIR = "$PWD\\.live-artifacts"
copilotcode preflight --require-auth
copilotcode validate --include-live
```

Optional deepest end-to-end live check:

```powershell
$env:COPILOTCODE_LIVE_ARTIFACT_DIR = "$PWD\\.live-artifacts"
$env:COPILOTCODE_RUN_VALIDATE_LIVE_E2E = "1"
py -3 -m pytest tests/test_live.py -q
```

## Artifact Expectations

When `COPILOTCODE_LIVE_ARTIFACT_DIR` is set, the live suite writes JSON artifacts that include:

- session id
- test label
- original prompt
- normalized assistant text
- raw message payloads

The `copilotcode smoke` command can also save:

- a JSON smoke report via `--save-report`
- live transcripts via `--save-transcript`

## Failure Triage

- `preflight --require-auth` fails:
  - Check Copilot login state or provide `COPILOTCODE_TEST_GITHUB_TOKEN`.
- `validate --include-packaging` fails before running tests:
  - Install `build` with `pip install build` or `pip install .[dev]`.
- `validate --coverage` fails before running tests:
  - Install `pytest-cov` with `pip install pytest-cov` or `pip install .[dev]`.
- live validation fails with weak or ambiguous output:
  - Inspect the JSON artifacts in `COPILOTCODE_LIVE_ARTIFACT_DIR`.
  - Confirm the workspace files and durable memory entries were created as expected.
  - Re-run the specific live subset with `py -3 -m pytest tests/test_live.py -q`.
- live validation is unexpectedly slow:
  - Confirm `COPILOTCODE_RUN_VALIDATE_LIVE_E2E` is not set unless you intentionally want the deepest end-to-end validation path.
