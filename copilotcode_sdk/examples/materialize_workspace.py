from __future__ import annotations

from pathlib import Path

from copilotcode_sdk import CopilotCodeClient, CopilotCodeConfig


def main() -> None:
    workspace = Path(".").resolve()
    client = CopilotCodeClient(CopilotCodeConfig(working_directory=workspace))
    claude_path, copilot_path = client.materialize_workspace_instructions(
        overwrite=False,
    )
    print(f"Wrote {claude_path}")
    print(f"Wrote {copilot_path}")


if __name__ == "__main__":
    main()
