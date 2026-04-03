from __future__ import annotations

from copilotcode_sdk import CopilotCodeClient, CopilotCodeConfig


def main() -> None:
    client = CopilotCodeClient(
        CopilotCodeConfig(
            working_directory=".",
        ),
    )
    report = client.preflight()
    print(report.to_text())

    session = client.create_session()
    event = session.send_and_wait(
        "Inspect this repository and summarize how tests are expected to run.",
        timeout=120.0,
    )
    print(event)


if __name__ == "__main__":
    main()
