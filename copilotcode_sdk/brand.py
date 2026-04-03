from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BrandSpec:
    public_name: str
    slug: str
    package_name: str
    distribution_name: str
    cli_name: str
    app_dirname: str
    source_inspiration: str = "Claude Code"

    @property
    def title(self) -> str:
        return self.public_name

    @property
    def module_name(self) -> str:
        return self.package_name

    @property
    def client_name(self) -> str:
        return self.package_name

    def app_home(self) -> Path:
        return Path(f"~/{self.app_dirname}").expanduser()

    def memory_home(self) -> Path:
        return self.app_home()

    def app_config_home(self) -> Path:
        return self.app_home() / "config"

    def state_home(self) -> Path:
        return self.app_home() / "state"

    def copilot_default_config_home(self) -> Path:
        return Path("~/.copilot").expanduser()


DEFAULT_BRAND = BrandSpec(
    public_name="CopilotCode",
    slug="copilotcode",
    package_name="copilotcode_sdk",
    distribution_name="copilotcode-sdk",
    cli_name="copilotcode",
    app_dirname=".copilotcode",
)
