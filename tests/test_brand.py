from __future__ import annotations

from pathlib import Path

from copilotcode_sdk.brand import BrandSpec, DEFAULT_BRAND


def test_default_brand_matches_public_contract() -> None:
    assert DEFAULT_BRAND.public_name == "CopilotCode"
    assert DEFAULT_BRAND.slug == "copilotcode"
    assert DEFAULT_BRAND.package_name == "copilotcode_sdk"
    assert DEFAULT_BRAND.distribution_name == "copilotcode-sdk"
    assert DEFAULT_BRAND.cli_name == "copilotcode"
    assert DEFAULT_BRAND.app_dirname == ".copilotcode"


def test_brand_spec_derived_names_and_paths() -> None:
    brand = BrandSpec(
        public_name="FuturePilot",
        slug="futurepilot",
        package_name="futurepilot_sdk",
        distribution_name="futurepilot-sdk",
        cli_name="futurepilot",
        app_dirname=".futurepilot",
        source_inspiration="Claude Code",
    )

    assert brand.title == "FuturePilot"
    assert brand.module_name == "futurepilot_sdk"
    assert brand.client_name == "futurepilot_sdk"
    assert brand.app_home() == Path("~/.futurepilot").expanduser()
    assert brand.memory_home() == Path("~/.futurepilot").expanduser()
    assert brand.app_config_home() == Path("~/.futurepilot/config").expanduser()
    assert brand.state_home() == Path("~/.futurepilot/state").expanduser()
    assert brand.copilot_default_config_home() == Path("~/.copilot").expanduser()
