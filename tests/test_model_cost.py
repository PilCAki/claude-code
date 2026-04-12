"""Tests for copilotcode_sdk.model_cost module."""
from __future__ import annotations

import pytest
from dataclasses import FrozenInstanceError

from copilotcode_sdk.model_cost import (
    MODEL_PRICING,
    UsageCost,
    _resolve_model,
    calculate_cost,
    get_knowledge_cutoff,
)


class TestCalculateCostOpus:
    def test_opus_input_only(self):
        cost = calculate_cost("claude-opus-4.6", input_tokens=1_000_000)
        assert cost.input_cost == pytest.approx(15.0)
        assert cost.output_cost == 0.0
        assert cost.total == pytest.approx(15.0)

    def test_opus_input_and_output(self):
        cost = calculate_cost("claude-opus-4-20250514", input_tokens=500_000, output_tokens=100_000)
        assert cost.input_cost == pytest.approx(7.5)
        assert cost.output_cost == pytest.approx(7.5)
        assert cost.total == pytest.approx(15.0)


class TestCalculateCostSonnet:
    def test_sonnet_basic(self):
        cost = calculate_cost("claude-sonnet-4.6", input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost.input_cost == pytest.approx(3.0)
        assert cost.output_cost == pytest.approx(15.0)
        assert cost.total == pytest.approx(18.0)


class TestCalculateCostHaiku:
    def test_haiku_cheapest(self):
        cost = calculate_cost("claude-3-haiku-20240307", input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost.input_cost == pytest.approx(0.25)
        assert cost.output_cost == pytest.approx(1.25)
        assert cost.total == pytest.approx(1.50)


class TestCalculateCostCache:
    def test_cache_tokens(self):
        cost = calculate_cost(
            "claude-opus-4.6",
            input_tokens=100_000,
            output_tokens=50_000,
            cache_read_tokens=200_000,
            cache_creation_tokens=80_000,
        )
        assert cost.input_cost == pytest.approx(1.5)
        assert cost.output_cost == pytest.approx(3.75)
        assert cost.cache_read_cost == pytest.approx(0.3)
        assert cost.cache_creation_cost == pytest.approx(1.5)
        assert cost.total == pytest.approx(7.05)


class TestCalculateCostUnknown:
    def test_unknown_model_returns_zero(self):
        cost = calculate_cost("gpt-4o-mini", input_tokens=1_000_000, output_tokens=500_000)
        assert cost.total == 0.0
        assert cost.input_cost == 0.0
        assert cost.output_cost == 0.0
        assert cost.cache_read_cost == 0.0
        assert cost.cache_creation_cost == 0.0


class TestUsageCost:
    def test_total_sums_all_components(self):
        uc = UsageCost(input_cost=1.0, output_cost=2.0, cache_read_cost=0.5, cache_creation_cost=0.25)
        assert uc.total == pytest.approx(3.75)

    def test_format_produces_dollar_string(self):
        uc = UsageCost(input_cost=0.015, output_cost=0.075, cache_read_cost=0.0003, cache_creation_cost=0.0)
        formatted = uc.format()
        assert formatted.startswith("$")
        assert "in: $0.0150" in formatted
        assert "out: $0.0750" in formatted
        assert "cache_r: $0.0003" in formatted
        assert "cache_w: $0.0000" in formatted

    def test_frozen_immutable(self):
        uc = UsageCost(input_cost=1.0, output_cost=2.0, cache_read_cost=0.0, cache_creation_cost=0.0)
        with pytest.raises(FrozenInstanceError):
            uc.input_cost = 99.0  # type: ignore[misc]


class TestResolveModel:
    def test_alias_opus(self):
        assert _resolve_model("opus") == "claude-opus-4.6"

    def test_alias_sonnet(self):
        assert _resolve_model("sonnet") == "claude-sonnet-4.6"

    def test_prefix_match(self):
        resolved = _resolve_model("claude-opus-4-2")
        assert resolved == "claude-opus-4-20250514"

    def test_exact_match(self):
        assert _resolve_model("claude-3-haiku-20240307") == "claude-3-haiku-20240307"

    def test_unknown_passthrough(self):
        assert _resolve_model("some-unknown-model") == "some-unknown-model"


class TestGetKnowledgeCutoff:
    def test_opus_4_6(self):
        assert get_knowledge_cutoff("claude-opus-4.6") == "May 2025"

    def test_sonnet_4_6(self):
        assert get_knowledge_cutoff("claude-sonnet-4.6") == "August 2025"

    def test_sonnet_4_dated(self):
        assert get_knowledge_cutoff("claude-sonnet-4-20250514") == "January 2025"

    def test_haiku_alias(self):
        assert get_knowledge_cutoff("haiku") == "February 2025"

    def test_3_5_sonnet(self):
        assert get_knowledge_cutoff("claude-3-5-sonnet-20241022") == "April 2024"

    def test_old_opus(self):
        assert get_knowledge_cutoff("claude-3-opus-20240229") == "Early 2024"


class TestModelPricingCompleteness:
    def test_has_all_major_models(self):
        expected = [
            "claude-opus-4-20250514",
            "claude-opus-4.6",
            "claude-sonnet-4-20250514",
            "claude-sonnet-4.6",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-haiku-4.5",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]
        for model_id in expected:
            assert model_id in MODEL_PRICING, f"Missing model: {model_id}"

    def test_pricing_tuples_have_four_elements(self):
        for model_id, pricing in MODEL_PRICING.items():
            assert len(pricing) == 4, f"{model_id} pricing should have 4 elements"
            assert all(isinstance(v, (int, float)) and v >= 0 for v in pricing), (
                f"{model_id} pricing values must be non-negative numbers"
            )
