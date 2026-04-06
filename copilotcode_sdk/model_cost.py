"""Model cost calculation for Claude API usage."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

# Pricing per million tokens (as of 2025)
# Format: (input_per_1M, output_per_1M, cache_read_per_1M, cache_write_per_1M)
MODEL_PRICING: dict[str, tuple[float, float, float, float]] = {
    # Claude 4.x / Opus
    "claude-opus-4-20250514": (15.0, 75.0, 1.50, 18.75),
    "claude-opus-4-6": (15.0, 75.0, 1.50, 18.75),
    # Claude 4.x / Sonnet
    "claude-sonnet-4-20250514": (3.0, 15.0, 0.30, 3.75),
    "claude-sonnet-4-6": (3.0, 15.0, 0.30, 3.75),
    # Claude 3.5/4.5
    "claude-3-5-sonnet-20241022": (3.0, 15.0, 0.30, 3.75),
    "claude-3-5-haiku-20241022": (0.80, 4.0, 0.08, 1.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0, 0.08, 1.0),
    # Claude 3
    "claude-3-opus-20240229": (15.0, 75.0, 1.50, 18.75),
    "claude-3-sonnet-20240229": (3.0, 15.0, 0.30, 3.75),
    "claude-3-haiku-20240307": (0.25, 1.25, 0.03, 0.30),
}

# Short aliases map to full model IDs
_ALIASES: dict[str, str] = {
    "opus": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
    "claude-opus": "claude-opus-4-6",
    "claude-sonnet": "claude-sonnet-4-6",
    "claude-haiku": "claude-haiku-4-5-20251001",
}

def _resolve_model(model: str) -> str:
    """Resolve model aliases and partial matches."""
    if model in MODEL_PRICING:
        return model
    if model in _ALIASES:
        return _ALIASES[model]
    # Try prefix match
    for key in MODEL_PRICING:
        if key.startswith(model):
            return key
    return model

@dataclass(frozen=True, slots=True)
class UsageCost:
    """Calculated cost breakdown for a usage record."""
    input_cost: float
    output_cost: float
    cache_read_cost: float
    cache_creation_cost: float

    @property
    def total(self) -> float:
        return self.input_cost + self.output_cost + self.cache_read_cost + self.cache_creation_cost

    def format(self) -> str:
        return f"${self.total:.4f} (in: ${self.input_cost:.4f}, out: ${self.output_cost:.4f}, cache_r: ${self.cache_read_cost:.4f}, cache_w: ${self.cache_creation_cost:.4f})"

def calculate_cost(
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> UsageCost:
    """Calculate the cost of API usage for a given model and token counts."""
    resolved = _resolve_model(model)
    pricing = MODEL_PRICING.get(resolved)
    if pricing is None:
        # Unknown model - return zero cost
        return UsageCost(0.0, 0.0, 0.0, 0.0)

    inp_rate, out_rate, cache_r_rate, cache_w_rate = pricing
    return UsageCost(
        input_cost=input_tokens * inp_rate / 1_000_000,
        output_cost=output_tokens * out_rate / 1_000_000,
        cache_read_cost=cache_read_tokens * cache_r_rate / 1_000_000,
        cache_creation_cost=cache_creation_tokens * cache_w_rate / 1_000_000,
    )

def get_knowledge_cutoff(model: str) -> str:
    """Return the knowledge cutoff date string for a model."""
    resolved = _resolve_model(model)
    if "opus-4-6" in resolved or "opus-4-5" in resolved:
        return "May 2025"
    if "sonnet-4-6" in resolved:
        return "August 2025"
    if "sonnet-4" in resolved:
        return "January 2025"
    if "haiku-4" in resolved:
        return "February 2025"
    if "opus" in resolved:
        return "Early 2024"
    if "3-5-sonnet" in resolved:
        return "April 2024"
    return "Early 2024"
