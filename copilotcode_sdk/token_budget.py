"""Token budget parsing and tracking for user-specified spending limits."""
from __future__ import annotations
import re
from dataclasses import dataclass

@dataclass(slots=True)
class TokenBudget:
    """A parsed token budget from user input."""
    raw: str           # Original text like "+500k"
    tokens: int        # Parsed token count (500000)
    consumed: int = 0  # Tokens consumed so far

    @property
    def remaining(self) -> int:
        return max(0, self.tokens - self.consumed)

    @property
    def progress(self) -> float:
        """Progress as 0.0-1.0."""
        if self.tokens <= 0:
            return 1.0
        return min(1.0, self.consumed / self.tokens)

    @property
    def exhausted(self) -> bool:
        return self.consumed >= self.tokens

# Pattern: +500k, +1.5m, +2b at start or end of message
_BUDGET_PATTERN = re.compile(
    r'(?:^|\s)\+(\d+(?:\.\d+)?)\s*([kmb])\b',
    re.IGNORECASE,
)

_VERBOSE_PATTERN = re.compile(
    r'(?:use|spend|budget)\s+(\d+(?:\.\d+)?)\s*([kmb])\s+tokens',
    re.IGNORECASE,
)

_MULTIPLIERS = {'k': 1_000, 'm': 1_000_000, 'b': 1_000_000_000}

def parse_token_budget(text: str) -> TokenBudget | None:
    """Parse a token budget directive from user text.

    Supports shorthand (+500k, +1.5m, +2b) and verbose
    ("use 1M tokens", "spend 500k tokens").
    Returns None if no budget found.
    """
    for pattern in (_BUDGET_PATTERN, _VERBOSE_PATTERN):
        match = pattern.search(text)
        if match:
            amount = float(match.group(1))
            suffix = match.group(2).lower()
            tokens = int(amount * _MULTIPLIERS[suffix])
            return TokenBudget(raw=match.group(0).strip(), tokens=tokens)
    return None

def strip_budget_directive(text: str) -> str:
    """Remove token budget directives from user text."""
    result = text
    for pattern in (_BUDGET_PATTERN, _VERBOSE_PATTERN):
        result = pattern.sub('', result)
    return result.strip()

def format_budget_status(budget: TokenBudget) -> str:
    """Format a human-readable budget status message."""
    pct = int(budget.progress * 100)
    consumed_k = budget.consumed / 1000
    total_k = budget.tokens / 1000
    if budget.exhausted:
        return f"Token budget exhausted ({consumed_k:.0f}k/{total_k:.0f}k tokens used, {pct}%)."
    return f"Token budget: {consumed_k:.0f}k/{total_k:.0f}k tokens used ({pct}%)."
