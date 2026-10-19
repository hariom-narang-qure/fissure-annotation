"""Price the responses of a slice from their token counts, since neither Portkey nor OpenAI reports the cost of a request.

How:
- Prices are in USD per million tokens, one set per service tier, from the config.
  - Each response says which tier served it, so a flex request that fell back to the default tier is priced as default.
- Input tokens are split into three kinds, each at its own price:
  - cached tokens, read from the prompt cache;
  - cache write tokens, written to it;
  - the rest, at the plain input price.
- Output tokens include the reasoning tokens, and all are priced as output.
"""

from pydantic import BaseModel

from fissure_annotation.agent import ResponseUsage

TOKENS_PER_PRICE_UNIT = 1_000_000


class TokenPrices(BaseModel):
    input: float
    cached_input: float
    cache_write: float
    output: float


def uncached_input_tokens(usage: ResponseUsage) -> int:
    return usage.input_tokens - usage.cached_input_tokens - usage.cache_write_tokens


def cost_usd(usage: ResponseUsage, prices: TokenPrices) -> float:
    priced_tokens = (
        uncached_input_tokens(usage) * prices.input
        + usage.cached_input_tokens * prices.cached_input
        + usage.cache_write_tokens * prices.cache_write
        + usage.output_tokens * prices.output
    )
    return priced_tokens / TOKENS_PER_PRICE_UNIT


def prices_for(usage: ResponseUsage, prices_by_tier: dict[str, TokenPrices]) -> TokenPrices:
    "OpenAI calls its standard tier default"
    return prices_by_tier[usage.service_tier or "default"]
