import pytest

from fissure_annotation.agent import ResponseUsage
from fissure_annotation.pricing import TokenPrices, cost_usd, prices_for

PRICES = {
    "default": TokenPrices(input=2.0, cached_input=0.1, cache_write=2.5, output=10.0),
    "flex": TokenPrices(input=1.0, cached_input=0.05, cache_write=1.25, output=5.0),
}


def test_each_kind_of_input_token_has_its_own_price():
    usage = ResponseUsage(
        input_tokens=1_000_000, cached_input_tokens=500_000, cache_write_tokens=100_000, output_tokens=0
    )
    assert cost_usd(usage, PRICES["default"]) == pytest.approx(0.4 * 2.0 + 0.5 * 0.1 + 0.1 * 2.5)


def test_reasoning_tokens_are_priced_as_output_and_not_twice():
    usage = ResponseUsage(input_tokens=0, output_tokens=1_000_000, reasoning_tokens=800_000)
    assert cost_usd(usage, PRICES["default"]) == pytest.approx(10.0)


def test_a_response_is_priced_at_the_tier_that_served_it():
    assert prices_for(ResponseUsage(input_tokens=1, output_tokens=1, service_tier="flex"), PRICES).input == 1.0
    assert prices_for(ResponseUsage(input_tokens=1, output_tokens=1), PRICES).input == 2.0
