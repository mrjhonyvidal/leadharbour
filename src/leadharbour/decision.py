"""Transparent value and capacity calculations, without automated sending."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    lead_id: str
    propensity: float
    value_if_converted: float
    contact_cost: float


def expected_net_value(candidate: Candidate) -> float:
    if not 0 <= candidate.propensity <= 1:
        raise ValueError("Propensity must be between zero and one")
    if candidate.value_if_converted < 0 or candidate.contact_cost < 0:
        raise ValueError("Value and cost must not be negative")
    return candidate.propensity * candidate.value_if_converted - candidate.contact_cost


def rank_for_review(candidates: list[Candidate], capacity: int) -> list[Candidate]:
    """Return positive-value candidates for human review within a capacity limit."""
    if capacity < 0:
        raise ValueError("Capacity must not be negative")
    if len({candidate.lead_id for candidate in candidates}) != len(candidates):
        raise ValueError("Lead IDs must be unique")
    ranked = sorted(candidates, key=lambda candidate: (-expected_net_value(candidate), candidate.lead_id))
    return [candidate for candidate in ranked if expected_net_value(candidate) > 0][:capacity]


def estimate_generation_cost(leads: int, draft_fraction: float, input_tokens: int,
                             output_tokens: int, input_price_per_million: float,
                             output_price_per_million: float) -> float:
    if leads < 0 or not 0 <= draft_fraction <= 1 or min(input_tokens, output_tokens) < 0:
        raise ValueError("Counts must be non-negative and draft fraction must be at most one")
    if min(input_price_per_million, output_price_per_million) < 0:
        raise ValueError("Prices must not be negative")
    drafts = leads * draft_fraction
    return round(drafts * (input_tokens * input_price_per_million
                           + output_tokens * output_price_per_million) / 1_000_000, 2)
