import pytest

from leadharbour.decision import Candidate, estimate_generation_cost, rank_for_review


def test_value_rank_respects_capacity_and_avoids_negative_value():
    candidates = [Candidate("a", 0.4, 100, 5), Candidate("b", 0.2, 100, 5),
                  Candidate("c", 0.01, 100, 5)]
    assert [candidate.lead_id for candidate in rank_for_review(candidates, 2)] == ["a", "b"]
    with pytest.raises(ValueError, match="unique"):
        rank_for_review([candidates[0], candidates[0]], 2)
    with pytest.raises(ValueError, match="zero and one"):
        rank_for_review([Candidate("bad", 1.2, 100, 5)], 1)


def test_costs_scale_with_drafts_not_all_leads():
    assert estimate_generation_cost(10_000, 0.1, 500, 150, 1.0, 2.0) == 0.80
    with pytest.raises(ValueError):
        estimate_generation_cost(10_000, 1.2, 500, 150, 1.0, 2.0)
