"""Narrow ADK tool interfaces with no outbound send capability."""

from __future__ import annotations

import os
from pathlib import Path

from .data import FEATURES
from .model import score


def score_lead(default: str, housing: str, loan: str, poutcome: str, previous: int) -> dict:
    """Estimate historical response propensity from five pre-contact fields."""
    fields = dict(zip(FEATURES, (default, housing, loan, poutcome, previous)))
    probability = score(fields, Path(os.getenv("LEADHARBOUR_MODEL_PATH", "artifacts/model.joblib")))
    return {"propensity": probability, "meaning": "Historical response propensity, not causal uplift"}


def request_human_review(lead_reference: str, reason: str) -> dict:
    """Prepare a review request; no CRM or email system is called."""
    if not lead_reference.strip() or not reason.strip():
        raise ValueError("A reference and reason are required")
    return {"status": "review_required", "lead_reference": lead_reference[:80],
            "reason": reason[:240], "sent": False}
