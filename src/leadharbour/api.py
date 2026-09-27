"""A small scoring API; Cloud Run IAM controls cloud callers."""

from __future__ import annotations

import hmac
import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .model import load_artifact, score_with_artifact

LOGGER = logging.getLogger("leadharbour.scoring")


class LeadFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    default: Literal["yes", "no", "unknown"]
    housing: Literal["yes", "no", "unknown"]
    loan: Literal["yes", "no", "unknown"]
    poutcome: Literal["success", "failure", "nonexistent"]
    previous: int = Field(ge=0, le=50)


def create_app(artifact_path: Path, *, local_token: str | None = None,
               cloud_iam: bool = False) -> FastAPI:
    if not cloud_iam and (local_token is None or len(local_token) < 32):
        raise ValueError("Set LEADHARBOUR_API_TOKEN to at least 32 random characters")
    if cloud_iam and not os.environ.get("K_SERVICE"):
        raise ValueError("Cloud IAM mode requires a Cloud Run service")
    artifact = load_artifact(artifact_path)
    app = FastAPI(title="LeadHarbour scoring", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        if request.url.path == "/score" and not cloud_iam:
            provided = request.headers.get("Authorization", "")
            if not hmac.compare_digest(provided, f"Bearer {local_token}"):
                return JSONResponse({"detail": "Unauthorised"}, status_code=401)
        return await call_next(request)

    @app.get("/healthz")
    def health() -> dict:
        return {"status": "ready"}

    @app.post("/score")
    def predict(features: LeadFeatures) -> dict:
        started = time.perf_counter()
        probability = score_with_artifact(features.model_dump(), artifact)
        trace_id = secrets.token_hex(8)
        LOGGER.info(json.dumps({
            "event": "score_completed", "trace_id": trace_id,
            "model_version": artifact["source_sha256"][:12],
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        }))
        return {
            "propensity": probability,
            "model_version": artifact["source_sha256"][:12],
            "trace_id": trace_id,
            "meaning": "Historical response propensity, not uplift or contact approval",
        }

    return app


def app_from_environment() -> FastAPI:
    return create_app(
        Path(os.getenv("LEADHARBOUR_MODEL_PATH", "artifacts/model.joblib")),
        local_token=os.getenv("LEADHARBOUR_API_TOKEN"),
        cloud_iam=os.getenv("LEADHARBOUR_AUTH_MODE") == "cloud_iam",
    )
