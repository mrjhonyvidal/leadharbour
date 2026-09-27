from types import SimpleNamespace
import logging

import pytest
from fastapi.testclient import TestClient

from leadharbour.agent_tools import request_human_review
from leadharbour.api import create_app
from leadharbour.drafts import draft_for_review


def test_api_checks_auth_before_body_and_rejects_bad_schema(trained_lab, caplog):
    caplog.set_level(logging.INFO, logger="leadharbour.scoring")
    _, artifact_path, _ = trained_lab
    app = create_app(artifact_path, local_token="a" * 40)
    client = TestClient(app)
    assert client.post("/score", json={"duration": 500}).status_code == 401
    headers = {"Authorization": "Bearer " + "a" * 40}
    assert client.post("/score", headers=headers, json={"duration": 500}).status_code == 422
    assert client.post("/score", headers=headers, json={
        "default": "no", "housing": "yes", "loan": "no", "poutcome": "nonexistent",
        "previous": 0, "duration": 500,
    }).status_code == 422
    response = client.post("/score", headers=headers, json={
        "default": "no", "housing": "yes", "loan": "no", "poutcome": "nonexistent", "previous": 0,
    })
    assert response.status_code == 200
    assert 0 <= response.json()["propensity"] <= 1
    assert response.json()["trace_id"]
    assert response.json()["trace_id"] in caplog.text
    assert "housing" not in caplog.text


def test_cloud_mode_requires_cloud_run_and_review_tool_never_sends(trained_lab, monkeypatch):
    _, artifact_path, _ = trained_lab
    monkeypatch.delenv("K_SERVICE", raising=False)
    with pytest.raises(ValueError, match="Cloud Run"):
        create_app(artifact_path, cloud_iam=True)
    assert request_human_review("demo-1", "Check consent")["sent"] is False


def test_vertex_draft_requires_review_and_has_no_send_path():
    calls = []
    client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs:
        calls.append(kwargs) or SimpleNamespace(text="Draft for review")))
    with pytest.raises(ValueError, match="consent"):
        draft_for_review("Approved public context", consent_checked=False,
                         project="demo", location="europe-west2", client=client)
    result = draft_for_review("Approved public context", consent_checked=True,
                              project="demo", location="europe-west2", client=client)
    assert result == {"draft": "Draft for review", "sent": False, "review_required": True}
    assert len(calls) == 1
