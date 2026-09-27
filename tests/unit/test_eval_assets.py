import json
from pathlib import Path

import pytest


def test_adk_eval_assets_match_installed_schema():
    pytest.importorskip("google.adk")
    from google.adk.evaluation.eval_config import EvalConfig
    from google.adk.evaluation.eval_set import EvalSet

    folder = Path(__file__).parents[1] / "evals"
    cases = EvalSet.model_validate_json((folder / "review.evalset.json").read_text())
    config = EvalConfig.model_validate_json((folder / "eval_config.json").read_text())
    assert len(cases.eval_cases) == 2
    assert config.criteria["tool_trajectory_avg_score"] == 1.0
    assert json.loads((folder / "review.evalset.json").read_text())["eval_set_id"]
