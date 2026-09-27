import pandas as pd
import pytest

from leadharbour.data import FEATURES, FORBIDDEN_FEATURES, feature_frame, load_dataset
from leadharbour.model import evaluate, ordered_split, score


def test_training_uses_only_precontact_fields_and_ordered_holdout(trained_lab):
    csv_path, artifact_path, report = trained_lab
    assert set(FEATURES).isdisjoint(FORBIDDEN_FEATURES)
    assert "duration" not in feature_frame(load_dataset(csv_path)).columns
    training, validation, test = ordered_split(load_dataset(csv_path))
    assert (len(training), len(validation), len(test)) == (36, 12, 12)
    assert report["test"] == evaluate(csv_path, artifact_path)["test"]
    assert 0 <= score({"default": "no", "housing": "yes", "loan": "no",
                       "poutcome": "nonexistent", "previous": 0}, artifact_path) <= 1


def test_evaluation_rejects_changed_source(trained_lab):
    csv_path, artifact_path, _ = trained_lab
    changed = pd.read_csv(csv_path, sep=";")
    changed.loc[0, "y"] = "no"
    changed.to_csv(csv_path, sep=";", index=False)
    with pytest.raises(ValueError, match="Source data differs"):
        evaluate(csv_path, artifact_path)


def test_schema_and_feature_validation(trained_lab, tmp_path):
    _, artifact_path, _ = trained_lab
    invalid = tmp_path / "invalid.csv"
    invalid.write_text("y;duration\nyes;120\n")
    with pytest.raises(ValueError, match="Missing required columns"):
        load_dataset(invalid)
    with pytest.raises(ValueError, match="Expected exactly"):
        score({"duration": 120}, artifact_path)
    features = {"default": "no", "housing": "yes", "loan": "no",
                "poutcome": "nonexistent", "previous": 0}
    with pytest.raises(ValueError, match="Unsupported default"):
        score({**features, "default": "maybe"}, artifact_path)
    with pytest.raises(ValueError, match="previous must"):
        score({**features, "previous": True}, artifact_path)
