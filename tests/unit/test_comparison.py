import numpy as np
import pandas as pd
import pytest

from leadharbour.comparison import compare_models, history_feature_frame, period_profile
from leadharbour.model import probability_report


def test_comparison_uses_same_ordered_periods_and_keeps_test_separate(trained_lab):
    pytest.importorskip("xgboost")
    csv_path, _, _ = trained_lab
    result = compare_models(csv_path)
    assert result["profiles"]["training"]["rows"] == 36
    assert result["profiles"]["validation"]["rows"] == 12
    assert result["profiles"]["later_test"]["rows"] == 12
    assert set(result["results"]) == {
        "training_rate", "logistic_regression", "logistic_with_history_flag", "xgboost"}
    assert result["results"]["training_rate"]["validation"]["roc_auc"] == 0.5
    assert "precision_at_10_percent" not in result["results"]["training_rate"]["later_test"]


def test_engineered_feature_uses_only_prior_contact_data(trained_lab):
    csv_path, _, _ = trained_lab
    frame = pd.read_csv(csv_path, sep=";")
    features = history_feature_frame(frame)
    assert "duration" not in features
    assert features["had_previous_contact"].tolist() == frame["previous"].gt(0).astype(int).tolist()
    assert period_profile(frame)["rows"] == 60


def test_probability_report_checks_ranking_and_probability_quality():
    labels = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    good = probability_report(labels, np.array([0.1, 0.9] * 5))
    assert good["roc_auc"] == 1
    assert good["precision_at_10_percent"] == 1
    assert good["lift_at_10_percent"] == 2
    assert good["brier_score"] == 0.01
    tied = probability_report(labels, np.full(10, 0.5))
    assert tied["precision_at_10_percent"] == 0.5
    assert tied["precision_at_10_percent_bounds"] == [0.0, 1.0]
    assert tied["ties_at_cutoff"] == 10
    with pytest.raises(ValueError, match="finite"):
        probability_report(labels, np.array([np.nan] * 10))
