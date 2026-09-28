"""Fixed, reproducible model comparison for learning, not model selection."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data import feature_frame, load_dataset
from .model import make_pipeline, make_preprocessor, ordered_split, probability_report


def history_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """An inspection-only feature derived from a field known before contact."""
    features = feature_frame(frame)
    features["had_previous_contact"] = features["previous"].gt(0).astype(int)
    return features


def period_profile(frame: pd.DataFrame) -> dict:
    """Show source changes that can undermine a fixed historical model."""
    return {
        "rows": len(frame),
        "positive_rate": round(float(frame["y"].eq("yes").mean()), 4),
        "had_previous_contact": round(float(frame["previous"].gt(0).mean()), 4),
        "previous_success": round(float(frame["poutcome"].eq("success").mean()), 4),
        "unknown_default": round(float(frame["default"].eq("unknown").mean()), 4),
    }


def compare_models(csv_path: Path) -> dict:
    """Fit on the early period and report validation and later test separately."""
    try:
        from xgboost import XGBClassifier, __version__ as xgboost_version
    except ImportError as error:
        raise RuntimeError("Install the comparison extra: pip install -e '.[comparison]'") from error

    frame = load_dataset(csv_path)
    training, validation, test = ordered_split(frame)
    training_labels = training["y"].eq("yes").astype(int)
    if training_labels.nunique() != 2:
        raise ValueError("Training period must contain both outcomes")

    logistic = make_pipeline()
    boosted = Pipeline([
        ("transform", make_preprocessor()),
        ("classifier", XGBClassifier(
            objective="binary:logistic", tree_method="hist", eval_metric="logloss",
            n_estimators=120, max_depth=2, learning_rate=0.05,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=10, random_state=17, n_jobs=2,
        )),
    ])
    engineered = Pipeline([
        ("transform", make_preprocessor(("previous", "had_previous_contact"))),
        ("classifier", LogisticRegression(max_iter=1000, random_state=17)),
    ])
    models = {
        "training_rate": (DummyClassifier(strategy="prior"), feature_frame),
        "logistic_regression": (logistic, feature_frame),
        "logistic_with_history_flag": (engineered, history_feature_frame),
        "xgboost": (boosted, feature_frame),
    }
    results = {}
    for name, (model, features_for_model) in models.items():
        model.fit(features_for_model(training), training_labels)
        periods = {}
        for period_name, period in (("validation", validation), ("later_test", test)):
            labels = period["y"].eq("yes").astype(int)
            probabilities = model.predict_proba(features_for_model(period))[:, 1]
            report = probability_report(labels, probabilities)
            if name == "training_rate":
                for key in ("review_capacity", "precision_at_10_percent",
                            "precision_at_10_percent_bounds", "ties_at_cutoff",
                            "selected_from_cutoff_tie", "recall_at_10_percent",
                            "lift_at_10_percent"):
                    report.pop(key)
            periods[period_name] = report
        results[name] = periods

    return {
        "source_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "framework_versions": {"scikit_learn": sklearn.__version__,
                               "xgboost": xgboost_version},
        "split": "source order: 60% training, 20% validation, 20% later test",
        "features": ["default", "housing", "loan", "poutcome", "previous"],
        "profiles": {"training": period_profile(training),
                     "validation": period_profile(validation), "later_test": period_profile(test)},
        "parameters": {
            "logistic_regression": {"max_iter": 1000, "random_state": 17},
            "logistic_with_history_flag": {"derived_feature": "previous > 0",
                                           "max_iter": 1000, "random_state": 17},
            "xgboost": {"n_estimators": 120, "max_depth": 2, "learning_rate": 0.05,
                         "min_child_weight": 10, "subsample": 0.8,
                         "colsample_bytree": 0.8, "reg_lambda": 10, "random_state": 17},
        },
        "results": results,
        "interpretation": (
            "Fixed teaching settings. This source and its later period have been inspected; "
            "a new model-selection claim needs a fresh future holdout. "
            "Historical response propensity is not causal uplift or contact approval."
        ),
    }
