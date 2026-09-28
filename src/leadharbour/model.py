"""A reviewable propensity baseline with an ordered holdout."""

from __future__ import annotations

import hashlib
import json
from math import ceil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import CATEGORICAL_FEATURES, FEATURES, FORBIDDEN_FEATURES, NUMERIC_FEATURES, feature_frame, load_dataset


def ordered_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep the UCI file's documented date order for a future-period test."""
    if len(frame) < 30:
        raise ValueError("At least 30 ordered rows are needed for this teaching split")
    training_end = int(len(frame) * 0.6)
    validation_end = int(len(frame) * 0.8)
    return frame.iloc[:training_end], frame.iloc[training_end:validation_end], frame.iloc[validation_end:]


def make_preprocessor(numeric_features: tuple[str, ...] = NUMERIC_FEATURES) -> ColumnTransformer:
    if set(FEATURES).intersection(FORBIDDEN_FEATURES):
        raise AssertionError("A post-contact or sensitive field entered the feature list")
    return ColumnTransformer([
        ("categories", Pipeline([
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), list(CATEGORICAL_FEATURES)),
        ("numbers", SimpleImputer(strategy="median"), list(numeric_features)),
    ])
def make_pipeline() -> Pipeline:
    return Pipeline([("transform", make_preprocessor()),
                     ("classifier", LogisticRegression(max_iter=1000, random_state=17))])


def quality_report(pipeline: Pipeline, frame: pd.DataFrame) -> dict:
    labels = frame["y"].eq("yes").astype(int)
    probabilities = pipeline.predict_proba(feature_frame(frame))[:, 1]
    return probability_report(labels, probabilities)


def probability_report(labels: pd.Series, probabilities: np.ndarray) -> dict:
    """Measure ranking, probability quality and a fixed review capacity."""
    if labels.nunique() != 2:
        raise ValueError("Evaluation period must contain both outcomes")
    if len(labels) != len(probabilities) or not np.isfinite(probabilities).all():
        raise ValueError("Expected one finite probability per label")
    scores = np.asarray(probabilities)
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be probabilities between zero and one")
    capacity = ceil(len(labels) * 0.1)
    cutoff = np.partition(scores, len(scores) - capacity)[len(scores) - capacity]
    above = scores > cutoff
    tied = scores == cutoff
    labels_array = labels.to_numpy()
    needed_from_tie = capacity - int(above.sum())
    tied_positives = int(labels_array[tied].sum())
    tied_negatives = int(tied.sum()) - tied_positives
    positives_above = int(labels_array[above].sum())
    expected_positives = positives_above + needed_from_tie * tied_positives / int(tied.sum())
    precision = expected_positives / capacity
    lower_precision = (positives_above + max(0, needed_from_tie - tied_negatives)) / capacity
    upper_precision = (positives_above + min(needed_from_tie, tied_positives)) / capacity
    positive_rate = float(labels.mean())
    return {
        "rows": len(labels),
        "positive_rate": round(float(labels.mean()), 4),
        "roc_auc": round(float(roc_auc_score(labels, probabilities)), 4),
        "average_precision": round(float(average_precision_score(labels, probabilities)), 4),
        "brier_score": round(float(brier_score_loss(labels, probabilities)), 4),
        "log_loss": round(float(log_loss(labels, probabilities, labels=[0, 1])), 4),
        "review_capacity": capacity,
        "precision_at_10_percent": round(precision, 4),
        "precision_at_10_percent_bounds": [round(lower_precision, 4), round(upper_precision, 4)],
        "ties_at_cutoff": int(tied.sum()),
        "selected_from_cutoff_tie": needed_from_tie,
        "recall_at_10_percent": round(float(expected_positives / labels.sum()), 4),
        "lift_at_10_percent": round(precision / positive_rate, 4),
    }


def train(csv_path: Path, artifact_path: Path) -> dict:
    frame = load_dataset(csv_path)
    training, validation, test = ordered_split(frame)
    pipeline = make_pipeline()
    pipeline.fit(feature_frame(training), training["y"].eq("yes").astype(int))
    source_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    report = {
        "source_sha256": source_hash,
        "framework_versions": {"scikit_learn": sklearn.__version__},
        "features": list(FEATURES),
        "excluded_examples": list(FORBIDDEN_FEATURES),
        "split": "first 60% train, next 20% validation, final 20% test in source order",
        "validation": quality_report(pipeline, validation),
        "test": quality_report(pipeline, test),
        "limits": [
            "Historical contact outcomes are response propensity, not causal uplift.",
            "Records lack a stable person ID, so repeat contacts may cross the split.",
            "This model is a teaching baseline and must not choose real recipients.",
        ],
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "source_sha256": source_hash, "features": list(FEATURES)}, artifact_path)
    artifact_path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def evaluate(csv_path: Path, artifact_path: Path) -> dict:
    artifact = load_artifact(artifact_path)
    if hashlib.sha256(csv_path.read_bytes()).hexdigest() != artifact["source_sha256"]:
        raise ValueError("Source data differs from the data used to train this artifact")
    _, validation, test = ordered_split(load_dataset(csv_path))
    return {"validation": quality_report(artifact["pipeline"], validation),
            "test": quality_report(artifact["pipeline"], test)}


def load_artifact(path: Path) -> dict:
    """Load only an artifact built locally or from a trusted image."""
    if not path.exists():
        raise FileNotFoundError(f"Model artifact is missing: {path}. Run leadharbour train first")
    artifact = joblib.load(path)
    if artifact.get("features") != list(FEATURES):
        raise ValueError("Model feature schema does not match this code version")
    return artifact


def score(features: dict, artifact_path: Path) -> float:
    return score_with_artifact(features, load_artifact(artifact_path))


def score_with_artifact(features: dict, artifact: dict) -> float:
    if set(features) != set(FEATURES):
        raise ValueError(f"Expected exactly these features: {', '.join(FEATURES)}")
    for field in CATEGORICAL_FEATURES[:3]:
        if features[field] not in {"yes", "no", "unknown"}:
            raise ValueError(f"Unsupported {field} value")
    if features["poutcome"] not in {"success", "failure", "nonexistent"}:
        raise ValueError("Unsupported poutcome value")
    if (type(features["previous"]) is not int or
            not 0 <= features["previous"] <= 50):
        raise ValueError("previous must be an integer between 0 and 50")
    probability = artifact["pipeline"].predict_proba(pd.DataFrame([features]))[0, 1]
    return round(float(probability), 6)
