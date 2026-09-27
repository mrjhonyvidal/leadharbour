import pandas as pd
import pytest

from leadharbour.model import train


@pytest.fixture
def trained_lab(tmp_path):
    frame = pd.DataFrame([
        {
            "default": "no" if index % 3 else "unknown",
            "housing": "yes" if index % 2 else "no",
            "loan": "no" if index % 4 else "yes",
            "poutcome": "success" if index % 5 == 0 else "nonexistent",
            "previous": index % 3,
            "age": 20 + index % 50,
            "duration": 500 if index % 5 == 0 else 30,
            "y": "yes" if index % 5 == 0 else "no",
        }
        for index in range(60)
    ])
    csv_path = tmp_path / "source.csv"
    artifact_path = tmp_path / "model.joblib"
    frame.to_csv(csv_path, sep=";", index=False)
    report = train(csv_path, artifact_path)
    return csv_path, artifact_path, report
