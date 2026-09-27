"""Fetch and validate the original UCI Bank Marketing teaching dataset."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

SOURCE_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/222/bank+marketing"
INNER_ARCHIVE = "bank-additional.zip"
SOURCE_FILE = "bank-additional/bank-additional-full.csv"
CATEGORICAL_FEATURES = ("default", "housing", "loan", "poutcome")
NUMERIC_FEATURES = ("previous",)
FEATURES = (*CATEGORICAL_FEATURES, *NUMERIC_FEATURES)
FORBIDDEN_FEATURES = ("duration", "campaign", "contact", "age", "marital", "education", "job")
REQUIRED_COLUMNS = (*FEATURES, "age", "y")


def fetch_dataset(destination: Path) -> dict:
    """Save the source CSV and a provenance manifest; never modify source rows."""
    destination.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "LeadHarbour/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as outer:
        with zipfile.ZipFile(io.BytesIO(outer.read(INNER_ARCHIVE))) as inner:
            csv_bytes = inner.read(SOURCE_FILE)
    csv_path = destination / "bank-additional-full.csv"
    csv_path.write_bytes(csv_bytes)
    manifest = {
        "source": SOURCE_PAGE,
        "licence": "CC BY 4.0",
        "citation": "Moro, Rita and Cortez (2014), UCI Bank Marketing, doi:10.24432/C5K306",
        "rows": len(load_dataset(csv_path)),
        "sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "date_range": "May 2008 to November 2010",
    }
    (destination / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=";")
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if frame.empty or not frame["y"].isin(["yes", "no"]).all():
        raise ValueError("Expected non-empty source data with yes/no outcomes")
    return frame


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Select only fields known before a new contact."""
    return frame.loc[:, FEATURES].copy()
