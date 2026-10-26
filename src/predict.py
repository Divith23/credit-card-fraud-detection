import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "fraud_pipeline.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"

model = joblib.load(MODEL_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

EXPECTED_FEATURES = metadata["expected_features"]
THRESHOLD = float(metadata["threshold"])
MODEL_NAME = metadata["model_name"]


def predict_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        raise ValueError("The input contains no transactions.")

    if "Class" in data.columns:
        raise ValueError(
            "'Class' is the target label, not an input feature."
        )

    missing_columns = set(EXPECTED_FEATURES) - set(data.columns)
    extra_columns = set(data.columns) - set(EXPECTED_FEATURES)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if extra_columns:
        raise ValueError(
            f"Unexpected columns: {sorted(extra_columns)}"
        )

    features = data[EXPECTED_FEATURES].copy()

    for column in EXPECTED_FEATURES:
        features[column] = pd.to_numeric(
            features[column],
            errors="raise",
        )

    values = features.to_numpy(dtype=float)

    if np.isinf(values).any():
        raise ValueError("Input contains infinite values.")

    # Missing numerical values can be handled by the fitted imputer.
    scores = model.predict_proba(features)[:, 1]

    predictions = np.where(
        scores >= THRESHOLD,
        "fraud",
        "legitimate",
    )

    return pd.DataFrame({
        "risk_score": scores,
        "prediction": predictions,
    })