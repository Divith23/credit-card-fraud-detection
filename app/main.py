import io

import numpy as np
import pandas as pd

from fastapi import FastAPI, File, HTTPException, UploadFile
from src.predict import (
    EXPECTED_FEATURES,
    MODEL_NAME,
    THRESHOLD,
    predict_dataframe,
)


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Predicts fraud risk from anonymized transaction features.",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": True,
        "model_name": MODEL_NAME,
    }


@app.post("/predict")
def predict_one(payload: dict[str, float]):
    try:
        data = pd.DataFrame([payload])
        result = predict_dataframe(data).iloc[0]

        return {
            "prediction": str(result["prediction"]),
            "risk_score": float(result["risk_score"]),
            "threshold": THRESHOLD,
        }

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file.",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    # Starter-project upload limits.
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="File is too large. Maximum size is 10 MB.",
        )

    try:
        data = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded CSV.",
        ) from exc

    if data.empty:
        raise HTTPException(
            status_code=400,
            detail="The CSV contains no transactions.",
        )

    if len(data) > 50000:
        raise HTTPException(
            status_code=413,
            detail="The file contains too many rows. Maximum is 50,000.",
        )

    try:
        results = predict_dataframe(data)

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    results.insert(
        0,
        "row_number",
        np.arange(1, len(results) + 1),
    )

    return {
        "rows_processed": len(results),
        "fraud_flagged": int(
            (results["prediction"] == "fraud").sum()
        ),
        "threshold": THRESHOLD,
        "results": results.to_dict(orient="records"),
    }