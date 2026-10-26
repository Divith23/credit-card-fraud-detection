import pandas as pd

from fastapi.testclient import TestClient

from app.main import app
from src.predict import EXPECTED_FEATURES


client = TestClient(app)


def make_sample_transaction():
    return {
        column: 0.0
        for column in EXPECTED_FEATURES
    }


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_single_prediction():
    response = client.post(
        "/predict",
        json=make_sample_transaction(),
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in {
        "fraud",
        "legitimate",
    }

    assert 0 <= result["risk_score"] <= 1


def test_missing_feature():
    sample = make_sample_transaction()
    sample.pop(EXPECTED_FEATURES[0])

    response = client.post(
        "/predict",
        json=sample,
    )

    assert response.status_code == 422


def test_batch_prediction():
    data = pd.DataFrame([
        make_sample_transaction(),
        make_sample_transaction(),
    ])

    files = {
        "file": (
            "sample.csv",
            data.to_csv(index=False),
            "text/csv",
        )
    }

    response = client.post(
        "/predict-file",
        files=files,
    )

    assert response.status_code == 200
    assert response.json()["rows_processed"] == 2