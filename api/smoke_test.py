from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api import main


class DummyModel:
    def predict_proba(self, x_frame: pd.DataFrame) -> np.ndarray:
        return np.array([[0.2, 0.8]], dtype=float)


def fake_extract_agency_frame(engine: object, agency_id: int, as_of_date: str) -> pd.DataFrame:
    row: dict[str, object] = {
        "agency_id": agency_id,
        "snapshot_month": pd.Timestamp("2026-02-01"),
    }
    for feature_name in main.FEATURE_COLUMNS:
        row[feature_name] = 1.0
    return pd.DataFrame([row])


def run_smoke_test() -> None:
    main.state.engine = object()
    main.state.model = DummyModel()
    main.state.feature_columns = list(main.FEATURE_COLUMNS)
    main.state.class_to_tier = {0: 1, 1: 2}
    main._extract_agency_frame = fake_extract_agency_frame  # type: ignore[assignment]

    client = TestClient(main.app)

    health_response = client.get("/health")
    assert health_response.status_code == 200, health_response.text
    assert health_response.json() == {"status": "ok"}

    prediction_response = client.get("/predict/1", params={"as_of_date": "2026-02-01"})
    assert prediction_response.status_code == 200, prediction_response.text

    payload = prediction_response.json()
    assert payload["agency_id"] == 1
    assert payload["as_of_date"] == "2026-02-01"
    assert payload["predicted_tier_id"] == 2
    assert 0.0 <= payload["confidence_score"] <= 1.0

    print("API smoke test passed.")


if __name__ == "__main__":
    run_smoke_test()
