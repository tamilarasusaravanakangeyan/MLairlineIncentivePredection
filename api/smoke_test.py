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


class DummyRevenueModel:
    def predict(self, x_frame: pd.DataFrame) -> np.ndarray:
        return np.array([250000.0], dtype=float)


def fake_extract_agency_frame(engine: object, agency_id: int, as_of_date: str) -> pd.DataFrame:
    row: dict[str, object] = {
        "agency_id": agency_id,
        "snapshot_month": pd.Timestamp("2026-02-01"),
    }
    for feature_name in main.FEATURE_COLUMNS:
        row[feature_name] = 1.0
    return pd.DataFrame([row])


def fake_extract_tier_catalog(engine: object) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "program_id": 1,
                "program_name": "Core Incentive Program",
                "tier_id": 1,
                "tier_name": "Silver",
                "tier_level": 1,
                "min_annual_revenue": 0.0,
                "max_annual_revenue": 500000.0,
                "commission_percentage": 2.0,
            },
            {
                "program_id": 2,
                "program_name": "Growth Incentive Program",
                "tier_id": 2,
                "tier_name": "Gold",
                "tier_level": 2,
                "min_annual_revenue": 500000.0,
                "max_annual_revenue": 2000000.0,
                "commission_percentage": 4.5,
            },
        ]
    )


def run_smoke_test() -> None:
    main.state.engine = object()
    main.state.model = DummyModel()
    main.state.revenue_model = DummyRevenueModel()
    main.state.feature_columns = list(main.FEATURE_COLUMNS)
    main.state.class_to_tier = {0: 1, 1: 2}
    main._extract_agency_frame = fake_extract_agency_frame  # type: ignore[assignment]
    main._extract_tier_catalog = fake_extract_tier_catalog  # type: ignore[assignment]

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

    revenue_response = client.get("/predict-revenue/1", params={"as_of_date": "2026-02-01"})
    assert revenue_response.status_code == 200, revenue_response.text
    revenue_payload = revenue_response.json()
    assert revenue_payload["agency_id"] == 1
    assert revenue_payload["predicted_revenue_target"] > 0

    recommendation_response = client.get("/recommend/1", params={"as_of_date": "2026-02-01"})
    assert recommendation_response.status_code == 200, recommendation_response.text
    recommendation_payload = recommendation_response.json()
    assert recommendation_payload["agency_id"] == 1
    assert recommendation_payload["program_id"] in {1, 2}
    assert recommendation_payload["tier_id"] in {1, 2}
    assert recommendation_payload["commission_percentage"] >= 0
    assert recommendation_payload["predicted_revenue_target"] > 0

    print("API smoke test passed.")


if __name__ == "__main__":
    run_smoke_test()
