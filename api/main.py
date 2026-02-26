from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    db_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@host.docker.internal:5432/airline_incentives",
    )
    artifact_dir: Path = Path(os.getenv("ARTIFACT_DIR", "notebooks/artifacts"))

    @property
    def model_path(self) -> Path:
        return self.artifact_dir / "next_tier_xgb.joblib"

    @property
    def feature_path(self) -> Path:
        return self.artifact_dir / "feature_columns.joblib"

    @property
    def label_map_path(self) -> Path:
        return self.artifact_dir / "label_mapping.joblib"

    @property
    def revenue_model_path(self) -> Path:
        return self.artifact_dir / "next_revenue_xgb_regressor.joblib"


FEATURE_COLUMNS = [
    "txn_count_month",
    "tickets_month",
    "revenue_month",
    "commission_month",
    "avg_commission_rate",
    "loyalty_points_earned_month",
    "redemption_count_month",
    "points_redeemed_month",
    "avg_performance_rating_month",
    "airline_diversity_count",
    "route_diversity_count",
    "rolling_3m_revenue",
    "rolling_6m_revenue",
    "rolling_3m_tickets",
    "frequency_txn_per_quarter",
    "revenue_lag_1m",
    "tickets_lag_1m",
    "revenue_growth_pct",
    "ticket_growth_pct",
    "loyalty_redemption_ratio",
    "monetary_avg_revenue_per_txn",
    "recency_days",
    "month_num",
    "quarter_num",
]

INFERENCE_SQL = """
WITH monthly_txn AS (
    SELECT
        t.agency_id,
        date_trunc('month', t.transaction_date)::date AS snapshot_month,
        COUNT(*)::int AS txn_count_month,
        COALESCE(SUM(t.ticket_count), 0)::numeric AS tickets_month,
        COALESCE(SUM(t.revenue), 0)::numeric AS revenue_month,
        COALESCE(SUM(t.commission_earned), 0)::numeric AS commission_month,
        COALESCE(SUM(t.loyalty_points), 0)::numeric AS loyalty_points_earned_month,
        COUNT(DISTINCT t.airline_id)::int AS airline_diversity_count,
        COUNT(DISTINCT t.route_id)::int AS route_diversity_count,
        MAX(t.transaction_date)::date AS last_txn_date
    FROM transactions t
    WHERE t.agency_id = :agency_id
      AND t.transaction_date < CAST(:as_of_date AS date) + interval '1 day'
    GROUP BY t.agency_id, date_trunc('month', t.transaction_date)::date
),
monthly_redeem AS (
    SELECT
        ir.agency_id,
        date_trunc('month', ir.redemption_date)::date AS snapshot_month,
        COUNT(*)::int AS redemption_count_month,
        COALESCE(SUM(ir.points_redeemed), 0)::numeric AS points_redeemed_month
    FROM incentive_redemptions ir
    WHERE ir.agency_id = :agency_id
      AND ir.redemption_date < CAST(:as_of_date AS date) + interval '1 day'
    GROUP BY ir.agency_id, date_trunc('month', ir.redemption_date)::date
),
monthly_perf AS (
    SELECT
        ap.agency_id,
        date_trunc('month', ap.month_year)::date AS snapshot_month,
        COALESCE(AVG(ap.performance_rating), 0)::numeric AS avg_performance_rating_month
    FROM agency_performance ap
    WHERE ap.agency_id = :agency_id
      AND ap.month_year < CAST(:as_of_date AS date) + interval '1 day'
    GROUP BY ap.agency_id, date_trunc('month', ap.month_year)::date
),
combined AS (
    SELECT
        m.agency_id, m.snapshot_month, m.txn_count_month, m.tickets_month, m.revenue_month,
        m.commission_month, m.loyalty_points_earned_month, m.airline_diversity_count,
        m.route_diversity_count, m.last_txn_date,
        CASE WHEN m.revenue_month = 0 THEN 0 ELSE m.commission_month / NULLIF(m.revenue_month, 0) END AS avg_commission_rate,
        COALESCE(r.redemption_count_month, 0) AS redemption_count_month,
        COALESCE(r.points_redeemed_month, 0) AS points_redeemed_month,
        COALESCE(p.avg_performance_rating_month, 0) AS avg_performance_rating_month
    FROM monthly_txn m
    LEFT JOIN monthly_redeem r ON m.agency_id = r.agency_id AND m.snapshot_month = r.snapshot_month
    LEFT JOIN monthly_perf p ON m.agency_id = p.agency_id AND m.snapshot_month = p.snapshot_month
),
windowed AS (
    SELECT
        c.*,
        SUM(c.revenue_month) OVER (PARTITION BY c.agency_id ORDER BY c.snapshot_month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS rolling_3m_revenue,
        SUM(c.revenue_month) OVER (PARTITION BY c.agency_id ORDER BY c.snapshot_month ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS rolling_6m_revenue,
        SUM(c.tickets_month) OVER (PARTITION BY c.agency_id ORDER BY c.snapshot_month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS rolling_3m_tickets,
        SUM(c.txn_count_month) OVER (PARTITION BY c.agency_id ORDER BY c.snapshot_month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS frequency_txn_per_quarter,
        LAG(c.revenue_month, 1) OVER (PARTITION BY c.agency_id ORDER BY c.snapshot_month) AS revenue_lag_1m,
        LAG(c.tickets_month, 1) OVER (PARTITION BY c.agency_id ORDER BY c.snapshot_month) AS tickets_lag_1m
    FROM combined c
),
growth AS (
    SELECT
        w.*,
        LAG(w.rolling_3m_revenue, 1) OVER (PARTITION BY w.agency_id ORDER BY w.snapshot_month) AS prev_rolling_3m_revenue,
        LAG(w.rolling_3m_tickets, 1) OVER (PARTITION BY w.agency_id ORDER BY w.snapshot_month) AS prev_rolling_3m_tickets
    FROM windowed w
)
SELECT
    g.agency_id,
    g.snapshot_month,
    g.txn_count_month,
    g.tickets_month,
    g.revenue_month,
    g.commission_month,
    g.avg_commission_rate,
    g.loyalty_points_earned_month,
    g.redemption_count_month,
    g.points_redeemed_month,
    g.avg_performance_rating_month,
    g.airline_diversity_count,
    g.route_diversity_count,
    g.rolling_3m_revenue,
    g.rolling_6m_revenue,
    g.rolling_3m_tickets,
    g.frequency_txn_per_quarter,
    g.revenue_lag_1m,
    g.tickets_lag_1m,
    CASE WHEN COALESCE(g.prev_rolling_3m_revenue, 0) = 0 THEN 0 ELSE (g.rolling_3m_revenue - g.prev_rolling_3m_revenue) / NULLIF(g.prev_rolling_3m_revenue, 0) END AS revenue_growth_pct,
    CASE WHEN COALESCE(g.prev_rolling_3m_tickets, 0) = 0 THEN 0 ELSE (g.rolling_3m_tickets - g.prev_rolling_3m_tickets) / NULLIF(g.prev_rolling_3m_tickets, 0) END AS ticket_growth_pct,
    CASE WHEN g.txn_count_month = 0 THEN 0 ELSE g.revenue_month / NULLIF(g.txn_count_month, 0) END AS monetary_avg_revenue_per_txn,
    CASE WHEN COALESCE(g.loyalty_points_earned_month, 0) = 0 THEN 0 ELSE COALESCE(g.points_redeemed_month, 0) / NULLIF(g.loyalty_points_earned_month, 0) END AS loyalty_redemption_ratio,
    CASE WHEN g.last_txn_date IS NULL THEN NULL ELSE (g.snapshot_month - g.last_txn_date)::int END AS recency_days,
    EXTRACT(MONTH FROM g.snapshot_month)::int AS month_num,
    EXTRACT(QUARTER FROM g.snapshot_month)::int AS quarter_num
FROM growth g
ORDER BY g.snapshot_month;
"""

TIER_CATALOG_SQL = """
SELECT
    it.program_id,
    ip.program_name,
    it.tier_id,
    it.tier_name,
    it.tier_level,
    COALESCE(it.min_annual_revenue, 0)::numeric AS min_annual_revenue,
    COALESCE(it.max_annual_revenue, 999999999999::numeric) AS max_annual_revenue,
    COALESCE(it.commission_percentage, 0)::numeric AS commission_percentage
FROM incentive_tiers it
JOIN incentive_programs ip ON it.program_id = ip.program_id
ORDER BY it.tier_level ASC, it.min_annual_revenue ASC;
"""


class PredictionResponse(BaseModel):
    agency_id: int
    as_of_date: str
    predicted_tier_id: int
    confidence_score: float = Field(ge=0.0, le=1.0)


class RevenueTargetResponse(BaseModel):
    agency_id: int
    as_of_date: str
    predicted_revenue_target: float


class RecommendationResponse(BaseModel):
    agency_id: int
    as_of_date: str
    program_id: int
    program_name: str
    tier_id: int
    tier_name: str
    tier_level: int
    commission_percentage: float
    predicted_tier_id: int
    confidence_score: float = Field(ge=0.0, le=1.0)
    predicted_revenue_target: float


class AppState:
    def __init__(self) -> None:
        self.settings = Settings()
        self.engine: Engine | None = None
        self.model: xgb.XGBClassifier | None = None
        self.revenue_model: xgb.XGBRegressor | None = None
        self.feature_columns: list[str] = []
        self.class_to_tier: dict[int, int] = {}


state = AppState()
app = FastAPI(
    title="Next Best Incentive Tier API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["agency_id", "snapshot_month"]).copy()
    for feature_name in FEATURE_COLUMNS:
        if feature_name not in out.columns:
            out[feature_name] = 0
    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)
    return out


def _extract_agency_frame(engine: Engine, agency_id: int, as_of_date: str) -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql_query(
            text(INFERENCE_SQL),
            conn,
            params={"agency_id": agency_id, "as_of_date": as_of_date},
            parse_dates=["snapshot_month"],
        )
    return df


def _extract_tier_catalog(engine: Engine) -> pd.DataFrame:
    with engine.connect() as conn:
        tier_df = pd.read_sql_query(text(TIER_CATALOG_SQL), conn)
    return tier_df


def _predict(agency_id: int, as_of_date: str) -> PredictionResponse:
    if state.engine is None or state.model is None:
        raise RuntimeError("Model service is not initialized.")

    base_df = _extract_agency_frame(state.engine, agency_id=agency_id, as_of_date=as_of_date)
    if base_df.empty:
        raise ValueError(f"No historical data for agency_id={agency_id} up to {as_of_date}")

    agency_df = _build_features(base_df)
    latest_row = agency_df.iloc[-1:]
    feature_columns = state.feature_columns or FEATURE_COLUMNS
    x_score = latest_row[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)

    proba = state.model.predict_proba(x_score)[0]
    pred_class = int(np.argmax(proba))
    confidence_score = float(np.max(proba))

    class_to_tier = {int(k): int(v) for k, v in state.class_to_tier.items()}
    predicted_tier_id = int(class_to_tier[pred_class])

    return PredictionResponse(
        agency_id=agency_id,
        as_of_date=as_of_date,
        predicted_tier_id=predicted_tier_id,
        confidence_score=confidence_score,
    )


def _predict_revenue_target(agency_id: int, as_of_date: str) -> RevenueTargetResponse:
    if state.engine is None or state.revenue_model is None:
        raise RuntimeError("Revenue target model is not initialized.")

    base_df = _extract_agency_frame(state.engine, agency_id=agency_id, as_of_date=as_of_date)
    if base_df.empty:
        raise ValueError(f"No historical data for agency_id={agency_id} up to {as_of_date}")

    agency_df = _build_features(base_df)
    latest_row = agency_df.iloc[-1:]
    feature_columns = state.feature_columns or FEATURE_COLUMNS
    x_score = latest_row[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)

    predicted_revenue_target = float(state.revenue_model.predict(x_score)[0])
    predicted_revenue_target = max(0.0, predicted_revenue_target)

    return RevenueTargetResponse(
        agency_id=agency_id,
        as_of_date=as_of_date,
        predicted_revenue_target=predicted_revenue_target,
    )


def _recommend_next_incentive(agency_id: int, as_of_date: str) -> RecommendationResponse:
    if state.engine is None:
        raise RuntimeError("Database engine is not initialized.")

    tier_prediction = _predict(agency_id=agency_id, as_of_date=as_of_date)
    revenue_prediction = _predict_revenue_target(agency_id=agency_id, as_of_date=as_of_date)

    tier_catalog = _extract_tier_catalog(state.engine)
    if tier_catalog.empty:
        raise RuntimeError("Incentive tier catalog is empty.")

    annualized_revenue = revenue_prediction.predicted_revenue_target * 4.0
    candidates = tier_catalog[
        (annualized_revenue >= tier_catalog["min_annual_revenue"])
        & (annualized_revenue < tier_catalog["max_annual_revenue"])
    ].copy()

    if candidates.empty:
        candidates = tier_catalog.copy()

    max_commission = float(candidates["commission_percentage"].max()) if len(candidates) else 1.0
    if max_commission <= 0:
        max_commission = 1.0

    candidates["commission_score"] = candidates["commission_percentage"] / max_commission
    candidates["tier_match_score"] = np.where(
        candidates["tier_id"].astype(int) == int(tier_prediction.predicted_tier_id),
        1.0,
        0.0,
    )
    candidates["recommendation_score"] = (
        0.55 * candidates["commission_score"]
        + 0.35 * candidates["tier_match_score"]
        + 0.10 * float(tier_prediction.confidence_score)
    )

    best = candidates.sort_values(
        ["recommendation_score", "tier_level", "commission_percentage"],
        ascending=[False, True, False],
    ).iloc[0]

    return RecommendationResponse(
        agency_id=agency_id,
        as_of_date=as_of_date,
        program_id=int(best["program_id"]),
        program_name=str(best["program_name"]),
        tier_id=int(best["tier_id"]),
        tier_name=str(best["tier_name"]),
        tier_level=int(best["tier_level"]),
        commission_percentage=float(best["commission_percentage"]),
        predicted_tier_id=int(tier_prediction.predicted_tier_id),
        confidence_score=float(tier_prediction.confidence_score),
        predicted_revenue_target=float(revenue_prediction.predicted_revenue_target),
    )


@app.on_event("startup")
def startup_event() -> None:
    state.engine = create_engine(state.settings.db_url, pool_pre_ping=True, future=True)

    if not state.settings.model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {state.settings.model_path}")
    if not state.settings.feature_path.exists():
        raise FileNotFoundError(f"Feature artifact not found: {state.settings.feature_path}")
    if not state.settings.label_map_path.exists():
        raise FileNotFoundError(f"Label map artifact not found: {state.settings.label_map_path}")

    state.model = joblib.load(state.settings.model_path)
    state.feature_columns = list(joblib.load(state.settings.feature_path))
    state.class_to_tier = dict(joblib.load(state.settings.label_map_path))

    if state.settings.revenue_model_path.exists():
        state.revenue_model = joblib.load(state.settings.revenue_model_path)
    else:
        state.revenue_model = None
        logger.warning(
            "Revenue model artifact not found at %s. /predict-revenue and /recommend will return 503 until generated.",
            state.settings.revenue_model_path,
        )

    logger.info("API initialized. Artifacts loaded from %s", state.settings.artifact_dir)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/predict/{agency_id}", response_model=PredictionResponse)
def predict_endpoint(agency_id: int, as_of_date: date) -> PredictionResponse:
    try:
        return _predict(agency_id=agency_id, as_of_date=as_of_date.isoformat())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/predict-revenue/{agency_id}", response_model=RevenueTargetResponse)
def predict_revenue_endpoint(agency_id: int, as_of_date: date) -> RevenueTargetResponse:
    try:
        return _predict_revenue_target(agency_id=agency_id, as_of_date=as_of_date.isoformat())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/recommend/{agency_id}", response_model=RecommendationResponse)
def recommend_endpoint(agency_id: int, as_of_date: date) -> RecommendationResponse:
    try:
        return _recommend_next_incentive(agency_id=agency_id, as_of_date=as_of_date.isoformat())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
