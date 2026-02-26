from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
import requests
import streamlit as st

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DashboardConfig:
    api_base_url: str = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def _parse_agency_ids(raw: str) -> list[int]:
    ids: list[int] = []
    for value in raw.split(","):
        value = value.strip()
        if not value:
            continue
        ids.append(int(value))
    unique_ids = sorted(set(ids))
    if not unique_ids:
        raise ValueError("Provide at least one agency id (for example: 1,2,3).")
    return unique_ids


def _call_get(base_url: str, path: str, params: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        response = requests.get(url, params=params, timeout=20)
    except Exception as exc:  # pragma: no cover - network/runtime dependent
        logger.exception("Request failed for %s", url)
        return None, str(exc)

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        return None, f"{response.status_code}: {detail}"

    return response.json(), None


def _collect_row(base_url: str, agency_id: int, as_of_date: str) -> tuple[dict[str, Any], dict[str, Any]]:
    params = {"as_of_date": as_of_date}

    predict, predict_error = _call_get(base_url, f"/predict/{agency_id}", params)
    revenue, revenue_error = _call_get(base_url, f"/predict-revenue/{agency_id}", params)
    recommend, recommend_error = _call_get(base_url, f"/recommend/{agency_id}", params)

    row: dict[str, Any] = {
        "agency_id": agency_id,
        "as_of_date": as_of_date,
        "predicted_tier_id": predict.get("predicted_tier_id") if predict else None,
        "confidence_score": predict.get("confidence_score") if predict else None,
        "predicted_revenue_target": revenue.get("predicted_revenue_target") if revenue else None,
        "program_id": recommend.get("program_id") if recommend else None,
        "program_name": recommend.get("program_name") if recommend else None,
        "tier_id": recommend.get("tier_id") if recommend else None,
        "tier_name": recommend.get("tier_name") if recommend else None,
        "tier_level": recommend.get("tier_level") if recommend else None,
        "commission_percentage": recommend.get("commission_percentage") if recommend else None,
    }

    errors = {
        "predict_error": predict_error,
        "revenue_error": revenue_error,
        "recommend_error": recommend_error,
    }

    return row, {
        "predict": predict,
        "predict_revenue": revenue,
        "recommend": recommend,
        "errors": errors,
    }


def _render_metric_charts(results: pd.DataFrame) -> None:
    numeric_columns = [
        "confidence_score",
        "predicted_revenue_target",
        "commission_percentage",
    ]

    available_numeric = [column for column in numeric_columns if column in results.columns]
    if not available_numeric:
        st.info("No numeric values available for chart visualization.")
        return

    chart_df = results[["agency_id", *available_numeric]].copy()
    chart_df = chart_df.dropna(how="all", subset=available_numeric)

    if chart_df.empty:
        st.info("No successful predictions to visualize yet.")
        return

    st.subheader("Charts")
    for metric in available_numeric:
        st.markdown(f"**{metric} by agency**")
        metric_df = chart_df[["agency_id", metric]].dropna(subset=[metric]).set_index("agency_id")
        if not metric_df.empty:
            st.bar_chart(metric_df)
        else:
            st.caption(f"No values available for {metric}.")


def main() -> None:
    st.set_page_config(page_title="Incentive Model Dashboard", layout="wide")

    config = DashboardConfig()
    st.title("Incentive Model Outcome Dashboard")
    st.caption("Runs Option A, Option B, and Option C through the FastAPI service and visualizes outcomes.")

    with st.sidebar:
        st.header("Run Settings")
        api_base_url = st.text_input("API base URL", value=config.api_base_url)
        as_of_date = st.date_input("As of date", value=date.today())
        agency_ids_raw = st.text_input("Agency IDs (comma-separated)", value="1")
        run_button = st.button("Run predictions", type="primary")

    if not run_button:
        st.info("Configure inputs and click 'Run predictions' to load tables and charts.")
        return

    try:
        agency_ids = _parse_agency_ids(agency_ids_raw)
    except ValueError as exc:
        st.error(str(exc))
        return

    records: list[dict[str, Any]] = []
    payloads: dict[int, dict[str, Any]] = {}

    with st.spinner("Calling API endpoints..."):
        for agency_id in agency_ids:
            row, payload = _collect_row(api_base_url, agency_id=agency_id, as_of_date=as_of_date.isoformat())
            records.append(row)
            payloads[agency_id] = payload

    result_df = pd.DataFrame(records)

    st.subheader("Prediction Table")
    st.dataframe(result_df, use_container_width=True)

    failed_rows = result_df[
        result_df[["predicted_tier_id", "predicted_revenue_target", "program_id"]].isna().all(axis=1)
    ]
    if not failed_rows.empty:
        st.warning("Some agencies returned no complete prediction data. Expand raw payloads for endpoint-level errors.")

    _render_metric_charts(result_df)

    st.subheader("Raw API Payloads")
    for agency_id in agency_ids:
        with st.expander(f"Agency {agency_id}"):
            st.json(payloads[agency_id])


if __name__ == "__main__":
    main()
