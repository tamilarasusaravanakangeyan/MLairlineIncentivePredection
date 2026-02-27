from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DashboardConfig:
    db_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/airline_incentives",
    )
    api_base_url: str = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


@st.cache_resource(show_spinner=False)
def get_engine(db_url: str) -> Engine:
    return create_engine(db_url, pool_pre_ping=True, future=True)


@st.cache_data(ttl=120, show_spinner=False)
def fetch_tables(db_url: str) -> list[str]:
    engine = get_engine(db_url)
    query = text(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()
    return [str(row[0]) for row in rows]


@st.cache_data(ttl=90, show_spinner=False)
def fetch_table_data(db_url: str, table_name: str, row_limit: int) -> pd.DataFrame:
    engine = get_engine(db_url)
    sql = text(f'SELECT * FROM "{table_name}" LIMIT :row_limit')
    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn, params={"row_limit": row_limit})
    return df


@st.cache_data(ttl=90, show_spinner=False)
def fetch_transactions_monthly(db_url: str) -> pd.DataFrame:
    engine = get_engine(db_url)
    sql = text(
        """
        SELECT
            date_trunc('month', t.transaction_date)::date AS month,
            t.agency_id,
            COUNT(*)::int AS txn_count,
            COALESCE(SUM(t.ticket_count), 0)::numeric AS tickets,
            COALESCE(SUM(t.revenue), 0)::numeric AS revenue,
            COALESCE(SUM(t.commission_earned), 0)::numeric AS commission_earned
        FROM transactions t
        GROUP BY date_trunc('month', t.transaction_date)::date, t.agency_id
        ORDER BY month, t.agency_id;
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn, parse_dates=["month"])
    return df


def _parse_agency_ids(raw: str) -> list[int]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("Please provide at least one agency id, for example: 1,2,3")
    return sorted(set(int(value) for value in values))


def _call_api(base_url: str, path: str, params: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        response = requests.get(url, params=params, timeout=20)
    except Exception as exc:  # pragma: no cover
        logger.exception("API call failed for %s", url)
        return None, str(exc)

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        return None, f"{response.status_code}: {detail}"

    return response.json(), None


def _get_model_outputs(api_base_url: str, agency_id: int, as_of_date: str) -> dict[str, Any]:
    params = {"as_of_date": as_of_date}
    pred, pred_error = _call_api(api_base_url, f"/predict/{agency_id}", params)
    rev, rev_error = _call_api(api_base_url, f"/predict-revenue/{agency_id}", params)
    rec, rec_error = _call_api(api_base_url, f"/recommend/{agency_id}", params)

    return {
        "predict": pred,
        "predict_error": pred_error,
        "predict_revenue": rev,
        "predict_revenue_error": rev_error,
        "recommend": rec,
        "recommend_error": rec_error,
    }


def _render_table_explorer(db_url: str) -> None:
    st.subheader("Database Table Explorer")

    try:
        tables = fetch_tables(db_url)
    except Exception as exc:
        st.error(f"Failed to fetch tables: {exc}")
        return

    if not tables:
        st.warning("No tables found in public schema.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        selected_table = st.selectbox("Select table", options=tables, index=0)
    with col2:
        row_limit = st.number_input("Row limit", min_value=10, max_value=5000, value=200, step=10)

    try:
        table_df = fetch_table_data(db_url, selected_table, int(row_limit))
    except Exception as exc:
        st.error(f"Failed to fetch table data: {exc}")
        return

    st.markdown(f"**Rows loaded from `{selected_table}`: {len(table_df)}**")
    st.dataframe(table_df, use_container_width=True)

    if table_df.empty:
        return

    numeric_cols = table_df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        st.markdown("**Numeric summary**")
        st.dataframe(table_df[numeric_cols].describe().transpose(), use_container_width=True)

    st.markdown("**Table chart view**")
    x_options = table_df.columns.tolist()
    y_options = numeric_cols

    if not y_options:
        st.info("No numeric columns available for charting in this table.")
        return

    cx1, cx2, cx3 = st.columns(3)
    with cx1:
        x_col = st.selectbox("X-axis column", options=x_options, key="db_x_col")
    with cx2:
        y_col = st.selectbox("Y-axis column", options=y_options, key="db_y_col")
    with cx3:
        chart_type = st.selectbox("Chart type", options=["line", "bar", "scatter"], key="db_chart_type")

    plot_df = table_df[[x_col, y_col]].dropna().copy()
    if plot_df.empty:
        st.info("No non-null values for selected chart columns.")
        return

    plot_df = plot_df.rename(columns={x_col: "x", y_col: "y"})
    if chart_type == "line":
        st.line_chart(plot_df, x="x", y="y")
    elif chart_type == "bar":
        st.bar_chart(plot_df, x="x", y="y")
    else:
        st.scatter_chart(plot_df, x="x", y="y")


def _render_model_comparison(db_url: str, api_base_url: str, as_of_date_value: date, agency_ids_raw: str) -> None:
    st.subheader("Model Output Comparison vs Database History")

    try:
        agency_ids = _parse_agency_ids(agency_ids_raw)
    except ValueError as exc:
        st.error(str(exc))
        return

    as_of_date_str = as_of_date_value.isoformat()

    try:
        monthly_df = fetch_transactions_monthly(db_url)
    except Exception as exc:
        st.error(f"Failed to query transaction history: {exc}")
        return

    compare_rows: list[dict[str, Any]] = []
    raw_payloads: dict[int, dict[str, Any]] = {}

    for agency_id in agency_ids:
        payload = _get_model_outputs(api_base_url=api_base_url, agency_id=agency_id, as_of_date=as_of_date_str)
        raw_payloads[agency_id] = payload

        agency_hist = monthly_df[monthly_df["agency_id"] == agency_id].sort_values("month")
        latest_actual_revenue = float(agency_hist["revenue"].iloc[-1]) if not agency_hist.empty else None
        trailing_3m_avg_revenue = (
            float(agency_hist["revenue"].tail(3).mean()) if not agency_hist.empty else None
        )
        annualized_from_3m = trailing_3m_avg_revenue * 12.0 if trailing_3m_avg_revenue is not None else None

        predicted_revenue = None
        if payload.get("predict_revenue"):
            predicted_revenue = float(payload["predict_revenue"].get("predicted_revenue_target", 0.0))

        compare_rows.append(
            {
                "agency_id": agency_id,
                "as_of_date": as_of_date_str,
                "predicted_tier_id": payload.get("predict", {}).get("predicted_tier_id") if payload.get("predict") else None,
                "confidence_score": payload.get("predict", {}).get("confidence_score") if payload.get("predict") else None,
                "predicted_revenue_target": predicted_revenue,
                "latest_actual_monthly_revenue": latest_actual_revenue,
                "trailing_3m_avg_monthly_revenue": trailing_3m_avg_revenue,
                "annualized_revenue_from_3m_avg": annualized_from_3m,
                "recommended_program": payload.get("recommend", {}).get("program_name") if payload.get("recommend") else None,
                "recommended_tier": payload.get("recommend", {}).get("tier_name") if payload.get("recommend") else None,
                "recommendation_commission_percentage": payload.get("recommend", {}).get("commission_percentage") if payload.get("recommend") else None,
                "predict_error": payload.get("predict_error"),
                "predict_revenue_error": payload.get("predict_revenue_error"),
                "recommend_error": payload.get("recommend_error"),
            }
        )

    compare_df = pd.DataFrame(compare_rows)

    st.markdown("**Comparison table**")
    st.dataframe(compare_df, use_container_width=True)

    chart_cols = [
        "predicted_revenue_target",
        "latest_actual_monthly_revenue",
        "trailing_3m_avg_monthly_revenue",
    ]
    available_chart_cols = [column for column in chart_cols if column in compare_df.columns]
    chart_df = compare_df[["agency_id", *available_chart_cols]].copy()
    chart_df = chart_df.dropna(how="all", subset=available_chart_cols)

    if not chart_df.empty:
        st.markdown("**Revenue comparison chart**")
        st.bar_chart(chart_df.set_index("agency_id"))

    st.markdown("**Raw API payloads**")
    for agency_id in agency_ids:
        with st.expander(f"Agency {agency_id}"):
            st.json(raw_payloads[agency_id])


def main() -> None:
    st.set_page_config(page_title="DB + Model Comparison Dashboard", layout="wide")
    config = DashboardConfig()

    st.title("PostgreSQL Data Explorer and Model Comparison")
    st.caption(
        "Explore all database tables and compare API model outputs against historical transaction aggregates."
    )

    with st.sidebar:
        st.header("Configuration")
        db_url = st.text_input("PostgreSQL URL", value=config.db_url)
        api_base_url = st.text_input("API base URL", value=config.api_base_url)
        as_of_date_value = st.date_input("As of date", value=date.today())
        agency_ids_raw = st.text_input("Agency IDs (comma-separated)", value="1,2,3")

    table_tab, compare_tab = st.tabs(["Database", "Compare Output"])

    with table_tab:
        _render_table_explorer(db_url=db_url)

    with compare_tab:
        _render_model_comparison(
            db_url=db_url,
            api_base_url=api_base_url,
            as_of_date_value=as_of_date_value,
            agency_ids_raw=agency_ids_raw,
        )


if __name__ == "__main__":
    main()
