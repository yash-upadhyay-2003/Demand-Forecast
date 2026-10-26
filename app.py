"""
Daily Sales Forecasting — Web Application (Frontend)

Client interface connecting to the FastAPI backend service (Render or local).
Preserves the exact design, typography, date cards, KPI displays, Plotly charts,
data table, and 1-click CSV download.
"""

import os
import io
import warnings
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore
import requests
import streamlit as st

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Daily Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------
# Backend URL Configuration
# ---------------------------------------------------------
BACKEND_URL = os.getenv("BACKEND_URL")
if not BACKEND_URL:
    try:
        BACKEND_URL = st.secrets.get("BACKEND_URL")
    except Exception:
        pass
if not BACKEND_URL:
    BACKEND_URL = "http://127.0.0.1:8000"

BACKEND_URL = BACKEND_URL.rstrip("/")

# ---------------------------------------------------------
# Premium, Clean CSS Styling (Light Theme)
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
/* Root fonts and background */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #0F172A;
}

/* Background refinement */
.main {
    background-color: #F8FAFC;
}

/* Header typography */
.app-header {
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #E2E8F0;
}
.app-title {
    font-size: 1.85rem;
    font-weight: 700;
    color: #0F172A;
    letter-spacing: -0.025em;
    margin-bottom: 0.25rem;
}
.app-subtitle {
    font-size: 0.95rem;
    color: #64748B;
    margin: 0;
    font-weight: 400;
}

/* Date info card banner */
.date-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1.75rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}
.date-item {
    display: flex;
    flex-direction: column;
}
.date-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748B;
    margin-bottom: 0.3rem;
}
.date-value {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0F172A;
}
.date-value.highlight {
    color: #2563EB;
}

/* KPI cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.75rem;
}
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1.25rem 1.4rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}
.kpi-title {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #64748B;
    margin-bottom: 0.4rem;
}
.kpi-number {
    font-size: 1.75rem;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}
.kpi-subtext {
    font-size: 0.8rem;
    color: #64748B;
}

/* Section cards */
.content-box {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1.4rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}

/* Streamlit button and table styling tweaks */
div.stButton > button:first-child {
    background-color: #2563EB;
    color: white;
    font-weight: 600;
    border-radius: 6px;
    border: none;
    padding: 0.5rem 1.2rem;
    transition: all 0.15s ease-in-out;
}
div.stButton > button:first-child:hover {
    background-color: #1D4ED8;
    color: white;
}
div.stDownloadButton > button:first-child {
    background-color: #0F172A;
    color: white;
    font-weight: 600;
    border-radius: 6px;
    border: none;
    padding: 0.5rem 1.2rem;
    transition: all 0.15s ease-in-out;
}
div.stDownloadButton > button:first-child:hover {
    background-color: #1E293B;
    color: white;
}

/* Radio buttons container */
div[data-testid="stRadio"] > div {
    gap: 1.5rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Formatting Utilities
# ---------------------------------------------------------
def format_date(dt: pd.Timestamp) -> str:
    """Format dates consistently as DD Mon YYYY (e.g. 30 Jun 2014)."""
    if pd.isna(dt):
        return "—"
    return dt.strftime("%d %b %Y")


def format_number(val: float, decimals: int = 2) -> str:
    """Format numbers with thousand separators and fixed decimal places."""
    if pd.isna(val) or not np.isfinite(val):
        return "—"
    return f"{val:,.{decimals}f}"


# ---------------------------------------------------------
# Backend API Calls (Cached for Fast Responsive UI)
# ---------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def fetch_metadata(url: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Fetch dataset boundaries from backend GET /metadata."""
    try:
        resp = requests.get(f"{url}/metadata", timeout=30)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.Timeout:
        return None, "Connection timed out while reaching the backend service."
    except requests.exceptions.ConnectionError:
        return None, f"Could not connect to backend at {url}. Service may be starting up."
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_history(url: str, window_choice: str) -> Tuple[pd.Series, Optional[str]]:
    """Fetch historical daily sales from backend GET /history."""
    mapping = {
        "Recent 90 Days": "90",
        "Recent 180 Days": "180",
        "Recent 365 Days": "365",
        "Full History": "full",
    }
    days_val = mapping.get(window_choice, "90")
    try:
        resp = requests.get(f"{url}/history", params={"days": days_val}, timeout=30)
        resp.raise_for_status()
        records = resp.json().get("records", [])
        if not records:
            return pd.Series(dtype=float), None
        df_hist = pd.DataFrame(records)
        df_hist["date"] = pd.to_datetime(df_hist["date"])
        df_hist.set_index("date", inplace=True)
        return df_hist["sales"].astype(float), None
    except Exception as e:
        return pd.Series(dtype=float), str(e)


def request_forecast(url: str, horizon_days: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Call backend POST /forecast."""
    try:
        resp = requests.post(f"{url}/forecast", json={"horizon_days": horizon_days}, timeout=45)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.Timeout:
        return None, "Forecast computation timed out. Please retry."
    except requests.exceptions.ConnectionError:
        return None, f"Connection to {url} lost. Service may be sleeping or restarting."
    except requests.exceptions.HTTPError as e:
        detail = "Forecast request error"
        try:
            detail = resp.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        return None, detail
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1 class="app-title">Daily Sales Forecasting</h1>
        <p class="app-subtitle">Forecast future daily sales using historical transaction data and a validated two-stage machine learning model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Fetch Metadata from Backend
metadata, meta_error = fetch_metadata(BACKEND_URL)
if meta_error or metadata is None:
    st.error(
        f"⚠️ **Backend Connection Unavailable**\n\n"
        f"Unable to reach the forecasting backend service at `{BACKEND_URL}`.\n\n"
        f"**Error Details:** {meta_error}\n\n"
        f"*Note for Render deployments:* If hosted on Render free tier, web services automatically sleep after inactivity "
        f"and can take 30–60 seconds to spin up. Please wait a moment and click **Retry Connection** below."
    )
    if st.button("🔄 Retry Connection", type="primary"):
        st.cache_data.clear()
        st.rerun()
    st.stop()

dataset_start_date = pd.to_datetime(metadata["dataset_start_date"])
dataset_end_date = pd.to_datetime(metadata["dataset_end_date"])
forecast_start_date = pd.to_datetime(metadata["forecast_starts_on"])


# ---------------------------------------------------------
# Horizon Selector & Predict Button
# ---------------------------------------------------------
col_horizon, col_btn, col_history_view = st.columns([1.5, 0.9, 1.2])

with col_horizon:
    horizon_choice = st.radio(
        "Select Forecast Horizon:",
        options=["30 Days", "60 Days", "90 Days"],
        index=0,
        horizontal=True,
        help="Choose how many future daily sales values to forecast.",
    )

horizon_days = int(horizon_choice.split()[0])
forecast_end_date = forecast_start_date + pd.Timedelta(days=horizon_days - 1)

with col_btn:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    predict_btn = st.button(
        "Predict Forecast",
        type="primary",
        use_container_width=True,
        help="Click to generate daily sales forecast from the backend API for the selected horizon.",
    )

with col_history_view:
    history_window = st.selectbox(
        "Historical Context to Display in Chart:",
        options=["Recent 90 Days", "Recent 180 Days", "Recent 365 Days", "Full History"],
        index=0,
        help="Adjust how much past sales data is shown alongside the forecast for context.",
    )


# ---------------------------------------------------------
# Prominent Dataset & Forecast Date Information (Required)
# ---------------------------------------------------------
st.markdown(
    f"""
    <div class="date-strip">
        <div class="date-item">
            <span class="date-label">Dataset Start Date</span>
            <span class="date-value">{format_date(dataset_start_date)}</span>
        </div>
        <div class="date-item">
            <span class="date-label">Dataset End Date</span>
            <span class="date-value">{format_date(dataset_end_date)}</span>
        </div>
        <div class="date-item">
            <span class="date-label">Forecast Starts On</span>
            <span class="date-value highlight">{format_date(forecast_start_date)}</span>
        </div>
        <div class="date-item">
            <span class="date-label">Forecast Ends On</span>
            <span class="date-value highlight">{format_date(forecast_end_date)}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Forecast Execution & Session State Management
# ---------------------------------------------------------
if "forecast_df" not in st.session_state:
    st.session_state["forecast_df"] = None
    st.session_state["forecast_kpis"] = None
    st.session_state["forecast_horizon"] = None
    st.session_state["forecast_choice"] = None

if predict_btn:
    with st.spinner(f"Requesting {horizon_choice} forecast from backend API..."):
        fc_payload, fc_err = request_forecast(BACKEND_URL, horizon_days)
        if fc_err or fc_payload is None:
            st.error(f"Error generating forecast from API: {fc_err}")
            st.stop()

        # Parse forecast data into DataFrame
        records = fc_payload.get("forecast", [])
        df_parsed = pd.DataFrame(records)
        df_parsed["date"] = pd.to_datetime(df_parsed["date"])
        df_parsed.set_index("date", inplace=True)
        df_parsed.rename(
            columns={
                "predicted_sales": "Predicted_Sales",
                "probability_of_sale": "Probability_of_Sale",
                "median_component": "Median_Component",
                "expected_component": "Expected_Component",
            },
            inplace=True,
        )

        st.session_state["forecast_df"] = df_parsed
        st.session_state["forecast_kpis"] = fc_payload.get("kpis", {})
        st.session_state["forecast_horizon"] = horizon_days
        st.session_state["forecast_choice"] = horizon_choice

if st.session_state["forecast_df"] is None:
    st.markdown(
        """
        <div class="content-box" style="text-align: center; padding: 3.5rem 1.5rem; margin-top: 1rem;">
            <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">📈</div>
            <h3 style="font-size: 1.25rem; font-weight: 700; color: #0F172A; margin-bottom: 0.4rem;">Forecast Ready to Generate</h3>
            <p style="font-size: 0.95rem; color: #64748B; max-width: 520px; margin: 0 auto;">
                Select your desired horizon (30, 60, or 90 Days) above and click <strong>Predict Forecast</strong> to calculate projected daily sales, KPIs, and interactive charts via the backend API.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

forecast_df = st.session_state["forecast_df"]
kpis = st.session_state["forecast_kpis"]
active_horizon = st.session_state["forecast_horizon"]
active_choice = st.session_state["forecast_choice"]

if active_horizon != horizon_days:
    st.info(
        f"Horizon selection changed to **{horizon_choice}**. Click **Predict Forecast** to update results."
    )


# ---------------------------------------------------------
# Primary Forecast Results (KPI Cards)
# ---------------------------------------------------------
kpi_max_date = pd.to_datetime(kpis.get("highest_sales_day_date"))
kpi_start_date = pd.to_datetime(kpis.get("start_date"))
kpi_end_date = pd.to_datetime(kpis.get("end_date"))

st.markdown(
    f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">Total Forecast Sales</div>
            <div class="kpi-number">{format_number(kpis.get('total_sales', 0.0))}</div>
            <div class="kpi-subtext">Sum of predicted daily sales for {active_choice}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Average Daily Sales</div>
            <div class="kpi-number">{format_number(kpis.get('average_daily_sales', 0.0))}</div>
            <div class="kpi-subtext">Mean predicted sales per day over horizon</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Forecast Period</div>
            <div class="kpi-number">{active_horizon} Days</div>
            <div class="kpi-subtext">{format_date(kpi_start_date)} to {format_date(kpi_end_date)}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Highest Predicted Day</div>
            <div class="kpi-number">{format_number(kpis.get('highest_sales_day_value', 0.0))}</div>
            <div class="kpi-subtext">Expected on {format_date(kpi_max_date)}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Chart 1: Historical Sales and Future Forecast
# ---------------------------------------------------------
st.markdown("### Historical Sales & Future Forecast")

hist_slice, hist_err = fetch_history(BACKEND_URL, history_window)
if hist_err:
    st.warning(f"Unable to load historical context from API: {hist_err}")

fig = go.Figure()

# Historical Actual Sales trace
if not hist_slice.empty:
    fig.add_trace(
        go.Scatter(
            x=hist_slice.index,
            y=hist_slice.values,
            mode="lines",
            name="Historical Actual Sales",
            line=dict(color="#475569", width=1.5),
            hovertemplate="<b>Date:</b> %{x|%d %b %Y}<br><b>Actual Sales:</b> %{y:,.2f}<extra></extra>",
        )
    )

# Forecasted Daily Sales trace
fig.add_trace(
    go.Scatter(
        x=forecast_df.index,
        y=forecast_df["Predicted_Sales"],
        mode="lines+markers",
        name=f"Forecasted Daily Sales ({active_choice})",
        line=dict(color="#2563EB", width=2.5),
        marker=dict(size=4, color="#2563EB"),
        hovertemplate="<b>Forecast Date:</b> %{x|%d %b %Y}<br><b>Predicted Sales:</b> %{y:,.2f}<extra></extra>",
    )
)

# Vertical line separating historical data and forecast
fig.add_vline(
    x=forecast_start_date - pd.Timedelta(hours=12),
    line_width=1.5,
    line_dash="dash",
    line_color="#94A3B8",
)

fig.add_annotation(
    x=forecast_start_date,
    y=1.03,
    yref="paper",
    text="Forecast Starts",
    showarrow=False,
    font=dict(size=11, color="#2563EB", family="sans-serif"),
    bgcolor="#EFF6FF",
    bordercolor="#BFDBFE",
    borderwidth=1,
    borderpad=3,
)

fig.update_layout(
    height=420,
    margin=dict(l=20, r=20, t=30, b=20),
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255,255,255,0.8)",
    ),
    xaxis=dict(
        showgrid=True,
        gridcolor="#F1F5F9",
        linecolor="#E2E8F0",
        tickfont=dict(color="#64748B", size=11),
    ),
    yaxis=dict(
        title="Daily Sales",
        showgrid=True,
        gridcolor="#F1F5F9",
        linecolor="#E2E8F0",
        tickfont=dict(color="#64748B", size=11),
        title_font=dict(color="#475569", size=12),
    ),
)

st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# Optional Chart 2 & Forecast Table Layout
# ---------------------------------------------------------
col_weekly_chart, col_table = st.columns([1.2, 1.0])

with col_weekly_chart:
    st.markdown("### Weekly Forecast Aggregation")
    
    weekly_forecast = forecast_df["Predicted_Sales"].resample("W-SUN").sum().to_frame()
    weekly_forecast.columns = ["Weekly_Sales"]
    weekly_forecast["Week_Label"] = [
        f"Week ending {d.strftime('%d %b')}" for d in weekly_forecast.index
    ]

    fig_weekly = go.Figure()
    fig_weekly.add_trace(
        go.Bar(
            x=weekly_forecast["Week_Label"],
            y=weekly_forecast["Weekly_Sales"],
            marker_color="#3B82F6",
            hovertemplate="<b>%{x}</b><br>Total Forecast: %{y:,.2f}<extra></extra>",
        )
    )

    fig_weekly.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=50),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        xaxis=dict(
            tickangle=-25,
            tickfont=dict(color="#64748B", size=10),
            showgrid=False,
        ),
        yaxis=dict(
            title="Weekly Total Sales",
            showgrid=True,
            gridcolor="#F1F5F9",
            tickfont=dict(color="#64748B", size=11),
            title_font=dict(color="#475569", size=12),
        ),
    )
    st.plotly_chart(fig_weekly, use_container_width=True)


with col_table:
    st.markdown("### Forecast Data Table")

    display_table = pd.DataFrame(
        {
            "Forecast Date": [format_date(d) for d in forecast_df.index],
            "Predicted Daily Sales": [format_number(v) for v in forecast_df["Predicted_Sales"]],
        }
    )

    st.dataframe(
        display_table,
        use_container_width=True,
        height=300,
        hide_index=True,
    )

    # Clean Download CSV Button
    csv_buffer = io.StringIO()
    export_df = pd.DataFrame(
        {
            "Forecast_Date": forecast_df.index.strftime("%Y-%m-%d"),
            "Predicted_Daily_Sales": forecast_df["Predicted_Sales"].round(2),
        }
    )
    export_df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    st.download_button(
        label=f"⬇ Download Forecast CSV ({active_choice})",
        data=csv_bytes,
        file_name=f"sales_forecast_{active_horizon}d_{forecast_start_date.strftime('%Y%m%d')}.csv",
        mime="text/csv",
        help="Download clean forecast dates and predictions as a CSV file.",
    )


# ---------------------------------------------------------
# Footer / About Section
# ---------------------------------------------------------
with st.expander("About This Forecasting Model & Methodology", expanded=False):
    st.markdown(
        """
        - **Pipeline Architecture:** Direct multi-horizon two-stage model.
          - **Stage 1 (Occurrence):** Logistic Regression estimates probability of non-zero daily sales $P(\\text{Sales} > 0)$.
          - **Stage 2 (Amount):** Ridge Regression on $\\log(1 + \\text{Sales})$ predicting conditional sales amount.
          - **Blend Decision Rule:** Blends median point forecast with Duan-smeared expected sales ($\\lambda = 0.3$) to balance day-level WAPE with aggregate sales volume.
        - **Zero-Sales Calendar Integrity:** All non-recorded dates are treated as confirmed zero-sales calendar days.
        - **Data Privacy:** Raw transactions are aggregated locally; no customer records or sensitive identifiers are stored, logged, or transmitted externally.
        """
    )
