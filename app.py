"""
Daily Sales Forecasting — Local Web Application

A clean, modern, and professional web application for forecasting future daily sales
using historical sales data and the validated two-stage machine learning pipeline.
"""

from pathlib import Path
import io
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore
import streamlit as st

# Local modules
from forecasting_engine import (
    load_model_artifact,
    prepare_daily_sales_series,
    generate_forecast,
    compute_forecast_kpis,
)
from data_loader import resolve_default_paths, load_raw_dataset

# If executed directly with `python app.py` instead of `streamlit run app.py`,
# bootstrap Streamlit automatically so the user never gets ScriptRunContext errors.
if __name__ == "__main__":
    try:
        import streamlit as _st
        if hasattr(_st, "runtime") and not _st.runtime.exists():
            import sys
            from streamlit.web import cli as _stcli
            sys.argv = ["streamlit", "run", sys.argv[0]]
            sys.exit(_stcli.main())
    except Exception:
        pass

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
# Helper Functions
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
# Internal Data and Model Loading (No Settings UI Controls)
# ---------------------------------------------------------
default_csv, default_model = resolve_default_paths()

raw_df = None
if default_csv is not None and default_csv.exists():
    try:
        raw_df = load_raw_dataset(default_csv)
    except Exception as e:
        st.error(f"Error loading sales dataset: {e}")
else:
    st.error("Sales dataset CSV not found in project folder.")

model_artifact = None
if default_model is not None and default_model.exists():
    try:
        model_artifact = load_model_artifact(default_model)
    except Exception as e:
        st.error(f"Error loading model artifact: {e}")
else:
    st.error("Model artifact `final_sales_forecast_model.joblib` not found in project folder.")


# ---------------------------------------------------------
# Main Application Content
# ---------------------------------------------------------
# Page Title & Subtitle
st.markdown(
    """
    <div class="app-header">
        <h1 class="app-title">Daily Sales Forecasting</h1>
        <p class="app-subtitle">Forecast future daily sales using historical transaction data and a validated two-stage machine learning model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if raw_df is None or model_artifact is None:
    st.error(
        "Application cannot proceed without both a valid sales dataset and the trained model artifact. "
        "Please ensure the project files are present in the project folder."
    )
    st.stop()

# Prepare continuous daily sales series
try:
    daily_series = prepare_daily_sales_series(raw_df)
except Exception as e:
    st.error(f"Data processing error: {e}")
    st.stop()

dataset_start_date = daily_series.index.min()
dataset_end_date = daily_series.index.max()
forecast_start_date = dataset_end_date + pd.Timedelta(days=1)


# ---------------------------------------------------------
# Forecast Horizon Selector & Predict Button
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
        help="Click to generate daily sales forecast for the selected horizon.",
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
# Generate Forecast Execution (Triggered by Button)
# ---------------------------------------------------------
if "forecast_df" not in st.session_state:
    st.session_state["forecast_df"] = None
    st.session_state["forecast_kpis"] = None
    st.session_state["forecast_horizon"] = None
    st.session_state["forecast_choice"] = None

if predict_btn:
    with st.spinner(f"Generating {horizon_choice} sales forecast..."):
        try:
            forecast_df = generate_forecast(
                daily_series=daily_series,
                model_artifact=model_artifact,
                n_days=horizon_days,
            )
            kpis = compute_forecast_kpis(forecast_df)
            st.session_state["forecast_df"] = forecast_df
            st.session_state["forecast_kpis"] = kpis
            st.session_state["forecast_horizon"] = horizon_days
            st.session_state["forecast_choice"] = horizon_choice
        except Exception as e:
            st.error(f"Forecasting error: {e}")
            st.stop()

if st.session_state["forecast_df"] is None:
    st.markdown(
        """
        <div class="content-box" style="text-align: center; padding: 3.5rem 1.5rem; margin-top: 1rem;">
            <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">📈</div>
            <h3 style="font-size: 1.25rem; font-weight: 700; color: #0F172A; margin-bottom: 0.4rem;">Forecast Ready to Generate</h3>
            <p style="font-size: 0.95rem; color: #64748B; max-width: 520px; margin: 0 auto;">
                Select your desired horizon (30, 60, or 90 Days) above and click <strong>Predict Forecast</strong> to calculate projected daily sales, KPIs, and interactive charts.
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
st.markdown(
    f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-title">Total Forecast Sales</div>
            <div class="kpi-number">{format_number(kpis['total_sales'])}</div>
            <div class="kpi-subtext">Sum of predicted daily sales for {active_choice}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Average Daily Sales</div>
            <div class="kpi-number">{format_number(kpis['average_daily_sales'])}</div>
            <div class="kpi-subtext">Mean predicted sales per day over horizon</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Forecast Period</div>
            <div class="kpi-number">{active_horizon} Days</div>
            <div class="kpi-subtext">{format_date(kpis['start_date'])} to {format_date(kpis['end_date'])}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Highest Predicted Day</div>
            <div class="kpi-number">{format_number(kpis['highest_sales_day_value'])}</div>
            <div class="kpi-subtext">Expected on {format_date(kpis['highest_sales_day_date'])}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Chart 1: Historical Sales and Future Forecast
# ---------------------------------------------------------
st.markdown("### Historical Sales & Future Forecast")

# Slice historical series based on user selection
if history_window == "Recent 90 Days":
    hist_slice = daily_series.iloc[-90:]
elif history_window == "Recent 180 Days":
    hist_slice = daily_series.iloc[-180:]
elif history_window == "Recent 365 Days":
    hist_slice = daily_series.iloc[-365:]
else:
    hist_slice = daily_series

fig = go.Figure()

# 1. Historical Actual Sales trace
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

# 2. Forecasted Daily Sales trace (distinct bold color)
fig.add_trace(
    go.Scatter(
        x=forecast_df.index,
        y=forecast_df["Predicted_Sales"],
        mode="lines+markers",
        name=f"Forecasted Daily Sales ({horizon_choice})",
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

# Add clear annotation for the boundary
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
    
    # Resample forecast into weekly sums to show actionable business trends
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

    # Table formatted cleanly
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
    # Only export Date and Predicted Daily Sales (no private source data)
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
