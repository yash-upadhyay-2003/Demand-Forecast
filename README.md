# Daily Sales Forecasting — Full-Stack Application

A modern, professional daily sales forecasting application featuring a decoupled architecture:
- **Backend API (FastAPI):** High-performance REST service for model loading, feature engineering, and inference (ready for Render deployment).
- **Frontend UI (Streamlit):** Clean business user interface with interactive Plotly charts, dynamic date cards, KPIs, and CSV downloads.

---

## Architecture Overview

```text
┌───────────────────────────────┐           HTTP REST API          ┌───────────────────────────────┐
│     Streamlit Frontend        │ ───────────────────────────────> │       FastAPI Backend         │
│           (app.py)            │ <─────────────────────────────── │           (api.py)            │
│   • Interactive UI & Charts   │    /metadata, /history, /forecast│   • Preprocessing & Models    │
│   • No direct model dependencies│                                │   • Daily Sales Series        │
└───────────────────────────────┘                                  └───────────────────────────────┘
```

---

## Features

- **Decoupled Client-Server:** The frontend communicates with the backend via REST endpoints (`/health`, `/metadata`, `/history`, `/forecast`), keeping ML models and datasets encapsulated on the server.
- **Date Range Visibility:** Dynamically displays Dataset Start Date (`21 Jan 2011`), Dataset End Date (`29 Sep 2014`), Forecast Starts On (`30 Sep 2014`), and Forecast Ends On.
- **Horizon Selector & Predict Action:** Choose between **30 Days**, **60 Days**, and **90 Days** with a dedicated **Predict Forecast** button.
- **Key Performance Indicators (KPIs):**
  - Total Forecast Sales
  - Average Daily Sales
  - Forecast Period
  - Highest Predicted Day & Value
- **Interactive Visualizations:**
  - **Chart 1:** Historical actual sales vs. future forecasted daily sales with configurable context windows (Recent 90, 180, 365 Days, or Full History).
  - **Chart 2:** Weekly forecast aggregation to highlight weekly patterns and peaks.
- **Data Table & Export:** Clean forecast table showing each forecast date and predicted sales, with a 1-click **Download Forecast CSV** feature (exporting clean dates and predictions, ensuring customer privacy).

---

## Project Structure

```text
DDF/
├── api.py                            # FastAPI backend REST service
├── app.py                            # Streamlit web application & UI
├── forecasting_engine.py             # Core feature engineering & 2-stage inference pipeline
├── data_loader.py                    # Safe dataset and model loading utilities
├── final_sales_forecast_model.joblib # Trained Two-Stage ML model artifact
├── daily_sales_data.csv              # Sanitized daily sales dataset (dates & sales totals)
├── test_api.py                       # Backend REST endpoint test suite
├── test_app_integration.py           # Core ML engine integration & smoke tests
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
└── README.md                         # Documentation & instructions
```

---

## Quickstart & Local Setup

### 1. Install Dependencies

Ensure Python 3.10+ is installed:

```bash
pip install -r requirements.txt
```

### 2. Start the FastAPI Backend

Run the backend on port 8000:

```bash
python -m uvicorn api:app --reload --port 8000
```

- API Base URL: `http://127.0.0.1:8000`
- Interactive Swagger Docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

### 3. Start the Streamlit Frontend

In a separate terminal window, run:

```bash
python -m streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`. By default, it connects to the local backend at `http://127.0.0.1:8000`.

---

## Configuring `BACKEND_URL` (for Render / Production Deployment)

When deploying the frontend and backend separately (e.g. on Render):

### Option A: Via Environment Variable

Set `BACKEND_URL` to your live backend service URL:

**Linux / macOS:**
```bash
export BACKEND_URL="https://your-backend.onrender.com"
python -m streamlit run app.py
```

**Windows PowerShell:**
```powershell
$env:BACKEND_URL="https://your-backend.onrender.com"
python -m streamlit run app.py
```

**Render Dashboard:**
In your Streamlit web service settings on Render, add an Environment Variable:
- **Key:** `BACKEND_URL`
- **Value:** `https://your-backend-service.onrender.com`

### Option B: Via Streamlit Secrets

In your Streamlit deployment or locally in `.streamlit/secrets.toml`:

```toml
BACKEND_URL = "https://your-backend.onrender.com"
```

---

## API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service and model readiness status |
| `GET` | `/metadata` | Dataset boundaries, forecast start date, and calendar totals |
| `GET` | `/history?days={n}` | Historical sales data for charting (`90`, `180`, `365`, or `full`) |
| `POST` | `/forecast` | Generate multi-horizon forecast (`{"horizon_days": 30}`) |

---

## Running Automated Tests

Run the backend API test suite:
```bash
python test_api.py
```

Run the core forecasting integration test suite:
```bash
python test_app_integration.py
```

---

## Machine Learning Pipeline Summary

- **Architecture:** Direct multi-horizon two-stage model:
  - **Stage 1 (Occurrence):** Logistic Regression estimating $P(\text{Sales} > 0)$ using rolling zero-rates and calendar harmonics.
  - **Stage 2 (Amount):** Ridge Regression ($\alpha = 50.0$) on log-transformed positive sales amounts with Duan smearing correction.
  - **Blend Decision Rule:** Blends median point forecast with smeared expected sales ($\lambda = 0.3$) for optimal WAPE and total sales alignment.
- **Calendar Handling:** All non-recorded order dates are treated as confirmed zero-sales days, maintaining a complete and continuous daily calendar.
