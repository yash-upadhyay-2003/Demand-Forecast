"""
FastAPI Backend for Daily Sales Forecasting.

Provides RESTful endpoints for model readiness, dataset metadata,
historical sales context, and multi-horizon sales forecasting.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import numpy as np

# Reusable project modules
from data_loader import resolve_default_paths, load_raw_dataset
from forecasting_engine import (
    load_model_artifact,
    prepare_daily_sales_series,
    generate_forecast,
    compute_forecast_kpis,
)

logger = logging.getLogger("sales_forecast_api")
logging.basicConfig(level=logging.INFO)

# Global in-memory cache for startup persistence
app_data: Dict[str, Any] = {
    "daily_series": None,
    "model_artifact": None,
    "dataset_path": None,
    "model_path": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load dataset and model artifact once at application startup.
    """
    logger.info("Initializing Daily Sales Forecasting API...")
    dataset_path, model_path = resolve_default_paths(Path(__file__).parent)

    if dataset_path is None or not dataset_path.exists():
        logger.error(f"Dataset file not found at: {dataset_path}")
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    if model_path is None or not model_path.exists():
        logger.error(f"Model file not found at: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load dataset and create continuous daily series
    raw_df = load_raw_dataset(dataset_path)
    daily_series = prepare_daily_sales_series(raw_df)

    # Load model artifact
    model_artifact = load_model_artifact(model_path)

    app_data["daily_series"] = daily_series
    app_data["model_artifact"] = model_artifact
    app_data["dataset_path"] = dataset_path
    app_data["model_path"] = model_path

    logger.info(
        f"Backend startup complete. Dataset: {daily_series.index.min().date()} to {daily_series.index.max().date()} ({len(daily_series)} days). "
        f"Model: {model_artifact.get('selected_model_name')}."
    )
    yield
    logger.info("Shutting down API...")


app = FastAPI(
    title="Daily Sales Forecasting API",
    description="REST API powering the Daily Sales Forecasting web application.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for Streamlit frontend (supports both local and Render deployments)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    service: str
    model_ready: bool
    dataset_ready: bool
    last_history_date: Optional[str] = None


class MetadataResponse(BaseModel):
    dataset_start_date: str
    dataset_end_date: str
    forecast_starts_on: str
    total_calendar_days: int
    zero_sales_days: int
    supported_horizons: List[int] = [30, 60, 90]


class HistoryRecord(BaseModel):
    date: str
    sales: float


class HistoryResponse(BaseModel):
    window_requested: str
    total_records: int
    records: List[HistoryRecord]


class ForecastRequest(BaseModel):
    horizon_days: int = Field(30, description="Forecast horizon in days (must be 30, 60, or 90)")

    @field_validator("horizon_days")
    @classmethod
    def validate_horizon(cls, v: int) -> int:
        if v not in [30, 60, 90]:
            raise ValueError("horizon_days must be exactly 30, 60, or 90.")
        return v


class ForecastDayRecord(BaseModel):
    date: str
    predicted_sales: float
    probability_of_sale: float
    median_component: float
    expected_component: float


class ForecastKPIs(BaseModel):
    total_sales: float
    average_daily_sales: float
    highest_sales_day_value: float
    highest_sales_day_date: str
    days_with_predicted_sales: int
    total_days: int
    start_date: str
    end_date: str


class ForecastResponse(BaseModel):
    horizon_days: int
    forecast_start_date: str
    forecast_end_date: str
    kpis: ForecastKPIs
    forecast: List[ForecastDayRecord]


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
def get_health():
    """
    Check the service health and model readiness status.
    """
    model_ready = app_data["model_artifact"] is not None
    dataset_ready = app_data["daily_series"] is not None
    last_date = (
        str(app_data["daily_series"].index.max().date())
        if dataset_ready
        else None
    )

    return HealthResponse(
        status="healthy" if (model_ready and dataset_ready) else "degraded",
        service="Daily Sales Forecasting API",
        model_ready=model_ready,
        dataset_ready=dataset_ready,
        last_history_date=last_date,
    )


@app.get("/metadata", response_model=MetadataResponse, tags=["Dataset"])
def get_metadata():
    """
    Return dataset date boundaries, calendar totals, and forecasting start date.
    """
    series: Optional[pd.Series] = app_data["daily_series"]
    if series is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Historical sales dataset is not initialized."
        )

    start_date = series.index.min()
    end_date = series.index.max()
    forecast_start = end_date + pd.Timedelta(days=1)
    zero_days = int((series == 0).sum())

    return MetadataResponse(
        dataset_start_date=start_date.strftime("%Y-%m-%d"),
        dataset_end_date=end_date.strftime("%Y-%m-%d"),
        forecast_starts_on=forecast_start.strftime("%Y-%m-%d"),
        total_calendar_days=len(series),
        zero_sales_days=zero_days,
        supported_horizons=[30, 60, 90],
    )


@app.get("/history", response_model=HistoryResponse, tags=["Dataset"])
def get_history(
    days: Optional[str] = Query(
        "90",
        description="Number of past calendar days (90, 180, 365, or 'full')",
    )
):
    """
    Return historical daily sales for UI charting context.
    Supports 90, 180, 365 days or full history.
    """
    series: Optional[pd.Series] = app_data["daily_series"]
    if series is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Historical sales dataset is not initialized."
        )

    days_str = str(days).strip().lower()
    if days_str in ["full", "all"]:
        sliced = series
        window_label = "Full History"
    else:
        try:
            n_days = int(days_str)
            if n_days <= 0:
                raise ValueError()
            sliced = series.iloc[-n_days:]
            window_label = f"Recent {n_days} Days"
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query parameter 'days' must be an integer (e.g. 90, 180, 365) or 'full'."
            )

    records = [
        HistoryRecord(date=d.strftime("%Y-%m-%d"), sales=float(v))
        for d, v in sliced.items()
    ]

    return HistoryResponse(
        window_requested=window_label,
        total_records=len(records),
        records=records,
    )


@app.post("/forecast", response_model=ForecastResponse, tags=["Forecasting"])
def post_forecast(request: ForecastRequest):
    """
    Generate sales forecast for the requested horizon (30, 60, or 90 days).
    """
    series: Optional[pd.Series] = app_data["daily_series"]
    artifact: Optional[Dict[str, Any]] = app_data["model_artifact"]

    if series is None or artifact is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or dataset is not ready."
        )

    try:
        forecast_df = generate_forecast(
            daily_series=series,
            model_artifact=artifact,
            n_days=request.horizon_days,
        )
        kpi_dict = compute_forecast_kpis(forecast_df)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.exception("Inference error during forecast generation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forecast calculation failed: {str(e)}"
        )

    day_records = [
        ForecastDayRecord(
            date=d.strftime("%Y-%m-%d"),
            predicted_sales=float(row["Predicted_Sales"]),
            probability_of_sale=float(row["Probability_of_Sale"]),
            median_component=float(row["Median_Component"]),
            expected_component=float(row["Expected_Component"]),
        )
        for d, row in forecast_df.iterrows()
    ]

    kpis = ForecastKPIs(
        total_sales=kpi_dict["total_sales"],
        average_daily_sales=kpi_dict["average_daily_sales"],
        highest_sales_day_value=kpi_dict["highest_sales_day_value"],
        highest_sales_day_date=kpi_dict["highest_sales_day_date"].strftime("%Y-%m-%d"),
        days_with_predicted_sales=kpi_dict["days_with_predicted_sales"],
        total_days=kpi_dict["total_days"],
        start_date=kpi_dict["start_date"].strftime("%Y-%m-%d"),
        end_date=kpi_dict["end_date"].strftime("%Y-%m-%d"),
    )

    return ForecastResponse(
        horizon_days=request.horizon_days,
        forecast_start_date=kpi_dict["start_date"].strftime("%Y-%m-%d"),
        forecast_end_date=kpi_dict["end_date"].strftime("%Y-%m-%d"),
        kpis=kpis,
        forecast=day_records,
    )
