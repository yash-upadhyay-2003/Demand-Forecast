"""
Forecasting Engine for Daily Sales Forecasting.

Reuses the exact feature engineering and two-stage direct multi-horizon forecasting
pipeline developed and validated in the project notebook.
"""

from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import joblib


def load_model_artifact(model_path: str | Path) -> Dict[str, Any]:
    """
    Safely load the saved joblib model artifact.
    
    The artifact contains:
      - 'models': dict with 'occurrence', 'amount', 'log_residuals', 'smear'
      - 'blend_lambda': float weighting between median and smeared expected value
      - 'variant': point forecast variant ('median')
      - 'max_horizon': maximum trained horizon (e.g. 90)
      - 'last_history_date': string date of the training series end
      - 'selected_model_name': name of the selected model family
    """
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model artifact not found at {p.resolve()}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        artifact = joblib.load(p)

    required_keys = ["models", "blend_lambda", "variant"]
    for k in required_keys:
        if k not in artifact:
            raise KeyError(f"Invalid model artifact: missing '{k}'")

    if not isinstance(artifact["models"], dict) or "occurrence" not in artifact["models"] or "amount" not in artifact["models"]:
        raise ValueError("Model artifact is missing required sub-models ('occurrence', 'amount').")

    return artifact


def prepare_daily_sales_series(raw_df: pd.DataFrame) -> pd.Series:
    """
    Convert raw transaction-level DataFrame into a continuous daily Sales series.
    Absent dates represent confirmed zero-sales days as established in the project.
    
    Validates required columns: 'OrderDate' and 'Sales'.
    """
    if "OrderDate" not in raw_df.columns:
        raise ValueError("Dataset is missing required date column: 'OrderDate'")
    if "Sales" not in raw_df.columns:
        raise ValueError("Dataset is missing required sales target column: 'Sales'")

    # Parse dates safely
    order_dates = pd.to_datetime(raw_df["OrderDate"], utc=True, errors="coerce")
    if order_dates.isna().all():
        raise ValueError("All 'OrderDate' values failed to parse into valid datetime objects.")

    # Remove timezone and normalize to calendar day (00:00:00)
    normalized_dates = order_dates.dt.tz_localize(None).dt.normalize()
    
    # Filter valid rows
    valid_mask = normalized_dates.notna() & raw_df["Sales"].notna()
    if not valid_mask.any():
        raise ValueError("No valid date/sales records found in dataset.")

    sales_numeric = pd.to_numeric(raw_df.loc[valid_mask, "Sales"], errors="coerce").fillna(0.0)

    # Aggregate transaction sales to daily totals
    daily_aggregated = (
        pd.DataFrame({"Date": normalized_dates[valid_mask], "Sales": sales_numeric})
        .groupby("Date")["Sales"]
        .sum()
        .sort_index()
    )

    if daily_aggregated.empty:
        raise ValueError("No daily sales records could be aggregated.")

    # Reindex to a complete daily calendar, filling non-transaction days with 0.0
    full_calendar = pd.date_range(
        start=daily_aggregated.index.min(),
        end=daily_aggregated.index.max(),
        freq="D"
    )
    
    daily_complete = daily_aggregated.reindex(full_calendar, fill_value=0.0).astype(float)
    daily_complete.name = "Sales"
    daily_complete.index.name = "Date"

    return daily_complete


def compute_origin_features(history_values: np.ndarray) -> Dict[str, float]:
    """
    Compute features using ONLY the history available at the forecast origin.
    Matches features_v2 / origin_features from the validated project notebook.
    """
    v = np.asarray(history_values, dtype=float)
    positive_all = v[v > 0]
    fallback = float(np.log1p(positive_all).mean()) if len(positive_all) > 0 else 0.0

    def zero_rate(n: int) -> float:
        window = v[-n:] if len(v) >= n else v
        return float((window == 0).mean()) if len(window) > 0 else 0.0

    def log_level(n: int) -> float:
        window = v[-n:] if len(v) >= n else v
        pos = window[window > 0]
        return float(np.log1p(pos).mean()) if len(pos) > 0 else fallback

    w28 = v[-28:] if len(v) >= 28 else v
    log_mean_28 = float(np.log1p(w28).mean()) if len(w28) > 0 else 0.0

    return {
        "zero_rate_14": zero_rate(14),
        "zero_rate_28": zero_rate(28),
        "zero_rate_90": zero_rate(90),
        "zero_rate_180": zero_rate(180),
        "log_level_90": log_level(90),
        "log_level_365": log_level(365),
        "log_mean_28": log_mean_28,
    }


def compute_calendar_features(date: pd.Timestamp, horizon: int) -> Dict[str, float]:
    """
    Compute calendar and horizon features for a future forecast target date.
    Uses safe day-of-week and calendar calculations.
    """
    dow = getattr(date, "day_of_week", getattr(date, "dayofweek", 0))
    dom = date.day

    return {
        "horizon": float(horizon),
        "dow_sin": float(np.sin(2 * np.pi * dow / 7.0)),
        "dow_cos": float(np.cos(2 * np.pi * dow / 7.0)),
        "month_sin": float(np.sin(2 * np.pi * (date.month - 1) / 12.0)),
        "month_cos": float(np.cos(2 * np.pi * (date.month - 1) / 12.0)),
        "dom_sin": float(np.sin(2 * np.pi * (dom - 1) / 31.0)),
        "dom_cos": float(np.cos(2 * np.pi * (dom - 1) / 31.0)),
        "late_month": 1.0 if dom >= 29 else 0.0,
        "day_31": 1.0 if dom == 31 else 0.0,
        "weekend": 1.0 if dow >= 5 else 0.0,
    }


def make_forecast_feature_matrix(history_series: pd.Series, target_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Construct direct multi-horizon feature rows for forecasting target_dates
    based on the origin history ending at history_series.index[-1].
    """
    base_feats = compute_origin_features(history_series.to_numpy(dtype=float))
    origin_date = history_series.index[-1]

    rows = []
    for d in target_dates:
        horizon = (d - origin_date).days
        cal_feats = compute_calendar_features(d, horizon)
        row = {**base_feats, **cal_feats}
        rows.append(row)

    return pd.DataFrame(rows, index=target_dates)


def predict_two_stage(models: Dict[str, Any], X: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Execute stage 1 (occurrence probability) and stage 2 (amount model)
    and compute median, expected, and smeared-expected forecasts.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = models["occurrence"].predict_proba(X)[:, 1]
        mu = models["amount"].predict(X)

    # 1. Expected value (mean-type before smearing)
    expected = p * np.expm1(mu)

    # 2. Median forecast aligned with WAPE
    q_level = np.clip((p - 0.5) / np.maximum(p, 1e-9), 0.0, 1.0)
    median_amount = np.expm1(mu + np.quantile(models["log_residuals"], q_level))
    median_forecast = np.where(p > 0.5, median_amount, 0.0)

    # 3. Duan smearing correction for log-retransformed mean
    smear_factor = models.get("smear", 1.0)
    expected_smeared = p * np.maximum(smear_factor * np.exp(mu) - 1.0, 0.0)

    out = {
        "median": np.maximum(median_forecast, 0.0),
        "expected": np.maximum(expected, 0.0),
        "expected_smeared": np.maximum(expected_smeared, 0.0),
        "probability": np.clip(p, 0.0, 1.0),
    }

    if "hgb_median" in models:
        out["hgb_median"] = np.maximum(models["hgb_median"].predict(X), 0.0)

    return out


def generate_forecast(
    daily_series: pd.Series,
    model_artifact: Dict[str, Any],
    n_days: int = 30
) -> pd.DataFrame:
    """
    Generate an n_days daily sales forecast starting strictly on the calendar day
    immediately following the last date in daily_series.
    """
    max_h = model_artifact.get("max_horizon", 90)
    if n_days > max_h:
        raise ValueError(f"Requested horizon ({n_days} days) exceeds the model's supported horizon of {max_h} days.")
    if n_days <= 0:
        raise ValueError("Horizon must be a positive integer.")

    last_date = daily_series.index.max()
    forecast_start_date = last_date + pd.Timedelta(days=1)
    target_dates = pd.date_range(start=forecast_start_date, periods=n_days, freq="D")

    # Build features
    X = make_forecast_feature_matrix(daily_series, target_dates)

    # Model inference
    models = model_artifact["models"]
    predictions = predict_two_stage(models, X)

    variant = model_artifact.get("variant", "median")
    lam = float(model_artifact.get("blend_lambda", 0.3))

    if variant == "median":
        point_forecast = (1.0 - lam) * predictions["median"] + lam * predictions["expected_smeared"]
    elif variant in predictions:
        point_forecast = predictions[variant]
    else:
        point_forecast = predictions["median"]

    point_forecast = np.maximum(point_forecast, 0.0)

    df_out = pd.DataFrame(
        {
            "Predicted_Sales": point_forecast,
            "Median_Component": predictions["median"],
            "Expected_Component": predictions["expected_smeared"],
            "Probability_of_Sale": predictions["probability"],
        },
        index=target_dates
    )
    df_out.index.name = "Date"

    return df_out


def compute_forecast_kpis(forecast_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute headline KPIs from the generated forecast DataFrame.
    """
    total_sales = float(forecast_df["Predicted_Sales"].sum())
    avg_sales = float(forecast_df["Predicted_Sales"].mean())
    max_val = float(forecast_df["Predicted_Sales"].max())
    max_date = forecast_df["Predicted_Sales"].idxmax()
    active_days = int((forecast_df["Predicted_Sales"] > 0).sum())

    return {
        "total_sales": total_sales,
        "average_daily_sales": avg_sales,
        "highest_sales_day_value": max_val,
        "highest_sales_day_date": max_date,
        "days_with_predicted_sales": active_days,
        "total_days": len(forecast_df),
        "start_date": forecast_df.index.min(),
        "end_date": forecast_df.index.max(),
    }
