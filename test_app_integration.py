"""
Comprehensive Verification and Smoke Test for Daily Sales Forecasting App.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from forecasting_engine import (
    load_model_artifact,
    prepare_daily_sales_series,
    generate_forecast,
    compute_forecast_kpis,
)
from data_loader import resolve_default_paths, load_raw_dataset


def run_tests():
    print("=" * 60)
    print("STARTING DAILY SALES FORECASTING APPLICATION TESTS")
    print("=" * 60)

    # 1. Test Path Resolution
    print("\n[Test 1] Testing file path resolution...")
    csv_path, model_path = resolve_default_paths(".")
    assert csv_path is not None and csv_path.exists(), f"CSV not found: {csv_path}"
    assert model_path is not None and model_path.exists(), f"Model not found: {model_path}"
    print(f"  -> Found dataset: {csv_path.name}")
    print(f"  -> Found model: {model_path.name}")

    # 2. Test Data Loading & Processing
    print("\n[Test 2] Testing dataset loading and daily calendar construction...")
    raw_df = load_raw_dataset(csv_path)
    daily_series = prepare_daily_sales_series(raw_df)

    start_date = daily_series.index.min()
    end_date = daily_series.index.max()
    print(f"  -> Dataset Start Date: {start_date.strftime('%d %b %Y')}")
    print(f"  -> Dataset End Date:   {end_date.strftime('%d %b %Y')}")
    print(f"  -> Total calendar days: {len(daily_series)}")
    print(f"  -> Zero-sales days preserved: {(daily_series == 0).sum()}")

    assert start_date == pd.Timestamp("2011-01-21"), f"Unexpected start date: {start_date}"
    assert end_date == pd.Timestamp("2014-09-29"), f"Unexpected end date: {end_date}"
    assert daily_series.index.is_monotonic_increasing, "Index not monotonic increasing"
    assert daily_series.index.is_unique, "Index contains duplicate dates"
    assert (daily_series.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all(), "Calendar has missing gap days"

    # 3. Test Model Artifact Loading
    print("\n[Test 3] Testing model artifact loading...")
    model_artifact = load_model_artifact(model_path)
    print(f"  -> Model Family: {model_artifact.get('selected_model_name')}")
    print(f"  -> Variant: {model_artifact.get('variant')}")
    print(f"  -> Blend Lambda: {model_artifact.get('blend_lambda')}")
    print(f"  -> Max Horizon: {model_artifact.get('max_horizon')}")

    assert "models" in model_artifact
    assert "occurrence" in model_artifact["models"]
    assert "amount" in model_artifact["models"]

    # 4. Test 30, 60, and 90-day Forecasts
    print("\n[Test 4] Testing 30, 60, and 90-day forecast generation...")
    horizons = [30, 60, 90]
    expected_30_sum = 23245.37  # verified against notebook cell 133

    for h in horizons:
        forecast_df = generate_forecast(daily_series, model_artifact, n_days=h)
        kpis = compute_forecast_kpis(forecast_df)

        fc_start = forecast_df.index.min()
        fc_end = forecast_df.index.max()

        print(f"\n  --- Horizon: {h} Days ---")
        print(f"  Forecast Starts On: {fc_start.strftime('%d %b %Y')}")
        print(f"  Forecast Ends On:   {fc_end.strftime('%d %b %Y')}")
        print(f"  Total Sales:        {kpis['total_sales']:,.2f}")
        print(f"  Avg Daily Sales:    {kpis['average_daily_sales']:,.2f}")
        print(f"  Highest Day:        {kpis['highest_sales_day_value']:,.2f} on {kpis['highest_sales_day_date'].strftime('%d %b %Y')}")

        # Verification rules:
        # Rule a: Exactly 1 calendar day after dataset end date
        assert fc_start == end_date + pd.Timedelta(days=1), f"Start date error: {fc_start} != {end_date} + 1 day"
        # Rule b: Correct number of days
        assert len(forecast_df) == h, f"Length error: {len(forecast_df)} != {h}"
        # Rule c: Strictly no overlap with historical dates
        assert fc_start > end_date, "Forecast start overlaps historical data"
        # Rule d: Ordered and unique dates
        assert forecast_df.index.is_monotonic_increasing, "Forecast dates not sorted"
        assert forecast_df.index.is_unique, "Forecast dates not unique"
        # Rule e: Predictions are finite and non-negative
        assert np.isfinite(forecast_df["Predicted_Sales"].to_numpy()).all(), "Predictions contain NaN/Inf"
        assert (forecast_df["Predicted_Sales"] >= 0).all(), "Predictions contain negative values"

        # Check 30-day sum exact agreement with notebook
        if h == 30:
            assert np.isclose(kpis['total_sales'], expected_30_sum, atol=0.05), (
                f"30-day total {kpis['total_sales']} does not match notebook {expected_30_sum}"
            )

        # Rule f: Verify KPI values and table values agree
        assert np.isclose(kpis['total_sales'], forecast_df["Predicted_Sales"].sum())
        assert np.isclose(kpis['average_daily_sales'], forecast_df["Predicted_Sales"].mean())

    print("\n" + "=" * 60)
    print("ALL TEST CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
