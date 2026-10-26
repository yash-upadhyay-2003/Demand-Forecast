"""
Unit and Integration Test Suite for FastAPI Backend (api.py).
"""

from fastapi.testclient import TestClient
import numpy as np

from api import app


def run_all_tests():
    with TestClient(app) as client:
        # 1. Health check
        print("Testing GET /health...")
        r_health = client.get("/health")
        assert r_health.status_code == 200, f"Health status error: {r_health.text}"
        d_health = r_health.json()
        assert d_health["status"] == "healthy"
        assert d_health["model_ready"] is True
        assert d_health["dataset_ready"] is True
        assert d_health["last_history_date"] == "2014-09-29"
        print("  -> GET /health passed.")

        # 2. Metadata check
        print("Testing GET /metadata...")
        r_meta = client.get("/metadata")
        assert r_meta.status_code == 200, f"Metadata status error: {r_meta.text}"
        d_meta = r_meta.json()
        assert d_meta["dataset_start_date"] == "2011-01-21"
        assert d_meta["dataset_end_date"] == "2014-09-29"
        assert d_meta["forecast_starts_on"] == "2014-09-30"
        assert d_meta["total_calendar_days"] == 1348
        assert d_meta["zero_sales_days"] == 815
        assert d_meta["supported_horizons"] == [30, 60, 90]
        print("  -> GET /metadata passed.")

        # 3. History endpoint checks
        print("Testing GET /history...")
        r90 = client.get("/history?days=90")
        assert r90.status_code == 200
        assert r90.json()["total_records"] == 90

        r180 = client.get("/history?days=180")
        assert r180.status_code == 200
        assert r180.json()["total_records"] == 180

        r365 = client.get("/history?days=365")
        assert r365.status_code == 200
        assert r365.json()["total_records"] == 365

        r_full = client.get("/history?days=full")
        assert r_full.status_code == 200
        assert r_full.json()["total_records"] == 1348

        r_bad = client.get("/history?days=invalid")
        assert r_bad.status_code == 400
        print("  -> GET /history passed (90, 180, 365, full, and validation).")

        # 4. Forecast endpoint checks for 30, 60, 90 days
        print("Testing POST /forecast (30, 60, 90 days)...")
        expected_30_total = 23245.37
        for horizon in [30, 60, 90]:
            r_fc = client.post("/forecast", json={"horizon_days": horizon})
            assert r_fc.status_code == 200, f"Forecast error for {horizon}d: {r_fc.text}"
            d_fc = r_fc.json()
            assert d_fc["horizon_days"] == horizon
            assert d_fc["forecast_start_date"] == "2014-09-30"
            assert len(d_fc["forecast"]) == horizon

            kpis = d_fc["kpis"]
            assert kpis["total_days"] == horizon
            assert kpis["start_date"] == "2014-09-30"
            assert kpis["total_sales"] > 0
            assert kpis["average_daily_sales"] > 0

            if horizon == 30:
                assert np.isclose(kpis["total_sales"], expected_30_total, atol=0.05)
                assert d_fc["forecast_end_date"] == "2014-10-29"
            elif horizon == 60:
                assert d_fc["forecast_end_date"] == "2014-11-28"
            elif horizon == 90:
                assert d_fc["forecast_end_date"] == "2014-12-28"

            for day in d_fc["forecast"]:
                assert day["predicted_sales"] >= 0.0
                assert 0.0 <= day["probability_of_sale"] <= 1.0
        print("  -> POST /forecast passed (all 3 horizons and schema validation).")

        # 5. Invalid horizon validation
        print("Testing POST /forecast invalid horizons...")
        for bad_h in [15, 45, 120, -1, 0]:
            r_inv = client.post("/forecast", json={"horizon_days": bad_h})
            assert r_inv.status_code == 422, f"Expected 422 for horizon {bad_h}"
        print("  -> Invalid horizon validation passed.")

    print("\nALL BACKEND API TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_all_tests()
