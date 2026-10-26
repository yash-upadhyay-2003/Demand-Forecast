# Daily Sales Forecasting — Local Web Application

A clean, modern, and professional web application for forecasting future daily sales. Built using Python, Streamlit, Plotly, pandas, and the trained Two-Stage Machine Learning model artifact (`final_sales_forecast_model.joblib`).

---

## Features

- **Date Range Visibility:** Dynamically displays the dataset start date, dataset end date, forecast start date (strictly dataset end date + 1 calendar day), and forecast end date.
- **Horizon Selector:** Choose between **30 Days**, **60 Days**, and **90 Days** with instant dynamic updates.
- **Key Performance Indicators (KPIs):**
  - Total Forecast Sales
  - Average Daily Sales
  - Forecast Period
  - Highest Predicted Day & Value
- **Interactive Visualizations:**
  - **Chart 1:** Historical actual sales vs. future forecasted sales with configurable context windows (Recent 90, 180, 365 Days, or Full History).
  - **Chart 2:** Weekly forecast aggregation to monitor weekly trends and operational spikes.
- **Data Table & Export:** Clean forecast table showing each forecast date and predicted sales, with a 1-click **Download Forecast CSV** feature (exporting only clean dates and predictions, maintaining customer privacy).
- **Predict Forecast Action:** Interactive horizon selector with a dedicated **Predict Forecast** action button.

---

## Project Structure

```text
DDF/
├── app.py                            # Streamlit web application & UI
├── forecasting_engine.py             # Core feature engineering & 2-stage inference pipeline
├── data_loader.py                    # Safe dataset and model loading utilities
├── final_sales_forecast_model.joblib # Saved trained model artifact
├── daily_sales_data.csv              # Sanitized daily sales dataset
├── Daily_Sales_Forecasting_NB.ipynb  # Original research & validation notebook
├── test_app_integration.py           # Verification & smoke test suite
├── requirements.txt                  # Python package dependencies
├── .gitignore                        # Git ignore rules
└── README.md                         # Documentation & instructions
```

---

## Quickstart & Local Setup

### 1. Install Dependencies

Ensure you have Python 3.10+ installed. In your terminal or command prompt inside the project folder, run:

```bash
pip install -r requirements.txt
```

### 2. Launch the Web Application

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

### 3. Open in Browser

Streamlit will launch locally, normally at:

```
http://localhost:8501
```

---

## Machine Learning Pipeline Summary

- **Architecture:** Direct multi-horizon two-stage model:
  - **Stage 1 (Occurrence):** Logistic Regression estimating $P(\text{Sales} > 0)$ using rolling zero-rates and calendar harmonics.
  - **Stage 2 (Amount):** Ridge Regression ($\alpha = 50.0$) on log-transformed positive sales amounts with Duan smearing correction.
  - **Blend Decision Rule:** Blends median point forecast with smeared expected sales ($\lambda = 0.3$) for optimal WAPE and total sales alignment.
- **Calendar Handling:** All non-recorded order dates are treated as confirmed zero-sales days, maintaining a complete and continuous daily calendar.
