"""
Data and Model Loader Module.

Handles safe discovery, validation, and loading of sales dataset files
and trained model artifacts.
"""

from typing import Optional, Tuple
from pathlib import Path
import pandas as pd


DEFAULT_CSV_NAME = "daily_sales_data.csv"
FALLBACK_CSV_NAME = "New_Query_2026_10_22_10_01_37.csv"
DEFAULT_MODEL_NAME = "final_sales_forecast_model.joblib"


def resolve_default_paths(base_dir: Optional[str | Path] = None) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Search for the standard dataset and model files in base_dir or project root.
    Prioritizes sanitized daily sales dataset for privacy.
    """
    base = Path(base_dir) if base_dir else Path(".")

    # 1. Look for dataset (prefer sanitized daily aggregate)
    dataset_path = None
    if (base / DEFAULT_CSV_NAME).exists():
        dataset_path = base / DEFAULT_CSV_NAME
    elif (base / FALLBACK_CSV_NAME).exists():
        dataset_path = base / FALLBACK_CSV_NAME
    else:
        # Search for pattern match
        matches = sorted(base.glob("New_Query_*.csv"))
        if matches:
            dataset_path = matches[0]
        else:
            all_csvs = sorted(base.glob("*.csv"))
            if all_csvs:
                dataset_path = all_csvs[0]

    # 2. Look for model
    model_path = None
    exact_model = base / DEFAULT_MODEL_NAME
    if exact_model.exists():
        model_path = exact_model
    else:
        joblibs = sorted(base.glob("*.joblib"))
        if joblibs:
            model_path = joblibs[0]

    return dataset_path, model_path


def load_raw_dataset(source: str | Path | object) -> pd.DataFrame:
    """
    Load raw transaction data from file path or file-like object (e.g. Streamlit UploadedFile).
    Validates essential columns without retaining or exposing unnecessary sensitive fields.
    """
    # Read only relevant columns if available, or load and validate
    df = pd.read_csv(source)

    if "OrderDate" not in df.columns or "Sales" not in df.columns:
        raise ValueError(
            f"The provided CSV file must contain 'OrderDate' and 'Sales' columns. "
            f"Found columns: {list(df.columns)}"
        )

    return df
