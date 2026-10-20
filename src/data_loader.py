"""
data_loader.py
Module for loading and verifying the Electric Vehicle Population dataset.
"""

import os
from pathlib import Path
import pandas as pd
from typing import Optional, Dict, Any


def get_project_root() -> Path:
    """Return the root path of the project."""
    return Path(__file__).resolve().parent.parent


def load_raw_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw Electric Vehicle Population dataset.

    Parameters
    ----------
    filepath : Optional[str]
        Absolute or relative path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded raw DataFrame.
    """
    if filepath is None:
        root = get_project_root()
        filepath = os.path.join(root, "data", "raw", "Electric_Vehicle_Population_Data.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at: {filepath}")

    df = pd.read_csv(filepath)
    print(f"[DataLoader] Successfully loaded raw dataset from {filepath}")
    print(f"[DataLoader] Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def inspect_raw_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform structural and data integrity checks on raw data.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset.

    Returns
    -------
    dict
        Dictionary containing inspection summaries.
    """
    summary = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "ev_type_distribution": df["Electric Vehicle Type"].value_counts(normalize=True).to_dict()
        if "Electric Vehicle Type" in df.columns
        else {},
    }
    return summary


if __name__ == "__main__":
    raw_df = load_raw_data()
    info = inspect_raw_data(raw_df)
    print("\n--- Raw EV Data Summary ---")
    print(f"Total Rows: {info['num_rows']:,}, Columns: {info['num_cols']}")
    print(f"Duplicates: {info['duplicate_rows']}")
    print("EV Type Distribution:", info["ev_type_distribution"])
