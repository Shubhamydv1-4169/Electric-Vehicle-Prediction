"""
preprocessing.py
Handles data cleaning, missing value resolution, duplicate checks,
and categorical standardization for the Electric Vehicle dataset.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional


def get_project_root() -> Path:
    """Return the root path of the project."""
    return Path(__file__).resolve().parent.parent


def clean_ev_data(
    df: Optional[pd.DataFrame] = None,
    save_processed: bool = True
) -> pd.DataFrame:
    """
    Clean and preprocess the Electric Vehicle Population dataset.

    Steps:
    1. Fill missing values for County, City, Postal Code, and Electric Utility.
    2. Impute missing Legislative District with 0 (indicates out-of-state/unassigned).
    3. Remove duplicate records if any exist.
    4. Normalize CAFV Eligibility labels into concise categories.
    5. Encode target 'Electric Vehicle Type' into binary (BEV=1, PHEV=0).
    6. Create 'Range_Recorded' flag indicating whether positive electric range is logged.
    7. Standardize text casings across Make and Model.
    8. Persist cleaned data to data/processed/cleaned_ev_data.csv.

    Parameters
    ----------
    df : Optional[pd.DataFrame]
        Raw input DataFrame. If None, loaded from data/raw.
    save_processed : bool
        Whether to save to disk.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    root = get_project_root()
    if df is None:
        raw_path = os.path.join(root, "data", "raw", "Electric_Vehicle_Population_Data.csv")
        df = pd.read_csv(raw_path)

    df_cleaned = df.copy()

    # 1. Missing value handling
    df_cleaned["County"] = df_cleaned["County"].fillna("Unknown")
    df_cleaned["City"] = df_cleaned["City"].fillna("Unknown")
    df_cleaned["Postal Code"] = df_cleaned["Postal Code"].fillna(0).astype(int)
    df_cleaned["Electric Utility"] = df_cleaned["Electric Utility"].fillna("Unknown Utility")
    df_cleaned["Legislative District"] = df_cleaned["Legislative District"].fillna(0).astype(int)
    if "Vehicle Location" in df_cleaned.columns:
        df_cleaned["Vehicle Location"] = df_cleaned["Vehicle Location"].fillna("Unknown")

    # 2. Check and remove duplicates
    dups = df_cleaned.duplicated(subset=["DOL Vehicle ID"]).sum() if "DOL Vehicle ID" in df_cleaned.columns else df_cleaned.duplicated().sum()
    if dups > 0:
        print(f"[Preprocessing] Removing {dups} duplicate records.")
        df_cleaned = df_cleaned.drop_duplicates(subset=["DOL Vehicle ID"])
    else:
        print("[Preprocessing] 0 duplicate records detected.")

    # 3. Standardize CAFV Eligibility
    cafv_map = {
        "Clean Alternative Fuel Vehicle Eligible": "Eligible",
        "Not eligible due to low battery range": "Not Eligible (Low Range)",
        "Eligibility unknown as battery range has not been researched": "Unknown / Unresearched"
    }
    df_cleaned["CAFV_Status"] = df_cleaned["Clean Alternative Fuel Vehicle (CAFV) Eligibility"].map(cafv_map).fillna("Unknown / Unresearched")

    # 4. Binary target encoding: BEV = 1, PHEV = 0
    df_cleaned["EV_Type_Short"] = df_cleaned["Electric Vehicle Type"].apply(
        lambda x: "BEV" if "Battery Electric" in str(x) else "PHEV"
    )
    df_cleaned["EV_Type_Binary"] = (df_cleaned["EV_Type_Short"] == "BEV").astype(int)

    # 5. Range recorded flag
    df_cleaned["Range_Recorded"] = (df_cleaned["Electric Range"] > 0).astype(int)

    # 6. Make & Model text normalization
    df_cleaned["Make"] = df_cleaned["Make"].astype(str).str.strip().str.upper()
    df_cleaned["Model"] = df_cleaned["Model"].astype(str).str.strip().str.upper()

    # 7. Base MSRP normalization (0 means MSRP unlisted)
    df_cleaned["Base_MSRP_Reported"] = (df_cleaned["Base MSRP"] > 0).astype(int)

    if save_processed:
        out_dir = os.path.join(root, "data", "processed")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "cleaned_ev_data.csv")
        df_cleaned.to_csv(out_path, index=False)
        print(f"[Preprocessing] Cleaned EV dataset saved to: {out_path}")

    return df_cleaned


if __name__ == "__main__":
    cleaned = clean_ev_data()
    print("\n--- Cleaned EV Data Summary ---")
    print(f"Shape: {cleaned.shape}")
    print("Null count:\n", cleaned.isnull().sum())
    print("\nEV Type Distribution:\n", cleaned["EV_Type_Short"].value_counts(normalize=True))
