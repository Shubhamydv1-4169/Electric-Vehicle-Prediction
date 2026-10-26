"""
feature_engineering.py
Constructs domain-specific features for Electric Vehicle modeling,
applies categorical encoding, scales numerical predictors, and exports
reproducible stratified train-test splits.
"""

import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List


def get_project_root() -> Path:
    """Return the root path of the project."""
    return Path(__file__).resolve().parent.parent


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct high-signal domain features for EV modeling.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned EV dataset.

    Returns
    -------
    pd.DataFrame
        Dataset enriched with engineered features.
    """
    df_feat = df.copy()

    # 1. Vehicle Vintage / Age
    current_year = 2024
    df_feat["Vehicle_Age"] = np.maximum(0, current_year - df_feat["Model Year"])

    # 2. Modern Era Indicator (Model Year >= 2020)
    df_feat["Is_Modern_EV"] = (df_feat["Model Year"] >= 2020).astype(int)

    # 3. Automaker Brand Classification
    luxury_makes = {
        "TESLA", "BMW", "AUDI", "MERCEDES-BENZ", "PORSCHE",
        "LUCID", "RIVIAN", "POLESTAR", "LEXUS", "VOLVO",
        "JAGUAR", "LAND ROVER", "GENESIS"
    }
    df_feat["Is_Luxury_Make"] = df_feat["Make"].isin(luxury_makes).astype(int)
    df_feat["Is_Tesla"] = (df_feat["Make"] == "TESLA").astype(int)

    # Top 10 Makes grouping
    top_10_makes = [
        "TESLA", "CHEVROLET", "NISSAN", "FORD", "BMW",
        "KIA", "HYUNDAI", "VOLVO", "JEEP", "AUDI"
    ]
    df_feat["Make_Grouped"] = df_feat["Make"].apply(lambda m: m if m in top_10_makes else "OTHER")

    # 4. Range Category Binning
    bins = [-1, 0, 35, 100, 200, 300, 1000]
    labels = ["Unrecorded_Zero", "Short_Under35m", "Commuter_35_100m", "Mid_100_200m", "High_200_300m", "Ultra_300m_Plus"]
    df_feat["Range_Cohort"] = pd.cut(df_feat["Electric Range"], bins=bins, labels=labels).astype(str)

    # 5. Geographic Metropolitan Indicator (King County = Seattle Metro)
    df_feat["Is_King_County"] = (df_feat["County"] == "King").astype(int)

    # 6. CAFV Eligibility Flags
    df_feat["Is_CAFV_Eligible"] = (df_feat["CAFV_Status"] == "Eligible").astype(int)
    df_feat["Is_CAFV_Low_Range"] = (df_feat["CAFV_Status"] == "Not Eligible (Low Range)").astype(int)

    # 7. Pricing reporting flag
    df_feat["Has_MSRP"] = (df_feat["Base MSRP"] > 0).astype(int)

    return df_feat


def prepare_modeling_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    save_splits: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Transform engineered dataframe into encoded, scaled, stratified train-test splits.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned input DataFrame.
    test_size : float
        Holdout test ratio.
    random_state : int
        Reproducibility seed.
    save_splits : bool
        Whether to export processed CSVs.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]
        (X_train_scaled, X_test_scaled, y_train, y_test, feature_names)
    """
    root = get_project_root()
    df_eng = engineer_features(df)

    # Separate target: EV_Type_Binary (1 = BEV, 0 = PHEV)
    y = df_eng["EV_Type_Binary"].copy()

    # Features to drop (identifiers, target, and target-leakage predictors)
    # Range and CAFV columns are post-test characteristics that directly leak vehicle powertrain
    drop_cols = [
        "VIN (1-10)", "DOL Vehicle ID", "Electric Vehicle Type",
        "Clean Alternative Fuel Vehicle (CAFV) Eligibility",
        "Vehicle Location", "2020 Census Tract", "EV_Type_Short",
        "EV_Type_Binary", "Make", "Model", "County", "City",
        "State", "Electric Utility", "Electric Range", "Range_Recorded",
        "Range_Cohort", "CAFV_Status", "Is_CAFV_Eligible", "Is_CAFV_Low_Range"
    ]
    X_raw = df_eng.drop(columns=[c for c in drop_cols if c in df_eng.columns])

    # One-Hot Encoding for nominal categorical features (Make groups)
    one_hot_cols = ["Make_Grouped"]
    X_encoded = pd.get_dummies(X_raw, columns=one_hot_cols, drop_first=True, dtype=int)

    feature_names = list(X_encoded.columns)
    print(f"[Feature Engineering] Final predictor feature count (leakage-free): {len(feature_names)}")

    # Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scaling numerical features (Fit ONLY on Training split)
    num_scale_cols = ["Model Year", "Vehicle_Age", "Base MSRP", "Legislative District", "Postal Code"]
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[num_scale_cols] = scaler.fit_transform(X_train[num_scale_cols])
    X_test_scaled[num_scale_cols] = scaler.transform(X_test[num_scale_cols])

    # Save splits and artifacts
    if save_splits:
        processed_dir = os.path.join(root, "data", "processed")
        models_dir = os.path.join(root, "outputs", "models")
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        X_train_scaled.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
        X_test_scaled.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)

        joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
        with open(os.path.join(models_dir, "feature_names.json"), "w") as f:
            json.dump(feature_names, f, indent=2)

        print(f"[Feature Engineering] Saved splits and artifacts to {processed_dir} & {models_dir}")

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names


if __name__ == "__main__":
    from preprocessing import clean_ev_data
    df_clean = clean_ev_data()
    X_tr, X_te, y_tr, y_te, feats = prepare_modeling_data(df_clean)
    print("\n--- Feature Engineering Complete ---")
    print(f"X_train: {X_tr.shape}, y_train: {y_tr.shape}")
    print(f"X_test:  {X_te.shape}, y_test:  {y_te.shape}")
    print("Sample features:", feats[:8], f"... ({len(feats)} total)")
