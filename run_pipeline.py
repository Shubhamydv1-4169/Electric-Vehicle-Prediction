"""
run_pipeline.py
Master execution pipeline for the Electric Vehicle Prediction & Analytics Project.
Runs end-to-end: data ingestion -> cleaning -> EDA -> feature engineering ->
modeling -> evaluation.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loader import load_raw_data, inspect_raw_data
from preprocessing import clean_ev_data
from eda import run_full_eda
from feature_engineering import prepare_modeling_data
from models import train_and_benchmark_models, save_models_and_metrics
from evaluate import evaluate_and_generate_visualizations


def run():
    print("=" * 80)
    print("⚡ STARTING ELECTRIC VEHICLE PREDICTION & ANALYTICS PIPELINE")
    print("=" * 80)
    start_time = time.time()

    # Step 1: Ingestion
    print("\n[Step 1/6] Ingesting raw EV population dataset...")
    raw_df = load_raw_data()
    inspect_summary = inspect_raw_data(raw_df)
    print(f"Ingested {inspect_summary['num_rows']:,} records across {inspect_summary['num_cols']} attributes.")

    # Step 2: Preprocessing
    print("\n[Step 2/6] Cleaning data & handling anomalies...")
    cleaned_df = clean_ev_data(raw_df, save_processed=True)

    # Step 3: EDA & 5 Objectives Investigation
    print("\n[Step 3/6] Investigating 5 core EV business objectives...")
    eda_insights = run_full_eda(cleaned_df)

    # Step 4: Feature Engineering
    print("\n[Step 4/6] Engineering domain features and encoding splits...")
    X_train, X_test, y_train, y_test, feature_names = prepare_modeling_data(cleaned_df, save_splits=True)

    # Step 5: Model Training & Benchmarking
    print("\n[Step 5/6] Benchmarking 5 Machine Learning algorithms with 5-Fold Stratified CV...")
    metrics_df, fitted_models, best_model_name = train_and_benchmark_models(
        X_train, y_train, X_test, y_test
    )
    save_models_and_metrics(metrics_df, fitted_models, best_model_name)

    # Step 6: Evaluation
    print("\n[Step 6/6] Generating evaluation visuals...")
    evaluate_and_generate_visualizations()

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 80)
    print(f"✅ PIPELINE SUCCESSFULLY COMPLETED IN {elapsed}s")
    print(f"🏆 Best Selected Model: {best_model_name}")
    print(f"📁 Processed Datasets: {PROJECT_ROOT}/data/processed")
    print(f"📊 Figures & Charts:  {PROJECT_ROOT}/outputs/figures")
    print(f"📦 Serialized Models: {PROJECT_ROOT}/outputs/models")
    print("=" * 80)


if __name__ == "__main__":
    run()
