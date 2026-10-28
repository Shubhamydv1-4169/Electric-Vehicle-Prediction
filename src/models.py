"""
models.py
Trains, cross-validates, and benchmarks 5 Machine Learning classification algorithms:
1. Gradient Boosting Classifier (Boosting Ensemble)
2. Random Forest Classifier (Bagging Ensemble)
3. K-Nearest Neighbors Classifier (Instance-based Non-parametric)
4. Logistic Regression (L2 Regularized Baseline)
5. AdaBoost Classifier (Adaptive Boosting)

Identifies best-fit model and exports fitted weights and performance scorecard.
"""

import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from typing import Dict, Any, Tuple


def get_project_root() -> Path:
    """Return the root path of the project."""
    return Path(__file__).resolve().parent.parent


def get_model_zoo() -> Dict[str, Any]:
    """Return dictionary of candidate models with competitive configurations."""
    return {
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            random_state=42,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
        ),
        "Logistic Regression": LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=80,
            learning_rate=0.1,
            random_state=42,
        ),
    }


def train_and_benchmark_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_splits: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
    """
    Train and benchmark all 5 models using 5-Fold Stratified Cross-Validation
    and out-of-sample test evaluation.
    """
    model_zoo = get_model_zoo()
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    results = []
    fitted_models = {}

    print(f"\n{'='*75}\nTRAINING & BENCHMARKING 5 MACHINE LEARNING MODELS (EV PREDICTION)\n{'='*75}")

    for name, model in model_zoo.items():
        print(f"\n--> Training: {name}...")

        # 5-fold Stratified CV
        cv_auc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        cv_f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")

        # Fit on full training set
        model.fit(X_train, y_train)
        fitted_models[name] = model

        # Out-of-sample test predictions
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        test_auc = roc_auc_score(y_test, y_prob)

        results.append({
            "Model": name,
            "CV ROC-AUC (Mean)": round(float(np.mean(cv_auc_scores)), 4),
            "CV ROC-AUC (Std)": round(float(np.std(cv_auc_scores)), 4),
            "CV F1-Score": round(float(np.mean(cv_f1_scores)), 4),
            "Test Accuracy": round(float(acc), 4),
            "Test Precision": round(float(prec), 4),
            "Test Recall": round(float(rec), 4),
            "Test F1-Score": round(float(f1), 4),
            "Test ROC-AUC": round(float(test_auc), 4),
        })

    metrics_df = pd.DataFrame(results)
    # Composite score prioritizing ROC-AUC and F1
    metrics_df["Composite_Score"] = (metrics_df["Test F1-Score"] * 0.5) + (metrics_df["Test ROC-AUC"] * 0.5)
    metrics_df = metrics_df.sort_values(by="Composite_Score", ascending=False).reset_index(drop=True)

    best_model_name = metrics_df.iloc[0]["Model"]
    print("\n" + "="*75)
    print("FINAL BENCHMARK SCORECARD:")
    print("="*75)
    print(metrics_df.to_string(index=False))
    print(f"\n🏆 Best-Fit Selected Model: '{best_model_name}' (Composite Score: {metrics_df.iloc[0]['Composite_Score']:.4f})")

    return metrics_df, fitted_models, best_model_name


def save_models_and_metrics(
    metrics_df: pd.DataFrame,
    fitted_models: Dict[str, Any],
    best_model_name: str,
) -> None:
    """Persist best model, all candidate weights, and metrics scorecard."""
    root = get_project_root()
    models_dir = os.path.join(root, "outputs", "models")
    outputs_dir = os.path.join(root, "outputs")
    os.makedirs(models_dir, exist_ok=True)

    metrics_csv_path = os.path.join(outputs_dir, "metrics_summary.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"[Model Registry] Saved metrics scorecard to: {metrics_csv_path}")

    # Best model serialization
    best_model = fitted_models[best_model_name]
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_model, best_model_path)
    print(f"[Model Registry] Serialized best model ('{best_model_name}') to: {best_model_path}")

    # Save individual models
    for name, mdl in fitted_models.items():
        fname = name.lower().replace(" ", "_").replace("-", "_") + ".joblib"
        joblib.dump(mdl, os.path.join(models_dir, fname))


if __name__ == "__main__":
    from preprocessing import clean_ev_data
    from feature_engineering import prepare_modeling_data

    cleaned_df = clean_ev_data()
    X_train, X_test, y_train, y_test, features = prepare_modeling_data(cleaned_df)

    metrics_df, models, best_name = train_and_benchmark_models(X_train, y_train, X_test, y_test)
    save_models_and_metrics(metrics_df, models, best_name)
