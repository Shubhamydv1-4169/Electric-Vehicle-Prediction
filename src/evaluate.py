"""
evaluate.py
Evaluates trained Electric Vehicle machine learning models, generates
comparative performance charts, multi-model ROC curves, confusion matrices,
and feature importance diagrams.
"""

import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)


def get_project_root() -> Path:
    """Return the root path of the project."""
    return Path(__file__).resolve().parent.parent


def evaluate_and_generate_visualizations():
    """Generate all evaluation graphics and summaries."""
    root = get_project_root()
    processed_dir = os.path.join(root, "data", "processed")
    models_dir = os.path.join(root, "outputs", "models")
    figures_dir = os.path.join(root, "outputs", "figures")
    outputs_dir = os.path.join(root, "outputs")
    os.makedirs(figures_dir, exist_ok=True)

    # Load holdout test data
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze()
    with open(os.path.join(models_dir, "feature_names.json"), "r") as f:
        feature_names = json.load(f)

    # Load candidate models
    model_names = [
        ("Gradient Boosting", "gradient_boosting.joblib"),
        ("Random Forest", "random_forest.joblib"),
        ("K-Nearest Neighbors", "k_nearest_neighbors.joblib"),
        ("Logistic Regression", "logistic_regression.joblib"),
        ("AdaBoost", "adaboost.joblib"),
    ]
    models = {}
    for display_name, fname in model_names:
        fpath = os.path.join(models_dir, fname)
        if os.path.exists(fpath):
            models[display_name] = joblib.load(fpath)

    # -------------------------------------------------------------
    # 1. Comparative Metrics Bar Chart
    # -------------------------------------------------------------
    metrics_csv = os.path.join(outputs_dir, "metrics_summary.csv")
    metrics_df = pd.read_csv(metrics_csv)

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df = pd.melt(
        metrics_df,
        id_vars=["Model"],
        value_vars=["Test Accuracy", "Test Precision", "Test Recall", "Test F1-Score", "Test ROC-AUC"],
        var_name="Metric",
        value_name="Score"
    )

    sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", palette="tab10", ax=ax)
    ax.set_ylim(0.5, 1.05)
    ax.set_title("EV Powertrain Classification: Model Performance Comparison", fontweight="bold", fontsize=14)
    ax.set_ylabel("Score (0.5 - 1.0)")
    ax.legend(loc="lower right", frameon=True)
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "07_model_metrics_comparison.png")
    fig.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # 2. Multi-Model ROC Curves
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))
    palette = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#f39c12"]

    for (name, model), color in zip(models.items(), palette):
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.predict(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        model_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {model_auc:.3f})", color=color, linewidth=2.2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance Level (AUC = 0.500)")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontweight="bold", fontsize=14)
    ax.legend(loc="lower right", fontsize=11, frameon=True)
    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "08_roc_curves.png")
    fig.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # 3. Confusion Matrices Heatmap
    # -------------------------------------------------------------
    top_three = ["Gradient Boosting", "Random Forest", "Logistic Regression"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, name in enumerate(top_three):
        model = models[name]
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[idx],
            annot_kws={"size": 14, "weight": "bold"},
            xticklabels=["PHEV (0)", "BEV (1)"],
            yticklabels=["PHEV (0)", "BEV (1)"],
        )
        axes[idx].set_title(f"{name}\nConfusion Matrix", fontweight="bold", fontsize=13)
        axes[idx].set_xlabel("Predicted Powertrain")
        axes[idx].set_ylabel("True Powertrain")

    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "09_confusion_matrices.png")
    fig.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # 4. Feature Importance Plot (Gradient Boosting)
    # -------------------------------------------------------------
    gb_model = models.get("Gradient Boosting")
    if gb_model and hasattr(gb_model, "feature_importances_"):
        gb_imp = pd.Series(gb_model.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=gb_imp.values, y=gb_imp.index, palette="mako", ax=ax, hue=gb_imp.index, legend=False)
        ax.set_title("Top 15 Most Influential Predictors (Gradient Boosting)", fontweight="bold", fontsize=14)
        ax.set_xlabel("Relative Feature Importance")
        ax.set_ylabel("Features")

        for i, v in enumerate(gb_imp.values):
            ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9, fontweight="bold")

        plt.tight_layout()
        fig4_path = os.path.join(figures_dir, "10_feature_importance.png")
        fig.savefig(fig4_path, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"[Evaluation] All evaluation plots successfully created and saved to: {figures_dir}")


if __name__ == "__main__":
    evaluate_and_generate_visualizations()
