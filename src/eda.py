"""
eda.py
Comprehensive Exploratory Data Analysis module for the Electric Vehicle dataset.
Investigates 5 core business objectives and generates publication-grade visualizations.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_project_root() -> Path:
    """Return the root path of the project."""
    return Path(__file__).resolve().parent.parent


def set_plotting_style():
    """Configure modern, publication-grade aesthetics for plots."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 15,
        "figure.dpi": 300,
    })


def run_full_eda(df: pd.DataFrame, output_dir: str = None) -> dict:
    """
    Execute Exploratory Data Analysis across 5 core business objectives.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned electric vehicle dataset.
    output_dir : str
        Directory where generated figures will be stored.

    Returns
    -------
    dict
        Dictionary containing statistical insights.
    """
    root = get_project_root()
    if output_dir is None:
        output_dir = os.path.join(root, "outputs", "figures")
    os.makedirs(output_dir, exist_ok=True)

    set_plotting_style()
    insights = {}

    # -------------------------------------------------------------
    # Objective 1: EV Technology Adoption Dynamics (BEV vs PHEV)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Overall Donut
    type_counts = df["EV_Type_Short"].value_counts()
    axes[0].pie(
        type_counts,
        labels=[f"BEV (Battery Electric)\n({type_counts['BEV']:,})", f"PHEV (Plug-in Hybrid)\n({type_counts['PHEV']:,})"],
        autopct="%1.1f%%",
        startangle=140,
        colors=["#2980b9", "#27ae60"],
        explode=(0.04, 0.04),
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
    )
    axes[0].set_title("Overall Electric Powertrain Market Share", fontweight="bold")

    # Trend by Model Year (Filtered to 2013-2024 for clean representation)
    df_trend = df[(df["Model Year"] >= 2013) & (df["Model Year"] <= 2024)]
    yearly = df_trend.groupby(["Model Year", "EV_Type_Short"]).size().unstack(fill_value=0)
    yearly.plot(
        kind="bar",
        stacked=True,
        color=["#2980b9", "#27ae60"],
        ax=axes[1],
        edgecolor="none",
        width=0.75
    )
    axes[1].set_title("Annual EV Registrations by Powertrain Type (2013–2024)", fontweight="bold")
    axes[1].set_xlabel("Model Year")
    axes[1].set_ylabel("Vehicle Registrations")
    axes[1].legend(["BEV (Pure Electric)", "PHEV (Plug-in Hybrid)"], loc="upper left")
    plt.xticks(rotation=45)

    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "01_ev_technology_adoption.png")
    fig.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()

    insights["objective_1"] = {
        "bev_percentage": round(df["EV_Type_Binary"].mean() * 100, 2),
        "phev_percentage": round((1 - df["EV_Type_Binary"].mean()) * 100, 2),
        "total_vehicles_analyzed": len(df),
    }

    # -------------------------------------------------------------
    # Objective 2: Electric Range (Autonomy) Dynamics & Battery Capability
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Range > 0 cohort
    df_range = df[df["Electric Range"] > 0]

    sns.kdeplot(
        data=df_range,
        x="Electric Range",
        hue="EV_Type_Short",
        common_norm=False,
        fill=True,
        alpha=0.35,
        palette={"BEV": "#2980b9", "PHEV": "#27ae60"},
        ax=axes[0],
        linewidth=2
    )
    axes[0].set_title("Electric Range (Miles) Probability Density", fontweight="bold")
    axes[0].set_xlabel("Electric Range (Miles)")
    axes[0].set_ylabel("Density")

    sns.boxplot(
        data=df_range,
        x="EV_Type_Short",
        y="Electric Range",
        hue="EV_Type_Short",
        palette={"BEV": "#2980b9", "PHEV": "#27ae60"},
        ax=axes[1],
        width=0.4,
        legend=False
    )
    axes[1].set_title("Range Distribution Boxplot by Powertrain", fontweight="bold")
    axes[1].set_xlabel("Powertrain Type")
    axes[1].set_ylabel("Electric Range (Miles)")

    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "02_electric_range_distribution.png")
    fig.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()

    bev_median_range = df_range[df_range["EV_Type_Short"] == "BEV"]["Electric Range"].median()
    phev_median_range = df_range[df_range["EV_Type_Short"] == "PHEV"]["Electric Range"].median()

    insights["objective_2"] = {
        "bev_median_range_miles": float(bev_median_range),
        "phev_median_range_miles": float(phev_median_range),
        "range_gap_ratio": round(float(bev_median_range / phev_median_range), 2),
    }

    # -------------------------------------------------------------
    # Objective 3: Automaker Competitiveness & Brand Positioning
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    top_makes = df["Make"].value_counts().head(10).index
    df_top_makes = df[df["Make"].isin(top_makes)]

    make_type = df_top_makes.groupby(["Make", "EV_Type_Short"]).size().unstack(fill_value=0)
    make_type["Total"] = make_type.sum(axis=1)
    make_type = make_type.sort_values(by="Total", ascending=True)

    make_type[["BEV", "PHEV"]].plot(
        kind="barh",
        stacked=True,
        color=["#2980b9", "#27ae60"],
        ax=ax,
        edgecolor="none",
        width=0.7
    )
    ax.set_title("Top 10 Electric Vehicle Automakers & Powertrain Mix", fontweight="bold")
    ax.set_xlabel("Number of Registered Vehicles")
    ax.set_ylabel("Automaker (Make)")
    ax.legend(["BEV (Pure Electric)", "PHEV (Plug-in Hybrid)"], loc="lower right")

    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "03_automaker_market_share.png")
    fig.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close()

    tesla_share = (df["Make"] == "TESLA").mean() * 100
    top3_makes = list(df["Make"].value_counts().head(3).index)

    insights["objective_3"] = {
        "tesla_market_share_percent": round(tesla_share, 2),
        "top_3_makes": top3_makes,
    }

    # -------------------------------------------------------------
    # Objective 4: CAFV Policy Compliance Analysis
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))
    cafv_grouped = df.groupby(["CAFV_Status", "EV_Type_Short"]).size().unstack(fill_value=0)
    cafv_pct = cafv_grouped.div(cafv_grouped.sum(axis=1), axis=0) * 100

    cafv_pct.plot(
        kind="bar",
        color=["#2980b9", "#27ae60"],
        ax=ax,
        edgecolor="none",
        width=0.6
    )
    ax.set_title("Clean Alternative Fuel Vehicle (CAFV) Status by Powertrain", fontweight="bold")
    ax.set_ylabel("Percentage within Status (%)")
    ax.set_xlabel("CAFV Eligibility Status")
    ax.legend(["BEV", "PHEV"], loc="upper right")
    plt.xticks(rotation=15)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=9)

    plt.tight_layout()
    fig4_path = os.path.join(output_dir, "04_cafv_eligibility_analysis.png")
    fig.savefig(fig4_path, dpi=300, bbox_inches="tight")
    plt.close()

    insights["objective_4"] = {
        "eligible_count": int(df[df["CAFV_Status"] == "Eligible"].shape[0]),
        "low_range_ineligible_count": int(df[df["CAFV_Status"] == "Not Eligible (Low Range)"].shape[0]),
        "phev_ineligible_rate": round((df[df["EV_Type_Short"] == "PHEV"]["CAFV_Status"] == "Not Eligible (Low Range)").mean() * 100, 2),
    }

    # -------------------------------------------------------------
    # Objective 5: Geographic Diffusion & Utility Infrastructure
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Top 8 Counties
    top_counties = df["County"].value_counts().head(8).sort_values(ascending=True)
    sns.barplot(
        x=top_counties.values,
        y=top_counties.index,
        palette="Blues_r",
        ax=axes[0],
        hue=top_counties.index,
        legend=False
    )
    axes[0].set_title("Top 8 Counties by EV Adoption Volume", fontweight="bold")
    axes[0].set_xlabel("Registered Electric Vehicles")
    for i, v in enumerate(top_counties.values):
        axes[0].text(v + 100, i, f"{v:,}", va="center", fontsize=9, fontweight="bold")

    # Top 6 Electric Utilities
    top_utilities = df["Electric Utility"].value_counts().head(6).sort_values(ascending=True)
    clean_util_labels = [u.split("|")[0][:28] for u in top_utilities.index]
    sns.barplot(
        x=top_utilities.values,
        y=clean_util_labels,
        palette="Greens_r",
        ax=axes[1],
        hue=clean_util_labels,
        legend=False
    )
    axes[1].set_title("Top 6 Electric Utilities Servicing EVs", fontweight="bold")
    axes[1].set_xlabel("Connected Electric Vehicles")
    for i, v in enumerate(top_utilities.values):
        axes[1].text(v + 80, i, f"{v:,}", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig5_path = os.path.join(output_dir, "05_geographic_utility_distribution.png")
    fig.savefig(fig5_path, dpi=300, bbox_inches="tight")
    plt.close()

    top_county = df["County"].value_counts().index[0]
    top_county_share = (df["County"] == top_county).mean() * 100

    insights["objective_5"] = {
        "top_county": top_county,
        "top_county_share_percent": round(top_county_share, 2),
    }

    # -------------------------------------------------------------
    # Correlation Heatmap
    # -------------------------------------------------------------
    num_cols = ["Model Year", "Electric Range", "Base MSRP", "Legislative District", "Range_Recorded", "EV_Type_Binary"]
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", square=True, linewidths=1, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontweight="bold")
    plt.tight_layout()
    fig6_path = os.path.join(output_dir, "06_correlation_heatmap.png")
    fig.savefig(fig6_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[EDA] Generated 6 publication-grade figures in: {output_dir}")
    return insights


if __name__ == "__main__":
    from preprocessing import clean_ev_data
    df_clean = clean_ev_data()
    summary = run_full_eda(df_clean)
    print("\n--- EDA Insights Summary ---")
    for obj, res in summary.items():
        print(f"\n{obj.upper()}:")
        for k, v in res.items():
            print(f"  - {k}: {v}")
