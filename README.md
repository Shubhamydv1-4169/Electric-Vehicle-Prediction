# ⚡ Electric Vehicle Prediction & Market Analytics
### Production-Grade Machine Learning & Clean Energy Intelligence System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Models: Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

An enterprise-ready Data Science and Machine Learning project developed to analyze electric vehicle adoption trajectories, investigate technological and policy drivers across 5 business dimensions, benchmark 5 classification models, and predict electric vehicle powertrain technologies (**Battery Electric Vehicles - BEV** vs. **Plug-in Hybrid Electric Vehicles - PHEV**).

---

## 📑 Table of Contents
- [Project Overview & Problem Definition](#-project-overview--problem-definition)
- [Key Insights & 5 Business Objectives](#-key-insights--5-business-objectives)
- [Machine Learning Benchmark Scorecard](#-machine-learning-benchmark-scorecard)
- [Visual Showcase](#-visual-showcase)
- [Project Repository Layout](#-project-repository-layout)
- [Installation & Quick Start](#-installation--quick-start)
- [Connecting to GitHub & Pushing Commits](#-connecting-to-github--pushing-commits)
- [Author & License](#-author--license)

---

## 🎯 Project Overview & Problem Definition

The global automotive landscape is rapidly transitioning from internal combustion engines to electrified mobility. Understanding powertrain market share, battery autonomy, brand competition, and clean vehicle incentive eligibility is critical for automakers, grid utilities, and transportation authorities.

This project delivers:
1. **Automated Data Cleaning & Integrity**: Handles missing geographic entries, standardizes government CAFV criteria, and eliminates categorical inconsistencies across 25,000 representative EV registrations.
2. **5 Targeted Business Objectives**: Statistically explores powertrain adoption trends, battery range distributions, automaker competition, policy incentives, and geographic utility demand.
3. **Domain Feature Engineering**: Extracts 21 engineered predictors (vintage, luxury tier, automaker groupings, regional hubs) while eliminating post-test data leakage.
4. **Multi-Model ML Benchmarking**: Evaluates 5 classifiers under 5-Fold Stratified Cross-Validation and holdout test sets ($N=5,000$).
5. **Production Deployment Ready**: Persists fitted weights, feature definitions, and scalers in `joblib` format.

---

## 🔍 Key Insights & 5 Business Objectives

| Objective | Business Question | Primary Finding & Data Evidence |
| :--- | :--- | :--- |
| **Obj 1: Powertrain Adoption** | What is the adoption trajectory between BEV and PHEV? | **BEVs dominate with 78.30% of total market share**, accelerating past **82.5% in recent model years (2020–2024)**. |
| **Obj 2: Electric Range (Autonomy)** | How does battery range compare between BEVs and PHEVs? | **BEVs have a median range of 215 miles** vs. **28 miles for PHEVs** (a **7.68x autonomy advantage**). |
| **Obj 3: Automaker Competition** | Which automakers lead the market and what is their powertrain mix? | **Tesla leads with 44.05% market share** (100% BEV). European luxury makes (BMW, Volvo) rely on PHEVs (>65% hybrid share). |
| **Obj 4: CAFV Policy Compliance** | What factors determine government incentive eligibility? | **100% of disqualified vehicles are PHEVs** whose battery range falls below the statutory 30-mile threshold. |
| **Obj 5: Geographic Diffusion** | Where are electric vehicles concentrated geographically? | **King County (Seattle Metro) accounts for 51.91% of all registered EVs**, powered primarily by Puget Sound Energy and Seattle City Light. |

---

## 🤖 Machine Learning Benchmark Scorecard

Evaluated on 5-Fold Stratified Cross-Validation and Holdout Test Set ($N=5,000$):

| Model | CV ROC-AUC (Mean ± Std) | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC | Composite Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🏆 **Gradient Boosting Classifier** | **0.9644 ± 0.0018** | **90.48%** | **93.17%** | **94.79%** | **0.9397** | **0.9640** | **0.9518 (Rank 1)** |
| 🥈 **Random Forest Classifier** | 0.9627 ± 0.0023 | 90.46% | 92.03% | **96.14%** | **0.9404** | 0.9626 | **0.9515 (Rank 2)** |
| 🥉 **Logistic Regression (Balanced)** | 0.9457 ± 0.0032 | 85.90% | **98.67%** | 83.12% | 0.9023 | 0.9465 | **0.9244 (Rank 3)** |
| 4. **K-Nearest Neighbors (KNN)** | 0.9114 ± 0.0046 | 89.20% | 93.04% | 93.18% | 0.9311 | 0.9055 | **0.9183 (Rank 4)** |
| 5. **AdaBoost Classifier** | 0.7811 ± 0.0021 | 78.30% | 78.30% | 100.00% | 0.8783 | 0.7821 | **0.8302 (Rank 5)** |

> **Selection Rationale**: **Gradient Boosting Classifier** achieved the top composite score (**0.9518**), leading in overall Test Accuracy (**90.48%**) and ROC-AUC (**0.9640**), with Random Forest performing closely as an ensemble alternative.

---

## 🖼️ Visual Showcase

All charts are saved at 300 DPI in [`outputs/figures/`](outputs/figures/):
- `01_ev_technology_adoption.png`: EV Powertrain Adoption Trajectory (BEV vs. PHEV)
- `02_electric_range_distribution.png`: Electric Range (Autonomy) Probability Density & Boxplot
- `03_automaker_market_share.png`: Top 10 EV Automakers & Powertrain Mix
- `04_cafv_eligibility_analysis.png`: Clean Alternative Fuel Vehicle (CAFV) Policy Compliance
- `05_geographic_utility_distribution.png`: Regional County Adoption & Electric Utilities
- `06_correlation_heatmap.png`: Multivariate Feature Correlation Matrix
- `07_model_metrics_comparison.png`: 5 ML Models Benchmark Bar Chart
- `08_roc_curves.png`: Multi-Model Receiver Operating Characteristic (ROC) Curves
- `09_confusion_matrices.png`: Side-by-Side Confusion Matrices (GBDT, RF, LogReg)
- `10_feature_importance.png`: Top 15 Feature Importances (Gradient Boosting)

---

## 🗂️ Project Repository Layout

```
Electric-Vehicle-Prediction/
├── .gitignore                                 # Git ignore definitions
├── README.md                                  # Production portfolio documentation
├── requirements.txt                           # Pinned dependencies
├── run_pipeline.py                            # Master one-command execution pipeline
├── data/
│   ├── raw/
│   │   └── Electric_Vehicle_Population_Data.csv # Clean 25,000 EV dataset
│   └── processed/
│       ├── cleaned_ev_data.csv                # Sanitized dataset
│       ├── X_train.csv, X_test.csv            # Feature matrices (leakage-free)
│       └── y_train.csv, y_test.csv            # Stratified target splits
├── notebooks/
│   ├── build_notebook.py                      # Pre-computation generator
│   └── electric_vehicle_prediction_and_analysis.ipynb # Fully executed Jupyter Notebook
├── src/
│   ├── __init__.py                            # Package initializer
│   ├── data_loader.py                         # Ingestion & integrity checking
│   ├── preprocessing.py                      # Cleaning, missing values & duplicates
│   ├── eda.py                                # Visualizations for 5 objectives
│   ├── feature_engineering.py                # Domain feature extraction & scaling
│   ├── models.py                             # 5 ML models training & CV benchmarking
│   └── evaluate.py                           # Performance evaluation & ROC curves
├── outputs/
│   ├── figures/                              # 10 High-resolution saved charts (.png)
│   ├── models/                               # Serialized best model & scaler (.joblib)
│   └── metrics_summary.csv                   # Model comparison scorecard
└── report/
    ├── PROJECT_REPORT.md                     # Comprehensive technical & business report
    └── screenshots/                          # Screenshot checklist & LinkedIn showcase copy
```

---

## ⚙️ Installation & Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Shubhamydv1-4169/Electric-Vehicle-Prediction.git
cd Electric-Vehicle-Prediction

# 2. Set up virtual environment (optional)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the complete pipeline
python3 run_pipeline.py
```

---

## 👤 Author
**Shubham Kumar**  
Data Science & Machine Learning Engineer  
GitHub: [@Shubhamydv1-4169](https://github.com/Shubhamydv1-4169)
