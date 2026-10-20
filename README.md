# ⚡ Electric Vehicle Prediction & Market Analytics

An end-to-end Data Science and Machine Learning project developed to analyze electric vehicle adoption trends, investigate technological and policy drivers across 5 business objectives, and deploy machine learning models to predict electric vehicle technology types (**Battery Electric Vehicles - BEV** vs. **Plug-in Hybrid Electric Vehicles - PHEV**).

---

## 🎯 Project Overview & Problem Definition
The global automotive landscape is rapidly transitioning from internal combustion engines to electric powertrains. Understanding market penetration, battery autonomy (electric range), automaker competitiveness, and clean fuel incentive eligibility is critical for automakers, policymakers, and utility providers.

This project delivers:
1. **Data Ingestion & Hygiene**: Automated preprocessing, missing value resolution, duplicate checks, and type standardization on the **Electric Vehicle Population dataset** (from data.gov / Washington State DOL).
2. **Exploratory Data Analysis (EDA)**: Investigating **5 distinct business objectives** with publication-grade visualizations.
3. **Feature Engineering**: Brand tier categorization, vehicle vintage/age, range efficiency metrics, and geographic clustering.
4. **Machine Learning Pipeline**: Benchmarking **5 classification algorithms** (Logistic Regression, Random Forest, Gradient Boosting, AdaBoost, and K-Nearest Neighbors) using 5-Fold Stratified Cross-Validation.
5. **Model Evaluation & Deliverables**: Full metrics scorecard, ROC curves, confusion matrices, serialized models (`.joblib`), executed Jupyter Notebook, and an executive report.

---

## 🗂️ Project Repository Structure

```
Electric-Vehicle-Prediction/
├── data/
│   ├── raw/                       # Original EV population dataset
│   │   └── Electric_Vehicle_Population_Data.csv
│   └── processed/                 # Cleaned and split datasets
│       ├── cleaned_ev_data.csv
│       ├── X_train.csv, X_test.csv
│       └── y_train.csv, y_test.csv
├── notebooks/                     # Executed Jupyter analysis & modeling notebook
│   └── electric_vehicle_prediction_and_analysis.ipynb
├── src/                           # Modular Python production source code
│   ├── __init__.py
│   ├── data_loader.py             # Data ingestion and verification
│   ├── preprocessing.py          # Data cleaning and handling missing values
│   ├── eda.py                    # Exploratory Data Analysis & visual plots
│   ├── feature_engineering.py    # Encoders, scalers, and domain features
│   ├── models.py                 # ML algorithms training & tuning
│   └── evaluate.py               # Evaluation metrics, ROC, confusion matrices
├── outputs/
│   ├── figures/                  # High-resolution saved visualizations
│   ├── models/                   # Serialized best model & scaler (.joblib)
│   └── metrics_summary.csv       # Comparative performance scorecard
├── report/
│   ├── PROJECT_REPORT.md         # Comprehensive executive & technical report
│   └── screenshots/              # Project execution, notebook & social proofs
├── requirements.txt              # Environment dependencies
├── run_pipeline.py               # Single-command end-to-end execution script
└── README.md
```

---

## ⚙️ Quick Start Installation

```bash
# Clone the repository
git clone https://github.com/Shubhamydv1-4169/Electric-Vehicle-Prediction.git
cd Electric-Vehicle-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the complete end-to-end pipeline
python3 run_pipeline.py
```
