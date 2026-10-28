# 📈 Comprehensive Data Science & Machine Learning Project Report
## Electric Vehicle Prediction & Market Analytics

**Author**: Shubham Kumar  
**Domain**: Automotive Engineering, Clean Energy & Predictive Analytics  
**Target**: Classify Electric Vehicle Powertrain Technology (**Battery Electric Vehicle - BEV** vs. **Plug-in Hybrid Electric Vehicle - PHEV**)  
**Artifacts**: Source Code, Executed Jupyter Notebook, Serialized Models, Datasets, Visualizations  
**Date**: October 2026  

---

## Executive Summary

The worldwide automotive sector is undergoing a transformative paradigm shift from fossil fuel combustion to electrified powertrains. As nations mandate zero-emission transport targets, understanding electric vehicle (EV) market penetration, battery range evolution, automaker competitive positioning, and charging infrastructure bottlenecks is of paramount importance to automakers, utilities, and clean transportation policymakers.

This project implements an end-to-end, production-grade Data Science and Machine Learning workflow utilizing the **Electric Vehicle Population dataset** (sourced from Washington State Department of Licensing / Data.gov):
1. Ingest, sanitize, and validate 25,000 representative electric vehicle records across 17 attributes.
2. Investigate **5 distinct core business objectives** exploring powertrain adoption, battery autonomy, brand competition, clean vehicle incentives, and geographic utility demand.
3. Architect high-yield domain features while systematically isolating post-test leakage variables to ensure legitimate predictive capability.
4. Benchmark **5 distinct classification algorithms** (Gradient Boosting, Random Forest, Logistic Regression, K-Nearest Neighbors, AdaBoost) under **5-Fold Stratified Cross-Validation**.
5. Select and serialize the top-performing production model (**Gradient Boosting Classifier**, attaining **0.9640 Test ROC-AUC**, **90.48% Accuracy**, and **0.9397 F1-Score** on holdout test observations).
6. Deliver strategic, data-driven recommendations for automakers, grid operators, and transportation authorities.

---

## 1. Problem Statement & Automotive Clean Energy Context

Electric vehicles encompass two fundamentally distinct powertrain architectures:
- **Battery Electric Vehicles (BEV)**: Pure zero-emission vehicles reliant entirely on high-capacity chemical batteries and electric motors.
- **Plug-in Hybrid Electric Vehicles (PHEV)**: Dual-powertrain systems integrating smaller battery packs with conventional internal combustion backup engines.

The primary objectives of this project are:
- **Powertrain Prediction**: Accurately predict whether an incoming vehicle is a BEV or PHEV based on its registration specifications, automaker, price tier, and vintage without relying on post-test characteristics like measured battery range.
- **Technology Penetration Diagnostics**: Uncover the structural factors accelerating pure BEV adoption over transitional hybrid powertrains.
- **Policy & Grid Planning**: Provide utilities and regional transportation departments with quantitative insights into EV density and charging needs.

---

## 2. Dataset Architecture & Hygiene

The project utilizes the standardized **Electric Vehicle Population dataset** published by the State of Washington Department of Licensing (DOL) via Data.gov:
- **Total Records Analyzed**: 25,000 stratified, representative vehicle registrations.
- **Original Attributes**: 17 attributes spanning vehicle identifiers (VIN), geographical data (County, City, Postal Code, Legislative District), technical specs (Model Year, Make, Model, EV Type, Electric Range, Base MSRP), and policy metadata (CAFV Eligibility, Electric Utility).
- **Target Variable**: `Electric Vehicle Type` (Binary encoded as `BEV = 1` [78.30%] vs. `PHEV = 0` [21.70%]).

### Data Preprocessing & Anomaly Resolution
- **Geographic and Utility Imputations**: Missing values in `County` (0), `City` (0), and `Electric Utility` (0) were sanitized. Missing values in `Legislative District` (52 records) were imputed with `0` (reflecting out-of-state or unassigned districts).
- **De-duplication**: De-duplication was performed against unique `DOL Vehicle ID` entries, verifying 0 duplicate registrations.
- **Categorical Standardization**: Complex government strings under `Clean Alternative Fuel Vehicle (CAFV) Eligibility` were mapped to standardized categories: `Eligible`, `Not Eligible (Low Range)`, and `Unknown / Unresearched`.
- **Text Normalization**: Stripped whitespace and standardized `Make` and `Model` strings to uppercase to eliminate duplicate categories caused by casing discrepancies.

---

## 3. Investigation of Five Core Business Objectives

### 🎯 Objective 1: EV Powertrain Adoption Dynamics (BEV vs. PHEV)
- **Core Hypothesis**: Consumer demand and automaker supply have rapidly shifted toward pure Battery Electric Vehicles (BEV) over hybrid alternatives (PHEV) over the past decade.
- **Visual Evidence**: `outputs/figures/01_ev_technology_adoption.png`
- **Key Findings**:
  - Across the entire population, **BEVs dominate with 78.30% market share**, compared to **21.70% for PHEVs**.
  - In earlier model years (2013–2016), adoption was relatively balanced (45–55% PHEV share) due to range anxiety and limited charging networks.
  - Starting with model year 2018 (accelerated by the launch of mass-market BEVs like the Tesla Model 3 and Chevrolet Bolt), BEV market share surged past 80%, exceeding **82.5% in 2023–2024**.

### 🎯 Objective 2: Electric Range (Autonomy) Dynamics & Battery Capability
- **Core Hypothesis**: BEVs provide significantly greater zero-emission autonomy than PHEVs, creating a substantial performance disparity.
- **Visual Evidence**: `outputs/figures/02_electric_range_distribution.png`
- **Key Findings**:
  - For models with recorded battery ratings, **BEVs achieve a median range of 215.0 miles**, whereas **PHEVs cluster around a median range of 28.0 miles** (a **7.68x autonomy advantage**).
  - PHEVs exhibit a narrow range distribution (interquartile range: 21 to 38 miles), designed specifically for short suburban commutes before engaging the internal combustion engine.
  - BEVs span a broad autonomy spectrum (100 to 330+ miles), reflecting rapid advances in lithium-ion energy density and battery thermal management.

### 🎯 Objective 3: Automaker Competitiveness & Brand Powertrain Strategy
- **Core Hypothesis**: EV market leadership is heavily consolidated, with distinct automaker strategies distinguishing pure-EV pioneers from legacy manufacturers.
- **Visual Evidence**: `outputs/figures/03_automaker_market_share.png`
- **Key Findings**:
  - **Tesla dominates the market with 44.05% of all registered electric vehicles**, producing exclusively pure BEVs.
  - The top 3 automakers (**Tesla, Chevrolet, Nissan**) account for over **58% of the total EV volume**, with Chevrolet and Nissan focusing heavily on pure BEVs (Bolt, Leaf).
  - Conversely, legacy European luxury brands (**BMW, Volvo, Jeep**) remain heavily reliant on PHEVs as a transitional technology (over 65% of BMW and Volvo registrations in the dataset are PHEVs).

### 🎯 Objective 4: Clean Alternative Fuel Vehicle (CAFV) Policy Compliance
- **Core Hypothesis**: Strict statutory battery range requirements for clean fuel incentives disproportionately disqualify hybrid vehicles.
- **Visual Evidence**: `outputs/figures/04_cafv_eligibility_analysis.png`
- **Key Findings**:
  - Out of 11,807 vehicles with evaluated range ratings, **9,057 qualify for full CAFV incentives**, while **2,750 are disqualified**.
  - **100% of disqualified vehicles are PHEVs** that fail to meet the state statutory threshold of 30 miles of all-electric range.
  - Over **50.67% of all PHEVs** fail CAFV eligibility criteria due to undersized battery packs (15–25 miles range).

### 🎯 Objective 5: Geographic Diffusion & Leading EV Hubs
- **Core Hypothesis**: EV adoption is heavily clustered in affluent urban metropolitan tech corridors supported by progressive municipal utilities.
- **Visual Evidence**: `outputs/figures/05_geographic_utility_distribution.png`
- **Key Findings**:
  - **King County (Seattle metropolitan region) accounts for 51.91% of all electric vehicles** in the state (12,977 registered vehicles in this sample).
  - Secondary suburban hubs (**Snohomish County: 11.2%**, **Pierce County: 7.5%**) represent the next largest clusters.
  - Two electric utilities—**Puget Sound Energy (PSE)** and **Seattle City Light**—power over 65% of all electric vehicles, highlighting critical geographic focus areas for grid modernization.

---

## 4. Feature Engineering & Leakage Prevention Strategy

To build an authentic, production-grade predictive model that accurately classifies vehicle powertrain without artificial test leakage:
- **Excluded Variables**: Post-test attributes (`Electric Range`, `Range_Recorded`, `Range_Cohort`, and `CAFV_Status` flags) were deliberately removed from predictive features, as knowing a vehicle has under 35 miles of range trivially reveals that it is a hybrid.
- **Engineered Predictors**:
  1. `Vehicle_Age`: Vintage calculated as $\max(0, 2024 - \text{Model Year})$.
  2. `Is_Modern_EV`: Indicator for model years $\ge 2020$.
  3. `Is_Luxury_Make`: Binary classification for luxury automakers (`TESLA`, `BMW`, `AUDI`, `MERCEDES-BENZ`, `PORSCHE`, `RIVIAN`, `LUCID`, `VOLVO`, `LEXUS`, etc.).
  4. `Is_Tesla`: Binary indicator for the dominant pure-BEV market leader.
  5. `Make_Grouped`: One-hot encoded top 10 automakers with smaller volume makes pooled into `OTHER`.
  6. `Is_King_County`: Binary indicator for the primary urban adoption epicenter.
  7. `Base_MSRP_Reported` & `Has_MSRP`: Indicators capturing premium pricing disclosures.
- **Data Scaling & Splitting**:
  - Stratified 80/20 train/test split ($N_{\text{train}} = 20,000$, $N_{\text{test}} = 5,000$).
  - `StandardScaler` was fitted strictly on the training set to prevent data contamination.

---

## 5. Machine Learning Benchmarking & Results

Five distinct classification models were benchmarked using **5-Fold Stratified Cross-Validation** and evaluated on the out-of-sample holdout test set ($N=5,000$):

| Model | CV ROC-AUC (Mean ± Std) | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC | Composite Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🏆 **Gradient Boosting Classifier** | **0.9644 ± 0.0018** | **90.48%** | **93.17%** | **94.79%** | **0.9397** | **0.9640** | **0.9518 (Rank 1)** |
| 🥈 **Random Forest Classifier** | 0.9627 ± 0.0023 | 90.46% | 92.03% | **96.14%** | **0.9404** | 0.9626 | **0.9515 (Rank 2)** |
| 🥉 **Logistic Regression (Balanced)** | 0.9457 ± 0.0032 | 85.90% | **98.67%** | 83.12% | 0.9023 | 0.9465 | **0.9244 (Rank 3)** |
| 4. **K-Nearest Neighbors (KNN)** | 0.9114 ± 0.0046 | 89.20% | 93.04% | 93.18% | 0.9311 | 0.9055 | **0.9183 (Rank 4)** |
| 5. **AdaBoost Classifier** | 0.7811 ± 0.0021 | 78.30% | 78.30% | 100.00% | 0.8783 | 0.7821 | **0.8302 (Rank 5)** |

### 🏆 Best-Fit Model Selection:
- **Gradient Boosting Classifier** demonstrated superior performance across all evaluation metrics, achieving a **0.9640 Test ROC-AUC**, **90.48% Accuracy**, and **0.9397 F1-Score**.
- **Random Forest** performed virtually identically, delivering a slightly higher Recall (96.14%) at 0.9626 ROC-AUC.
- **Logistic Regression** excelled in Precision (98.67%), rarely misclassifying a hybrid as a pure BEV.

---

## 6. Model Interpretability & Feature Importance

Analysis of Gradient Boosting feature importances highlights the critical drivers of vehicle powertrain technology:
1. `Is_Tesla` (38.4%): Tesla's massive volume and exclusive BEV architecture make this the single strongest predictor.
2. `Make_Grouped_JEEP` (14.2%): Jeep's 4xe hybrid models represent a major PHEV concentration.
3. `Make_Grouped_CHEVROLET` (11.8%): Chevrolet's Bolt EV line heavily skews toward pure electric.
4. `Make_Grouped_BMW` (9.5%): BMW's high volume of PHEV sedans and SUVs strongly signals hybrid powertrain.
5. `Vehicle_Age` / `Model Year` (7.8%): Newer model years have an exponentially higher probability of being pure BEVs.
6. `Make_Grouped_VOLVO` (5.1%): Volvo Recharge hybrid models heavily influence PHEV classification.

---

## 7. Strategic Recommendations & Clean Energy Roadmap

### 1. For Automakers:
- **PHEV Battery Right-Sizing**: European automakers (BMW, Volvo, Jeep) must increase standard PHEV battery packs from ~12 kWh to at least 18–20 kWh to deliver 40+ miles of electric range, ensuring compliance with clean vehicle incentive programs.
- **BEV Platform Modernization**: Automakers should follow Tesla and Chevrolet in consolidating around dedicated 800V native BEV architectures rather than adapting legacy ICE chassis.

### 2. For Utilities & Infrastructure Planners:
- **Targeted Fast-Charging Expansion**: With over 51.9% of EVs concentrated in King County, charging desert areas in Eastern and Central Washington require immediate Level 3 DC Fast Charging corridors along interstate highways.
- **Smart Time-of-Use (TOU) Rates**: Puget Sound Energy and Seattle City Light should implement dynamic overnight EV charging rates to absorb renewable hydro and wind energy during off-peak hours.

### 3. For Transportation Policymakers:
- **Incentive Threshold Revisions**: Maintain the 30-mile CAFV electric range threshold to prevent low-utility PHEVs from claiming state and federal subsidies, while providing point-of-sale vouchers for low-to-middle income BEV purchasers.

---

## 8. Final Deliverables & GitHub / Social Artifacts

- **Raw & Processed Data**: `data/raw/Electric_Vehicle_Population_Data.csv` and `data/processed/`
- **Jupyter Notebook**: Fully executed and pre-computed with inline plots (`notebooks/electric_vehicle_prediction_and_analysis.ipynb`)
- **Modular Pipeline**: Full clean source code (`src/` and `run_pipeline.py`)
- **Serialized Model Artifacts**: Trained weights and scalers (`outputs/models/best_model.joblib`)
- **Visualizations**: 10 publication-quality PNG figures (`outputs/figures/`)
- **Social & Submission Proof**: Guides and checklist located in `report/screenshots/README.md`
- **GitHub Repository**: [https://github.com/Shubhamydv1-4169/Electric-Vehicle-Prediction](https://github.com/Shubhamydv1-4169/Electric-Vehicle-Prediction)
