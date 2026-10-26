# Customer Behavioral Segmentation and Purchase Intent Prediction

An end-to-end e-commerce analytics project combining session-level exploratory analysis, unsupervised behavioral clustering, and leakage-conscious purchase intention classification.

## 1. Problem Statement & Objectives
Modern e-commerce platforms observe thousands of anonymous visitor sessions every day. To personalize browsing interventions and allocate marketing resources, digital analysts need to:
1. Discover natural behavioral groupings across browsing patterns (pages viewed, session durations, abandonment signals).
2. Predict in real-time whether an ongoing session is likely to conclude with a purchase transaction (`Revenue = True`).
3. Ensure predictive workflows do not introduce artificial target leakage or confuse session behaviors with persistent customer identities.

## 2. Dataset Attribution & Provenance
- **Dataset:** UCI Online Shoppers Purchasing Intention Dataset
- **Repository Source:** [https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- **Reference:** Sakar, C.O., Polat, S.O., Katircioglu, M. et al. (2019). *Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks.* Neural Computing and Applications, 31, 6893–6908.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Scope:** 12,330 distinct web sessions over a 1-year period (10 calendar months represented).

## 3. Data Architecture & Preprocessing
- **Features Used for Clustering:** 8 numerical browsing metrics:
  - `Administrative`, `Administrative_Duration`
  - `Informational`, `Informational_Duration`
  - `ProductRelated`, `ProductRelated_Duration`
  - `BounceRates`, `ExitRates`
  *(The purchase outcome `Revenue` is strictly excluded from clustering inputs).*
- **Target Leakage Prevention:**
  - `PageValues` (average page value before transaction) is excluded from baseline predictive features because in real-world logging systems it frequently relies on post-transaction event attribution, artificially inflating apparent ROC-AUC. An educational toggle is provided in the UI to demonstrate this effect.
- **Pipeline Structure:**
  - Standard scaling (`StandardScaler`) for numerical variables.
  - One-hot encoding (`OneHotEncoder(handle_unknown='ignore')`) for categorical fields (`Month`, `VisitorType`, `Weekend`, `OperatingSystems`, `Browser`, `TrafficType`).
  - Class balancing via `class_weight='balanced'` to accommodate the 15.5% purchase base rate.

## 4. Modeling Methodologies
### A. Unsupervised Behavioral Segmentation
- **Algorithm:** K-Means clustering ($k \in [2, 6]$, default $k = 4$, seed 42).
- **Dimensionality Reduction:** 2-Component Principal Component Analysis (PCA) for visual cluster separation and centroid tracking.
- **Dynamic Segment Labeling:** Clusters are annotated by their empirical profile (e.g., High Catalog Exploration, Quick Exit / High Bounce, Research Focused).

### B. Supervised Purchase Intent Classification
- **Holdout Evaluation:** 80/20 stratified train/test split.
- **Algorithms:**
  - **Logistic Regression:** Linear decision boundary, high interpretability, calibrated probabilities.
  - **Random Forest:** Ensemble of 120 trees (max depth 10) capturing non-linear feature interactions.
- **Metrics Evaluated:** Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
- **Threshold Simulator:** Interactive threshold slider ($0.10$ to $0.90$) allowing operators to tune for high recall (sales intervention) or high precision (prevent coupon wastage).

## 5. Key Findings
- **Exit Rate as Primary Negative Predictor:** Higher exit rates on product pages provide the single strongest discriminating signal of session abandonment.
- **Visitor Intent Asymmetry:** First-time visitors exhibit higher conversion rates (~24%) than returning visitors (~14%), likely indicating targeted search or intent-driven landing.
- **Cluster Conversion Disparities:** Unsupervised segments vary dramatically in conversion, from under 2% in bounce-heavy clusters to over 28% in deep catalog explorers.

## 6. Limitations & Governance
- **Session-Level vs. Customer-Level:** The dataset tracks individual browser sessions, not persistent human identities. A single customer browsing on mobile and buying on desktop appears as two separate sessions.
- **Non-Causal Interpretability:** Longer product duration is associated with purchase, but does not prove that making users stay on a page artificially causes purchase.

## 7. How to Run
From the repository root:
```bash
streamlit run projects/customer_segmentation/app.py
```
Or execute unit tests:
```bash
pytest projects/customer_segmentation/tests/
```
