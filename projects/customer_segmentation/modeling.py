"""Clustering and classification modeling pipelines for session analysis."""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Browsing features for clustering (target Revenue is strictly excluded)
CLUSTERING_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
]

# Classification feature definitions
NUMERIC_FEATURES_NO_LEAKAGE = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "SpecialDay",
]

CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]


def perform_kmeans_clustering(
    df: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
    sample_size: int = 2500,
) -> Dict[str, Any]:
    """Execute K-Means clustering and PCA 2D projection on browsing features."""
    X = df[CLUSTERING_FEATURES].copy().fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Attach cluster assignments
    df_clustered = df.copy()
    df_clustered["Cluster"] = cluster_labels
    df_clustered["Cluster_Name"] = [f"Segment {c + 1}" for c in cluster_labels]

    # Cluster profiles (averages per segment)
    profile_cols = CLUSTERING_FEATURES + ["Revenue"]
    profiles = df_clustered.groupby("Cluster")[profile_cols].mean()
    cluster_counts = df_clustered["Cluster"].value_counts().sort_index()
    profiles["Session_Count"] = cluster_counts
    profiles["Session_Share"] = cluster_counts / len(df)
    profiles["Purchase_Rate"] = profiles["Revenue"]

    # Compute descriptive labels dynamically based on dominant traits
    segment_descriptions = {}
    for c in range(n_clusters):
        row = profiles.loc[c]
        traits = []
        if row["ProductRelated"] > profiles["ProductRelated"].mean() * 1.25:
            traits.append("High Catalog Exploration")
        elif row["ProductRelated"] < profiles["ProductRelated"].mean() * 0.75:
            traits.append("Low Catalog Browsing")

        if row["BounceRates"] > profiles["BounceRates"].mean() * 1.25:
            traits.append("High Bounce / Quick Exit")
        elif row["BounceRates"] < profiles["BounceRates"].mean() * 0.75:
            traits.append("Low Bounce / Engaged")

        if row["Informational"] > profiles["Informational"].mean() * 1.3:
            traits.append("Research & Policy Focused")

        descr = ", ".join(traits) if traits else "Balanced Session Behavior"
        cr_pct = row["Purchase_Rate"] * 100
        segment_descriptions[c] = f"Segment {c + 1}: {descr} (Conversion: {cr_pct:.1f}%)"

    df_clustered["Segment_Description"] = df_clustered["Cluster"].map(segment_descriptions)

    # 2D PCA representation
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(X_scaled)
    explained_var = pca.explained_variance_ratio_

    pca_df = pd.DataFrame({
        "PC1": pca_coords[:, 0],
        "PC2": pca_coords[:, 1],
        "Cluster": [f"Segment {c + 1}" for c in cluster_labels],
        "Cluster_ID": cluster_labels,
        "Revenue": df["Revenue"].values,
        "ProductRelated": df["ProductRelated"].values,
        "BounceRates": df["BounceRates"].values,
        "ExitRates": df["ExitRates"].values,
    })

    # Sample for smooth visualization rendering
    if len(pca_df) > sample_size:
        pca_sample = pca_df.sample(n=sample_size, random_state=random_state)
    else:
        pca_sample = pca_df

    # PCA Centroids
    centroids_scaled = kmeans.cluster_centers_
    centroids_pca = pca.transform(centroids_scaled)
    centroids_df = pd.DataFrame({
        "PC1": centroids_pca[:, 0],
        "PC2": centroids_pca[:, 1],
        "Cluster": [f"Segment {c + 1}" for c in range(n_clusters)],
    })

    return {
        "model": kmeans,
        "scaler": scaler,
        "profiles": profiles,
        "segment_descriptions": segment_descriptions,
        "df_assigned": df_clustered,
        "pca_sample": pca_sample,
        "centroids_pca": centroids_df,
        "explained_variance": explained_var,
        "features": CLUSTERING_FEATURES,
    }


def build_preprocessing_pipeline(include_page_values: bool = False) -> Tuple[Pipeline, List[str]]:
    """Construct an honest scikit-learn ColumnTransformer pipeline."""
    numeric_cols = list(NUMERIC_FEATURES_NO_LEAKAGE)
    if include_page_values:
        numeric_cols.append("PageValues")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor, numeric_cols + CATEGORICAL_FEATURES


def train_classification_models(
    df: pd.DataFrame,
    include_page_values: bool = False,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train and evaluate Logistic Regression and Random Forest models with stratified holdout."""
    preprocessor, feature_names = build_preprocessing_pipeline(include_page_values=include_page_values)

    X = df[feature_names].copy()
    y = df["Revenue"].astype(int).values

    # Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state
            )),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=120,
                max_depth=10,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1
            )),
        ]),
    }

    results = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.50).astype(int)

        roc_auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)

        results[name] = {
            "pipeline": pipe,
            "y_prob": y_prob,
            "y_pred": y_pred,
            "metrics_at_50": {
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc,
            },
            "roc_curve": (fpr, tpr),
            "pr_curve": (precision_curve, recall_curve),
            "cm": confusion_matrix(y_test, y_pred),
        }

    # Extract feature names after one-hot encoding for feature importance
    fitted_preprocessor = models["Random Forest"].named_steps["preprocessor"]
    cat_encoder = fitted_preprocessor.named_transformers_["cat"]
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    all_feature_names = np.concatenate([
        fitted_preprocessor.transformers_[0][2],  # numeric cols
        cat_feature_names
    ])

    rf_clf = models["Random Forest"].named_steps["classifier"]
    rf_importances = pd.Series(rf_clf.feature_importances_, index=all_feature_names).sort_values(ascending=False)

    return {
        "results": results,
        "y_test": y_test,
        "feature_importances": rf_importances,
        "include_page_values": include_page_values,
        "test_size": len(y_test),
        "test_positive_count": int(np.sum(y_test)),
    }


def predict_with_threshold(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Calculate evaluation metrics under an interactive decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": threshold,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "cm": cm,
    }
