"""Unit and integration tests for customer segmentation project."""

import numpy as np
import pandas as pd

try:
    from projects.customer_segmentation.analysis import (
        compute_dataset_summary,
        compute_monthly_patterns,
        compute_visitor_conversion,
    )
    from projects.customer_segmentation.data import (
        EXPECTED_COLUMNS,
        generate_synthetic_shoppers_data,
        load_shoppers_data,
        validate_shoppers_dataframe,
    )
    from projects.customer_segmentation.modeling import (
        CLUSTERING_FEATURES,
        perform_kmeans_clustering,
        predict_with_threshold,
        train_classification_models,
    )
except ImportError:
    from analysis import (
        compute_dataset_summary,
    )
    from data import (
        EXPECTED_COLUMNS,
        generate_synthetic_shoppers_data,
        load_shoppers_data,
        validate_shoppers_dataframe,
    )
    from modeling import (
        perform_kmeans_clustering,
        predict_with_threshold,
        train_classification_models,
    )


def test_load_shoppers_data():
    """Verify that dataset loads successfully with required columns."""
    df, meta = load_shoppers_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 1000
    for col in EXPECTED_COLUMNS:
        assert col in df.columns
    assert "Revenue" in df.columns
    assert df["Revenue"].dtype == bool or np.issubdtype(df["Revenue"].dtype, np.integer)


def test_validate_shoppers_dataframe():
    """Verify schema validator handles valid and invalid schemas."""
    df_valid = generate_synthetic_shoppers_data(n_samples=200, seed=1)
    is_valid, msg = validate_shoppers_dataframe(df_valid)
    assert is_valid is True

    # Drop a column to test failure
    df_invalid = df_valid.drop(columns=["Revenue"])
    is_valid, msg = validate_shoppers_dataframe(df_invalid)
    assert is_valid is False
    assert "Revenue" in msg


def test_synthetic_data_reproducibility():
    """Verify synthetic dataset generator is deterministic with fixed seed."""
    df1 = generate_synthetic_shoppers_data(n_samples=500, seed=42)
    df2 = generate_synthetic_shoppers_data(n_samples=500, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_compute_dataset_summary():
    """Verify summary calculations produce consistent, mathematically sound metrics."""
    df, _ = load_shoppers_data()
    summary = compute_dataset_summary(df)

    assert summary["total_sessions"] == len(df)
    assert 0.0 <= summary["conversion_rate"] <= 1.0
    assert summary["purchase_count"] + summary["non_purchase_count"] == summary["total_sessions"]
    assert summary["avg_bounce_rate"] >= 0.0
    assert summary["avg_exit_rate"] >= summary["avg_bounce_rate"]  # Exit rates are >= bounce rates


def test_kmeans_clustering():
    """Verify K-means clustering produces expected cluster counts and shapes."""
    df = generate_synthetic_shoppers_data(n_samples=600, seed=42)
    k = 3
    results = perform_kmeans_clustering(df, n_clusters=k, random_state=42)

    assert len(results["profiles"]) == k
    assert "Cluster" in results["df_assigned"].columns
    assert set(results["df_assigned"]["Cluster"].unique()) == {0, 1, 2}
    # Target variable must NOT be in clustering features
    assert "Revenue" not in results["features"]
    assert len(results["pca_sample"]) <= len(df)


def test_classification_models_honest_leakage_exclusion():
    """Verify classification training runs without PageValues and evaluates properly."""
    df = generate_synthetic_shoppers_data(n_samples=800, seed=42)
    models_out = train_classification_models(df, include_page_values=False, random_state=42)

    assert "Logistic Regression" in models_out["results"]
    assert "Random Forest" in models_out["results"]

    for name in ["Logistic Regression", "Random Forest"]:
        metrics = models_out["results"][name]["metrics_at_50"]
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.5 <= metrics["roc_auc"] <= 1.0  # Better than or equal to random


def test_predict_with_threshold():
    """Verify threshold tuning shifts precision and recall logically."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9])

    # Low threshold -> High recall
    low_res = predict_with_threshold(y_true, y_prob, threshold=0.25)
    # High threshold -> Higher precision, lower recall
    high_res = predict_with_threshold(y_true, y_prob, threshold=0.75)

    assert low_res["recall"] >= high_res["recall"]
    assert high_res["precision"] >= low_res["precision"]
