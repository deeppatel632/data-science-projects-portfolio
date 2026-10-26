"""Unit and integration tests for student performance analysis project."""

import pandas as pd
import pytest

try:
    from projects.student_performance.analysis import (
        compute_academic_overview,
        compute_correlations,
        compute_group_comparisons,
    )
    from projects.student_performance.data import (
        EXPECTED_COLUMNS,
        generate_synthetic_student_data,
        load_student_data,
    )
    from projects.student_performance.modeling import (
        REGRESSION_FEATURES_EARLY,
        REGRESSION_FEATURES_PREVIOUS,
        predict_single_student,
        train_regression_models,
    )
except ImportError:
    from analysis import (
        compute_academic_overview,
        compute_correlations,
    )
    from data import (
        EXPECTED_COLUMNS,
        generate_synthetic_student_data,
        load_student_data,
    )
    from modeling import (
        REGRESSION_FEATURES_EARLY,
        REGRESSION_FEATURES_PREVIOUS,
        predict_single_student,
        train_regression_models,
    )


def test_load_student_data_both_subjects():
    """Verify loading functions work for both Math and Portuguese cohorts."""
    df_mat, meta_mat = load_student_data("mat")
    assert isinstance(df_mat, pd.DataFrame)
    assert len(df_mat) >= 300
    assert "G3" in df_mat.columns
    for col in EXPECTED_COLUMNS:
        assert col in df_mat.columns

    df_por, meta_por = load_student_data("por")
    assert isinstance(df_por, pd.DataFrame)
    assert len(df_por) >= 500
    assert "G3" in df_por.columns


def test_synthetic_student_data_reproducibility():
    """Verify synthetic data generator is deterministic with fixed seed."""
    df1 = generate_synthetic_student_data(subject="mat", n_samples=200, seed=42)
    df2 = generate_synthetic_student_data(subject="mat", n_samples=200, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_compute_academic_overview():
    """Verify academic overview indicators."""
    df, _ = load_student_data("mat")
    overview = compute_academic_overview(df, pass_threshold=10)

    assert overview["total_students"] == len(df)
    assert 0.0 <= overview["mean_grade"] <= 20.0
    assert 0.0 <= overview["median_grade"] <= 20.0
    assert 0.0 <= overview["pass_rate"] <= 1.0
    assert overview["passed_count"] <= overview["total_students"]


def test_compute_correlations():
    """Verify pairwise correlations are bounded and valid."""
    df, _ = load_student_data("mat")
    corr = compute_correlations(df)
    assert isinstance(corr, pd.DataFrame)
    assert "G1" in corr.columns and "G3" in corr.columns
    assert corr.loc["G1", "G1"] == pytest.approx(1.0)
    assert -1.0 <= corr.loc["G1", "G3"] <= 1.0


def test_regression_pipeline_early_vs_previous():
    """Verify regression pipelines train correctly across early and previous-grades scenarios."""
    df = generate_synthetic_student_data("mat", n_samples=300, seed=42)

    # Scenario 1: Early-support (no G1, G2)
    models_early = train_regression_models(df, include_previous_grades=False, random_state=42)
    assert "G3" not in REGRESSION_FEATURES_EARLY
    assert "G1" not in REGRESSION_FEATURES_EARLY
    assert "Dummy Baseline (Mean)" in models_early["results"]
    assert "Linear Regression (Ridge)" in models_early["results"]
    assert "Random Forest Regressor" in models_early["results"]

    # Scenario 2: Previous-grades (with G1, G2)
    models_prev = train_regression_models(df, include_previous_grades=True, random_state=42)
    assert "G1" in REGRESSION_FEATURES_PREVIOUS
    assert "G2" in REGRESSION_FEATURES_PREVIOUS
    assert "G3" not in REGRESSION_FEATURES_PREVIOUS

    # Modeling honesty verification: having G1 and G2 must lower test RMSE
    rf_early_rmse = models_early["results"]["Random Forest Regressor"]["rmse"]
    rf_prev_rmse = models_prev["results"]["Random Forest Regressor"]["rmse"]
    assert rf_prev_rmse < rf_early_rmse, "Models with G1/G2 should yield lower error than early models"


def test_predict_single_student():
    """Verify single-student prediction returns bounded grade estimates."""
    df = generate_synthetic_student_data("mat", n_samples=300, seed=42)
    models_prev = train_regression_models(df, include_previous_grades=True, random_state=42)
    pipeline = models_prev["results"]["Linear Regression (Ridge)"]["pipeline"]

    sample_student = df.iloc[0].to_dict()
    pred_g3 = predict_single_student(pipeline, sample_student)

    assert isinstance(pred_g3, float)
    assert 0.0 <= pred_g3 <= 20.0
