"""Unit and integration tests for restaurant analytics project."""

import pandas as pd
import pytest

try:
    from projects.restaurant_analytics.analysis import (
        compute_abc_analysis,
        compute_business_kpis,
        compute_channel_sales,
        compute_hourly_sales,
        compute_menu_item_performance,
        compute_outlet_sales,
        compute_sales_forecast,
        generate_operational_insights,
    )
    from projects.restaurant_analytics.data import (
        REQUIRED_COLUMNS,
        generate_synthetic_restaurant_data,
        load_restaurant_data,
        validate_restaurant_dataframe,
    )
except ImportError:
    from analysis import (
        compute_abc_analysis,
        compute_business_kpis,
        compute_channel_sales,
        compute_outlet_sales,
        compute_sales_forecast,
    )
    from data import (
        REQUIRED_COLUMNS,
        generate_synthetic_restaurant_data,
        load_restaurant_data,
        validate_restaurant_dataframe,
    )


def test_load_restaurant_data():
    """Verify default demonstration dataset loads with correct columns and non-empty rows."""
    df, meta = load_restaurant_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 5000
    for col in REQUIRED_COLUMNS:
        assert col in df.columns
    assert meta["is_synthetic"] is True


def test_synthetic_restaurant_reproducibility():
    """Verify synthetic generation is deterministic with fixed seed."""
    df1 = generate_synthetic_restaurant_data(n_orders=200, seed=42)
    df2 = generate_synthetic_restaurant_data(n_orders=200, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_validate_restaurant_dataframe():
    """Verify schema validator checks required columns and data types."""
    df_valid = generate_synthetic_restaurant_data(n_orders=100, seed=1)
    is_valid, msg = validate_restaurant_dataframe(df_valid)
    assert is_valid is True

    # Drop required column to test failure
    df_invalid = df_valid.drop(columns=["net_revenue"])
    is_valid, msg = validate_restaurant_dataframe(df_invalid)
    assert is_valid is False
    assert "net_revenue" in msg


def test_order_count_integrity_and_kpis():
    """Ensure multi-line orders are counted as single orders rather than individual rows."""
    df, _ = load_restaurant_data()
    kpis = compute_business_kpis(df)

    assert kpis["total_orders"] == df["order_id"].nunique()
    # Total rows must be greater than distinct orders
    assert len(df) > kpis["total_orders"]
    assert kpis["avg_items_per_order"] > 1.0

    # Financial integrity assertions
    assert kpis["total_revenue"] == pytest.approx(df["net_revenue"].sum(), rel=1e-3)
    assert kpis["total_gross_profit"] == pytest.approx(df["estimated_gross_profit"].sum(), rel=1e-3)
    assert kpis["aov"] == pytest.approx(kpis["total_revenue"] / kpis["total_orders"], rel=1e-3)
    assert 0.0 <= kpis["gross_margin"] <= 1.0


def test_outlet_and_channel_sales():
    """Verify outlet and channel breakdowns correctly sum to cohort totals."""
    df, _ = load_restaurant_data()
    total_rev = df["net_revenue"].sum()

    outlet_df = compute_outlet_sales(df)
    assert len(outlet_df) == 4
    assert outlet_df["revenue"].sum() == pytest.approx(total_rev, rel=1e-3)
    assert outlet_df["share_of_sales"].sum() == pytest.approx(1.0, rel=1e-3)

    channel_df = compute_channel_sales(df)
    assert len(channel_df) == 4
    assert channel_df["revenue"].sum() == pytest.approx(total_rev, rel=1e-3)


def test_menu_abc_analysis():
    """Verify ABC classification applies Pareto rule properly."""
    df, _ = load_restaurant_data()
    abc_df = compute_abc_analysis(df)

    assert "abc_class" in abc_df.columns
    classes = set(abc_df["abc_class"].unique())
    assert "Class A (Top 70%)" in classes
    assert (abc_df["gross_margin"] >= 0).all(), "No item should have negative margin unless cost > price"
    # Cumulative share must be sorted monotonic increasing
    assert abc_df["cumulative_share"].is_monotonic_increasing


def test_compute_sales_forecast_short_history_guard():
    """Verify chronological forecast produces data for valid ranges and guards short windows."""
    df, _ = load_restaurant_data()
    # 1. Full data produces valid forecast
    combined, err = compute_sales_forecast(df, forecast_days=14, rolling_window=7)
    assert err is None
    assert combined is not None
    assert len(combined[combined["is_forecast"]]) == 14

    # 2. Short date slice triggers safety guard
    short_df = df[df["date"] == df["date"].min()]
    combined_short, err_short = compute_sales_forecast(short_df, forecast_days=14, rolling_window=7)
    assert combined_short is None
    assert err_short is not None
    assert "required" in err_short
