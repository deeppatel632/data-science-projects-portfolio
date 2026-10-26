"""Exploratory data analysis functions for customer shopping sessions."""

from typing import Any, Dict

import pandas as pd

MONTH_ORDER = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def compute_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute high-level exploratory KPIs from shopping sessions."""
    total_sessions = len(df)
    purchase_count = int(df["Revenue"].sum())
    non_purchase_count = total_sessions - purchase_count
    conversion_rate = (purchase_count / total_sessions) if total_sessions > 0 else 0.0

    avg_product_pages = float(df["ProductRelated"].mean())
    avg_product_duration_min = float(df["ProductRelated_Duration"].mean() / 60.0)
    avg_bounce_rate = float(df["BounceRates"].mean())
    avg_exit_rate = float(df["ExitRates"].mean())
    avg_page_value = float(df["PageValues"].mean())

    weekend_sessions = int(df["Weekend"].sum())
    weekend_pct = (weekend_sessions / total_sessions) if total_sessions > 0 else 0.0

    return {
        "total_sessions": total_sessions,
        "purchase_count": purchase_count,
        "non_purchase_count": non_purchase_count,
        "conversion_rate": conversion_rate,
        "avg_product_pages": avg_product_pages,
        "avg_product_duration_min": avg_product_duration_min,
        "avg_bounce_rate": avg_bounce_rate,
        "avg_exit_rate": avg_exit_rate,
        "avg_page_value": avg_page_value,
        "weekend_sessions": weekend_sessions,
        "weekend_pct": weekend_pct,
    }


def compute_visitor_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze purchasing conversion rate segmented by visitor type."""
    summary = df.groupby("VisitorType", as_index=False).agg(
        total_sessions=("Revenue", "count"),
        purchasing_sessions=("Revenue", "sum"),
        avg_product_pages=("ProductRelated", "mean"),
        avg_bounce_rate=("BounceRates", "mean"),
    )
    summary["conversion_rate"] = summary["purchasing_sessions"] / summary["total_sessions"]
    summary["share_of_sessions"] = summary["total_sessions"] / len(df)
    return summary.sort_values(by="total_sessions", ascending=False)


def compute_monthly_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze seasonality and conversion trends across recorded calendar months."""
    monthly = df.groupby("Month", as_index=False).agg(
        sessions=("Revenue", "count"),
        purchases=("Revenue", "sum"),
        avg_exit_rate=("ExitRates", "mean"),
        avg_page_values=("PageValues", "mean"),
    )
    monthly["conversion_rate"] = monthly["purchases"] / monthly["sessions"]

    # Reorder according to calendar order
    monthly["month_cat"] = pd.Categorical(
        monthly["Month"],
        categories=[m for m in MONTH_ORDER if m in monthly["Month"].values],
        ordered=True
    )
    return monthly.sort_values("month_cat").drop(columns=["month_cat"])


def compute_weekend_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Compare weekday vs weekend session metrics."""
    weekend_df = df.groupby("Weekend", as_index=False).agg(
        sessions=("Revenue", "count"),
        purchases=("Revenue", "sum"),
        avg_product_pages=("ProductRelated", "mean"),
        avg_duration_sec=("ProductRelated_Duration", "mean"),
    )
    weekend_df["conversion_rate"] = weekend_df["purchases"] / weekend_df["sessions"]
    weekend_df["day_type"] = weekend_df["Weekend"].map({True: "Weekend", False: "Weekday"})
    return weekend_df


def generate_eda_observations(df: pd.DataFrame) -> Dict[str, str]:
    """Compute dynamic, factual observations based on the loaded dataset."""
    visitor_summary = compute_visitor_conversion(df)
    new_v = visitor_summary[visitor_summary["VisitorType"] == "New_Visitor"]
    ret_v = visitor_summary[visitor_summary["VisitorType"] == "Returning_Visitor"]

    new_cr = new_v["conversion_rate"].values[0] if len(new_v) > 0 else 0.0
    ret_cr = ret_v["conversion_rate"].values[0] if len(ret_v) > 0 else 0.0

    monthly = compute_monthly_patterns(df)
    peak_vol_month = monthly.loc[monthly["sessions"].idxmax()]["Month"] if len(monthly) > 0 else "N/A"
    peak_cr_month = monthly.loc[monthly["conversion_rate"].idxmax()]["Month"] if len(monthly) > 0 else "N/A"

    zero_bounce = df[df["BounceRates"] == 0]
    zero_bounce_cr = zero_bounce["Revenue"].mean() if len(zero_bounce) > 0 else 0.0
    high_bounce = df[df["BounceRates"] > 0.05]
    high_bounce_cr = high_bounce["Revenue"].mean() if len(high_bounce) > 0 else 0.0

    return {
        "visitor_comparison": (
            f"New visitors converted at {new_cr * 100:.1f}%, while returning visitors converted at "
            f"{ret_cr * 100:.1f}%. While returning visitors drove majority volume "
            f"({ret_v['share_of_sessions'].values[0] * 100:.1f}% of sessions), new visitors exhibited higher direct intent."
        ),
        "seasonality": (
            f"Session volume peaked in {peak_vol_month}, whereas the highest conversion rate was recorded in {peak_cr_month}."
        ),
        "bounce_impact": (
            f"Zero-bounce sessions achieved an empirical conversion rate of {zero_bounce_cr * 100:.1f}%, compared to only "
            f"{high_bounce_cr * 100:.1f}% for sessions with bounce rates exceeding 5%."
        ),
    }
