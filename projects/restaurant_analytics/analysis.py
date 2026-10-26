"""Business metric computations, menu engineering, ABC analysis, and baseline forecasting."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def compute_business_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate executive business metrics ensuring multi-line order integrity."""
    if len(df) == 0:
        return {
            "total_revenue": 0.0,
            "total_orders": 0,
            "total_units": 0,
            "aov": 0.0,
            "total_gross_profit": 0.0,
            "gross_margin": 0.0,
            "avg_items_per_order": 0.0,
            "total_discounts": 0.0,
        }

    total_revenue = float(df["net_revenue"].sum())
    total_orders = int(df["order_id"].nunique())
    total_units = int(df["quantity"].sum())
    total_gross_profit = float(df["estimated_gross_profit"].sum())
    total_discounts = float(df["discount_amount"].sum())

    aov = (total_revenue / total_orders) if total_orders > 0 else 0.0
    gross_margin = (total_gross_profit / total_revenue) if total_revenue > 0 else 0.0
    avg_items_per_order = (total_units / total_orders) if total_orders > 0 else 0.0

    return {
        "total_revenue": total_revenue,
        "total_orders": total_orders,
        "total_units": total_units,
        "aov": aov,
        "total_gross_profit": total_gross_profit,
        "gross_margin": gross_margin,
        "avg_items_per_order": avg_items_per_order,
        "total_discounts": total_discounts,
    }


def compute_daily_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily net revenue, distinct orders, and gross profit chronologically."""
    daily = df.groupby("date", as_index=False).agg(
        revenue=("net_revenue", "sum"),
        orders=("order_id", "nunique"),
        profit=("estimated_gross_profit", "sum"),
        units=("quantity", "sum"),
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")
    daily["aov"] = daily["revenue"] / daily["orders"]
    daily["margin"] = daily["profit"] / daily["revenue"]
    return daily


def compute_monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly sales and order volumes."""
    temp = df.copy()
    temp["month_year"] = pd.to_datetime(temp["order_timestamp"]).dt.to_period("M").astype(str)
    monthly = temp.groupby("month_year", as_index=False).agg(
        revenue=("net_revenue", "sum"),
        orders=("order_id", "nunique"),
        profit=("estimated_gross_profit", "sum"),
    )
    monthly["aov"] = monthly["revenue"] / monthly["orders"]
    return monthly.sort_values("month_year")


def compute_outlet_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Performance breakdown segmented by outlet."""
    outlet_df = df.groupby("outlet", as_index=False).agg(
        revenue=("net_revenue", "sum"),
        orders=("order_id", "nunique"),
        profit=("estimated_gross_profit", "sum"),
        units=("quantity", "sum"),
    )
    outlet_df["aov"] = outlet_df["revenue"] / outlet_df["orders"]
    outlet_df["margin"] = outlet_df["profit"] / outlet_df["revenue"]
    outlet_df["share_of_sales"] = outlet_df["revenue"] / df["net_revenue"].sum()
    return outlet_df.sort_values("revenue", ascending=False)


def compute_channel_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Performance breakdown segmented by ordering channel."""
    channel_df = df.groupby("ordering_channel", as_index=False).agg(
        revenue=("net_revenue", "sum"),
        orders=("order_id", "nunique"),
        profit=("estimated_gross_profit", "sum"),
    )
    channel_df["aov"] = channel_df["revenue"] / channel_df["orders"]
    channel_df["margin"] = channel_df["profit"] / channel_df["revenue"]
    channel_df["share_of_sales"] = channel_df["revenue"] / df["net_revenue"].sum()
    return channel_df.sort_values("revenue", ascending=False)


def compute_hourly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Order volume and revenue distribution across hours of the day (0–23)."""
    hourly = df.groupby("hour", as_index=False).agg(
        revenue=("net_revenue", "sum"),
        orders=("order_id", "nunique"),
    )
    # Ensure full 24-hour range
    full_hours = pd.DataFrame({"hour": range(24)})
    merged = full_hours.merge(hourly, on="hour", how="left").fillna(0)
    return merged.sort_values("hour")


def compute_weekday_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Sales performance by day of the week in standard Monday-Sunday sequence."""
    weekday = df.groupby("day_of_week", as_index=False).agg(
        revenue=("net_revenue", "sum"),
        orders=("order_id", "nunique"),
        profit=("estimated_gross_profit", "sum"),
    )
    weekday["day_cat"] = pd.Categorical(weekday["day_of_week"], categories=DAY_ORDER, ordered=True)
    return weekday.sort_values("day_cat").drop(columns=["day_cat"])


def compute_menu_item_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Detailed item-level profitability and volume analysis."""
    items = df.groupby(["item_name", "category"], as_index=False).agg(
        units_sold=("quantity", "sum"),
        gross_sales=("gross_sales", "sum"),
        net_revenue=("net_revenue", "sum"),
        cogs=("cogs", "sum"),
        estimated_gross_profit=("estimated_gross_profit", "sum"),
        avg_price=("unit_price", "mean"),
        avg_cost=("unit_cost", "mean"),
    )
    total_rev = items["net_revenue"].sum()
    items["gross_margin"] = items["estimated_gross_profit"] / items["net_revenue"]
    items["revenue_share"] = items["net_revenue"] / total_rev if total_rev > 0 else 0.0
    return items.sort_values("net_revenue", ascending=False)


def compute_abc_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Perform classic ABC menu analysis based on cumulative revenue contribution.

    - Class A: Top contributors generating first 70% of cumulative revenue.
    - Class B: Intermediate contributors generating next 20% of revenue (70% - 90%).
    - Class C: Long-tail items generating bottom 10% of revenue (90% - 100%).
    """
    items = compute_menu_item_performance(df).copy()
    items = items.sort_values("net_revenue", ascending=False).reset_index(drop=True)
    items["cumulative_revenue"] = items["net_revenue"].cumsum()
    items["cumulative_share"] = items["cumulative_revenue"] / items["net_revenue"].sum()

    def assign_class(share: float) -> str:
        if share <= 0.70:
            return "Class A (Top 70%)"
        elif share <= 0.90:
            return "Class B (Next 20%)"
        return "Class C (Tail 10%)"

    items["abc_class"] = items["cumulative_share"].apply(assign_class)
    return items


def generate_operational_insights(df: pd.DataFrame) -> Dict[str, str]:
    """Compute explicit, rule-based operational insights directly from dataset metrics."""
    if len(df) == 0:
        return {}

    hourly = compute_hourly_sales(df)
    peak_hour = int(hourly.loc[hourly["revenue"].idxmax()]["hour"])
    peak_hour_str = f"{peak_hour:02d}:00 – {peak_hour+1:02d}:00"

    outlets = compute_outlet_sales(df)
    top_outlet = outlets.iloc[0]["outlet"]
    top_outlet_share = outlets.iloc[0]["share_of_sales"] * 100

    items = compute_menu_item_performance(df)
    top_rev_item = items.iloc[0]["item_name"]
    top_vol_item = items.sort_values("units_sold", ascending=False).iloc[0]["item_name"]

    cats = df.groupby("category")["estimated_gross_profit"].sum() / df.groupby("category")["net_revenue"].sum()
    highest_margin_cat = cats.idxmax()
    highest_margin_pct = cats.max() * 100

    weekday_df = compute_weekday_sales(df)
    fri_sat = weekday_df[weekday_df["day_of_week"].isin(["Friday", "Saturday"])]["revenue"].sum()
    mon_tue = weekday_df[weekday_df["day_of_week"].isin(["Monday", "Tuesday"])]["revenue"].sum()
    weekend_lift = ((fri_sat / 2) - (mon_tue / 2)) / (mon_tue / 2) * 100 if mon_tue > 0 else 0.0

    return {
        "rush_hour": (
            f"Peak sales volume occurs during the dinner window at **{peak_hour_str}**, "
            f"representing the highest hourly transaction concentration."
        ),
        "outlet_dominance": (
            f"**{top_outlet}** is the leading revenue driver, accounting for "
            f"**{top_outlet_share:.1f}%** of total portfolio sales."
        ),
        "menu_heroes": (
            f"The highest grossing product is **{top_rev_item}**, while the highest unit volume driver is "
            f"**{top_vol_item}**. The **{highest_margin_cat}** category produces the healthiest profitability "
            f"with an estimated gross margin of **{highest_margin_pct:.1f}%**."
        ),
        "weekend_pattern": (
            f"Weekend daily revenue (Fri/Sat) is **{weekend_lift:.1f}% higher** on average than early weekday (Mon/Tue) daily sales."
        ),
    }


def compute_sales_forecast(
    df: pd.DataFrame,
    forecast_days: int = 14,
    rolling_window: int = 7
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Compute chronological rolling-average trend forecast for daily revenue."""
    daily = compute_daily_trends(df)

    if len(daily) < (rolling_window * 2):
        return None, (
            f"Selected date range has only {len(daily)} days of data. "
            f"At least {rolling_window * 2} daily observations are required to construct a valid chronological forecast."
        )

    daily["rolling_avg"] = daily["revenue"].rolling(window=rolling_window, min_periods=rolling_window).mean()

    # Simple linear baseline projection using recent trend
    recent = daily.tail(rolling_window * 2)
    x = np.arange(len(recent))
    y = recent["revenue"].values
    slope, intercept = np.polyfit(x, y, 1)

    last_date = daily["date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq="D")

    future_x = np.arange(len(recent), len(recent) + forecast_days)
    future_projected = np.clip(slope * future_x + intercept, 0, None)

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "projected_revenue": future_projected,
        "baseline_rolling": [daily["rolling_avg"].iloc[-1]] * forecast_days,
        "is_forecast": True,
    })

    combined = pd.concat([
        daily[["date", "revenue", "rolling_avg"]].assign(is_forecast=False),
        forecast_df
    ], ignore_index=True)

    return combined, None
