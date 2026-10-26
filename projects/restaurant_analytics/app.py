"""Interactive Streamlit dashboard for Restaurant Performance and Food Sales Analysis."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Add repo root to path for shared imports
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.styles import (
    apply_custom_css,
    render_callout,
    render_metric_card,
    render_page_header,
)
from shared.utilities import (
    format_currency,
    format_number,
    format_percentage,
    safe_dataframe_download,
)

try:
    from projects.restaurant_analytics.analysis import (
        compute_abc_analysis,
        compute_business_kpis,
        compute_channel_sales,
        compute_daily_trends,
        compute_hourly_sales,
        compute_menu_item_performance,
        compute_outlet_sales,
        compute_sales_forecast,
        compute_weekday_sales,
        generate_operational_insights,
    )
    from projects.restaurant_analytics.data import (
        REQUIRED_COLUMNS,
        load_restaurant_data,
        validate_restaurant_dataframe,
    )
    from projects.restaurant_analytics.visualizations import (
        plot_channel_share,
        plot_forecast_vs_actual,
        plot_hourly_distribution,
        plot_menu_item_ranking,
        plot_menu_scatter,
        plot_outlet_breakdown,
        plot_sales_trend,
    )
except ImportError:
    from analysis import (
        compute_abc_analysis,
        compute_business_kpis,
        compute_channel_sales,
        compute_daily_trends,
        compute_hourly_sales,
        compute_menu_item_performance,
        compute_outlet_sales,
        compute_sales_forecast,
        compute_weekday_sales,
        generate_operational_insights,
    )
    from data import (
        REQUIRED_COLUMNS,
        load_restaurant_data,
        validate_restaurant_dataframe,
    )
    from visualizations import (
        plot_channel_share,
        plot_forecast_vs_actual,
        plot_hourly_distribution,
        plot_menu_item_ranking,
        plot_menu_scatter,
        plot_outlet_breakdown,
        plot_sales_trend,
    )

PROJECT_KEY = "restaurant_analytics"


@st.cache_data
def get_cached_restaurant_data():
    """Cache default synthetic restaurant data."""
    return load_restaurant_data()


def render_restaurant_analytics_page():
    """Render the restaurant sales, operations, and menu analytics dashboard."""
    apply_custom_css(PROJECT_KEY)

    render_page_header(
        title="Restaurant Performance & Food Sales Analysis",
        subtitle=(
            "Multi-outlet revenue analysis, menu engineering profitability matrices, "
            "and chronological baseline forecasting for restaurant operations."
        ),
        tag="PROJECT 3 · COMMERCIAL ANALYTICS",
        project_key=PROJECT_KEY,
    )

    # Load baseline demonstration data
    df_default, meta_default = get_cached_restaurant_data()
    active_df = df_default.copy()
    is_custom_upload = False

    # Sidebar: Optional CSV Uploader
    st.sidebar.markdown("### Data Source")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Custom Transactions (CSV)",
        type=["csv"],
        help="Upload compatible restaurant transaction lines matching the project schema.",
    )

    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            is_valid, validation_msg = validate_restaurant_dataframe(user_df)
            if is_valid:
                user_df["order_timestamp"] = pd.to_datetime(user_df["order_timestamp"])
                user_df["date"] = user_df["order_timestamp"].dt.date
                user_df["hour"] = user_df["order_timestamp"].dt.hour
                user_df["day_of_week"] = user_df["order_timestamp"].dt.day_name()
                active_df = user_df
                is_custom_upload = True
                st.sidebar.success(f"Loaded {len(active_df):,} uploaded records.")
            else:
                st.sidebar.error(f"Incompatible CSV: {validation_msg}")
                st.sidebar.info("Preserving default demonstration data.")
        except Exception as e:
            st.sidebar.error(f"Failed to parse uploaded CSV: {e}")
            st.sidebar.info("Preserving default demonstration data.")

    # Data Provenance Banner
    if is_custom_upload:
        st.info("📊 Currently Viewing: User-Uploaded Custom Dataset.")
    else:
        st.caption(
            "✓ Data Mode: **Deterministic Synthetic Demonstration Data** (~18,000 line items across 7,500 orders) · "
            "No external APIs required."
        )

    # Sidebar Filters
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filter Operations")

    all_outlets = sorted(active_df["outlet"].unique())
    selected_outlets = st.sidebar.multiselect("Outlets:", options=all_outlets, default=all_outlets)

    all_channels = sorted(active_df["ordering_channel"].unique())
    selected_channels = st.sidebar.multiselect("Channels:", options=all_channels, default=all_channels)

    all_categories = sorted(active_df["category"].unique())
    selected_categories = st.sidebar.multiselect("Categories:", options=all_categories, default=all_categories)

    min_date = active_df["date"].min()
    max_date = active_df["date"].max()
    date_range = st.sidebar.date_input("Date Range:", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    # Apply filters
    filtered_df = active_df[
        active_df["outlet"].isin(selected_outlets)
        & active_df["ordering_channel"].isin(selected_channels)
        & active_df["category"].isin(selected_categories)
    ]
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_d, end_d = date_range
        filtered_df = filtered_df[
            (filtered_df["date"] >= start_d) & (filtered_df["date"] <= end_d)
        ]

    if len(filtered_df) == 0:
        st.warning("No transactions match the selected filters. Please broaden your selection.")
        return

    # Compute KPIs
    kpis = compute_business_kpis(filtered_df)

    # Previous Period Comparison
    prior_delta_str = ""
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        duration_days = (date_range[1] - date_range[0]).days + 1
        prior_start = date_range[0] - pd.Timedelta(days=duration_days)
        prior_end = date_range[0] - pd.Timedelta(days=1)
        prior_df = active_df[
            active_df["outlet"].isin(selected_outlets)
            & active_df["ordering_channel"].isin(selected_channels)
            & active_df["category"].isin(selected_categories)
            & (active_df["date"] >= prior_start)
            & (active_df["date"] <= prior_end)
        ]
        if len(prior_df) > 0:
            prior_kpis = compute_business_kpis(prior_df)
            rev_diff_pct = (kpis["total_revenue"] - prior_kpis["total_revenue"]) / prior_kpis["total_revenue"] * 100
            prior_delta_str = f"{rev_diff_pct:+.1f}% vs prior {duration_days} days"

    # Dashboard Tabs
    tab_overview, tab_sales, tab_menu, tab_ops, tab_forecast, tab_data = st.tabs([
        "1. Business Overview",
        "2. Sales Performance",
        "3. Menu Engineering",
        "4. Operational Insights",
        "5. Sales Forecast Baseline",
        "6. Data Explorer & Schema",
    ])

    # -------------------------------------------------------------
    # TAB 1: BUSINESS OVERVIEW
    # -------------------------------------------------------------
    with tab_overview:
        st.markdown("### Executive Financial & Volume Indicators")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_metric_card(
                title="Total Net Revenue",
                value=format_currency(kpis["total_revenue"]),
                sub=prior_delta_str or f"Gross sales: {format_currency(filtered_df['gross_sales'].sum())}",
            )
        with c2:
            render_metric_card(
                title="Total Distinct Orders",
                value=format_number(kpis["total_orders"]),
                sub=f"{format_number(kpis['total_units'])} units sold across orders",
            )
        with c3:
            render_metric_card(
                title="Average Order Value (AOV)",
                value=format_currency(kpis["aov"]),
                sub=f"{kpis['avg_items_per_order']:.1f} items per transaction",
            )
        with c4:
            render_metric_card(
                title="Estimated Gross Margin",
                value=format_percentage(kpis["gross_margin"]),
                sub=f"Gross profit: {format_currency(kpis['total_gross_profit'])}",
            )

        daily_df = compute_daily_trends(filtered_df)
        st.plotly_chart(plot_sales_trend(daily_df), use_container_width=True)

        render_callout(
            text=(
                "Financial calculations strictly distinguish <strong>Order IDs</strong> (which span multiple line items) "
                "from individual menu rows. Discounts are deducted proportionally before calculating Net Revenue and "
                "Estimated Gross Profit."
            ),
            title="Accounting Integrity Notice",
            project_key=PROJECT_KEY,
        )

    # -------------------------------------------------------------
    # TAB 2: SALES PERFORMANCE
    # -------------------------------------------------------------
    with tab_sales:
        st.markdown("### Temporal & Channel Breakdown")

        col_out, col_chan = st.columns(2)
        with col_out:
            outlet_df = compute_outlet_sales(filtered_df)
            st.plotly_chart(plot_outlet_breakdown(outlet_df), use_container_width=True)
        with col_chan:
            channel_df = compute_channel_sales(filtered_df)
            st.plotly_chart(plot_channel_share(channel_df), use_container_width=True)

        col_hour, col_week = st.columns([1.3, 1])
        with col_hour:
            hourly_df = compute_hourly_sales(filtered_df)
            st.plotly_chart(plot_hourly_distribution(hourly_df), use_container_width=True)
        with col_week:
            weekday_df = compute_weekday_sales(filtered_df)
            st.markdown("#### Day-of-Week Sales Pattern")
            st.dataframe(
                weekday_df.style.format({
                    "revenue": "${:,.0f}",
                    "orders": "{:,}",
                    "profit": "${:,.0f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

    # -------------------------------------------------------------
    # TAB 3: MENU ENGINEERING
    # -------------------------------------------------------------
    with tab_menu:
        st.markdown("### Menu Engineering Matrix & Contribution Analysis")

        items_df = compute_menu_item_performance(filtered_df)

        rank_metric = st.radio(
            "Rank Menu Items By:",
            options=["net_revenue", "estimated_gross_profit", "units_sold"],
            format_func=lambda x: {
                "net_revenue": "Net Sales Revenue ($)",
                "estimated_gross_profit": "Estimated Gross Profit ($)",
                "units_sold": "Units Sold (Quantity)",
            }[x],
            horizontal=True,
        )
        st.plotly_chart(plot_menu_item_ranking(items_df, metric=rank_metric, top_n=10), use_container_width=True)

        st.plotly_chart(plot_menu_scatter(items_df), use_container_width=True)
        st.caption(
            "**Matrix Quadrants:** "
            "• **Top-Right (Stars):** High Volume & High Profit (Protect quality). "
            "• **Top-Left (Puzzles):** Low Volume & High Profit (Promote visibility). "
            "• **Bottom-Right (Plowhorses):** High Volume & Low Margin (Review portion/cost). "
            "• **Bottom-Left (Dogs):** Low Volume & Low Margin (Consider re-engineering)."
        )

        st.markdown("---")
        st.markdown("#### ABC Contribution Classification (Pareto Curve)")
        abc_df = compute_abc_analysis(filtered_df)
        st.dataframe(
            abc_df[[
                "item_name", "category", "abc_class", "units_sold", "net_revenue",
                "estimated_gross_profit", "gross_margin", "cumulative_share"
            ]].style.format({
                "units_sold": "{:,}",
                "net_revenue": "${:,.2f}",
                "estimated_gross_profit": "${:,.2f}",
                "gross_margin": "{:.1%}",
                "cumulative_share": "{:.1%}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        safe_dataframe_download(
            abc_df,
            filename="menu_abc_analysis.csv",
            label="Download ABC Menu Classification (CSV)",
            key="dl_abc_csv",
        )

    # -------------------------------------------------------------
    # TAB 4: OPERATIONAL INSIGHTS
    # -------------------------------------------------------------
    with tab_ops:
        st.markdown("### Transparent Operational Opportunities")
        insights = generate_operational_insights(filtered_df)

        col_op1, col_op2 = st.columns(2)
        with col_op1:
            st.markdown("#### Peak Demand & Outlet Capacity")
            st.markdown(f"- **Rush Window:** {insights.get('rush_hour', 'N/A')}")
            st.markdown(f"- **Location Focus:** {insights.get('outlet_dominance', 'N/A')}")
            st.markdown(f"- **Weekend Shift:** {insights.get('weekend_pattern', 'N/A')}")
        with col_op2:
            st.markdown("#### Product Margins & Promotion Strategy")
            st.markdown(f"- **Top Contributors:** {insights.get('menu_heroes', 'N/A')}")
            st.markdown(
                "- **Discount Discipline:** Recorded discounts totaled "
                f"**{format_currency(kpis['total_discounts'])}** across the selected period "
                f"({kpis['total_discounts'] / (kpis['total_revenue'] + kpis['total_discounts']) * 100:.1f}% effective discount rate)."
            )

        render_callout(
            text=(
                "All operational findings are derived dynamically through rule-based mathematical aggregation of "
                "the filtered dataset. They highlight descriptive efficiency opportunities without assuming causal mechanisms."
            ),
            title="Analytical Transparency",
            project_key=PROJECT_KEY,
        )

    # -------------------------------------------------------------
    # TAB 5: SALES FORECAST BASELINE
    # -------------------------------------------------------------
    with tab_forecast:
        st.markdown("### Chronological Sales Trend & 14-Day Baseline Forecast")

        forecast_df, error_msg = compute_sales_forecast(filtered_df, forecast_days=14, rolling_window=7)
        if error_msg:
            st.warning(error_msg)
        else:
            st.plotly_chart(plot_forecast_vs_actual(forecast_df), use_container_width=True)
            st.markdown(
                """
                - **Methodology:** 7-day chronological moving average coupled with a linear drift projection.
                - **Evaluation Honesty:** Evaluated purely chronologically on trailing daily sales (no time-series shuffling).
                - **Operational Purpose:** Provides a practical baseline for kitchen purchasing and shift staffing.
                """
            )

    # -------------------------------------------------------------
    # TAB 6: DATA EXPLORER & SCHEMA
    # -------------------------------------------------------------
    with tab_data:
        st.markdown("### Filtered Transaction Explorer & Upload Specifications")

        st.dataframe(
            filtered_df.head(200)[[
                "order_id", "line_item_id", "date", "outlet", "ordering_channel",
                "item_name", "category", "quantity", "unit_price", "discount_rate",
                "net_revenue", "estimated_gross_profit"
            ]].style.format({
                "quantity": "{:,}",
                "unit_price": "${:,.2f}",
                "discount_rate": "{:.0%}",
                "net_revenue": "${:,.2f}",
                "estimated_gross_profit": "${:,.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        safe_dataframe_download(
            filtered_df,
            filename="filtered_restaurant_transactions.csv",
            label=f"Download Filtered Dataset ({len(filtered_df):,} Rows)",
            key="dl_filtered_csv",
        )

        st.markdown("#### Custom Dataset Schema Requirements")
        st.markdown(
            "To upload your own restaurant transaction data via the sidebar, ensure your CSV includes the following columns:"
        )
        schema_table = [
            {"Column": c, "Required Type": "Numeric" if "price" in c or "cost" in c or "rev" in c or "profit" in c or "rate" in c or "quant" in c else "String / Date"}
            for c in REQUIRED_COLUMNS
        ]
        st.dataframe(pd.DataFrame(schema_table), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Restaurant Sales & Performance",
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_restaurant_analytics_page()
