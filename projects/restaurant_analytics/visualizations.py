"""Plotly visualizations for restaurant sales, operations, and menu engineering."""

import pandas as pd
import plotly.graph_objects as go

from shared.styles import get_plotly_template

PROJECT_KEY = "restaurant_analytics"


def plot_sales_trend(daily_df: pd.DataFrame) -> go.Figure:
    """Chronological line & bar chart displaying daily revenue and order counts."""
    tpl = get_plotly_template(PROJECT_KEY)
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=daily_df["date"],
            y=daily_df["orders"],
            name="Daily Orders",
            marker=dict(color="#FED7AA"),
            yaxis="y2",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Orders: %{y:,}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=daily_df["date"],
            y=daily_df["revenue"],
            name="Net Revenue ($)",
            mode="lines",
            line=dict(color="#C2410C", width=2.2),
            yaxis="y1",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Revenue: $%{y:,.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        tpl["layout"],
        title="Chronological Daily Sales and Order Volume",
        xaxis=dict(title="Date", showgrid=True),
        yaxis=dict(title="Net Revenue ($)", showgrid=True),
        yaxis2=dict(
            title="Distinct Orders",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        height=330,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_outlet_breakdown(outlet_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart comparing net revenue across restaurant outlets."""
    tpl = get_plotly_template(PROJECT_KEY)
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=outlet_df["outlet"],
            x=outlet_df["revenue"],
            orientation="h",
            marker=dict(color="#C2410C"),
            text=[f"${r:,.0f} ({m:.1%} margin)" for r, m in zip(outlet_df["revenue"], outlet_df["margin"])],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Revenue: $%{x:,.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        tpl["layout"],
        title="Net Sales Revenue and Gross Margin by Outlet",
        xaxis_title="Net Revenue ($)",
        yaxis_title="",
        height=260,
    )
    return fig


def plot_channel_share(channel_df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing revenue across ordering channels."""
    tpl = get_plotly_template(PROJECT_KEY)
    fig = go.Figure()

    palette = ["#C2410C", "#EA580C", "#FB923C", "#FDBA74"]

    fig.add_trace(
        go.Bar(
            x=channel_df["ordering_channel"],
            y=channel_df["revenue"],
            marker=dict(color=palette[:len(channel_df)]),
            text=[f"${r:,.0f}<br>({s:.1%})" for r, s in zip(channel_df["revenue"], channel_df["share_of_sales"])],
            textposition="auto",
            hovertemplate="Channel: %{x}<br>Revenue: $%{y:,.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        tpl["layout"],
        title="Sales by Ordering Channel",
        xaxis_title="",
        yaxis_title="Revenue ($)",
        height=260,
    )
    return fig


def plot_hourly_distribution(hourly_df: pd.DataFrame) -> go.Figure:
    """Bar chart illustrating order volume by hour of day (0–23)."""
    tpl = get_plotly_template(PROJECT_KEY)
    fig = go.Figure()

    # Highlight lunch peak (11-14) and dinner peak (17-21)
    colors = []
    for h in hourly_df["hour"]:
        if 11 <= h <= 14:
            colors.append("#F97316")  # Lunch
        elif 17 <= h <= 21:
            colors.append("#C2410C")  # Dinner
        else:
            colors.append("#E2E8F0")

    fig.add_trace(
        go.Bar(
            x=[f"{h:02d}:00" for h in hourly_df["hour"]],
            y=hourly_df["orders"],
            marker=dict(color=colors),
            hovertemplate="Hour: %{x}<br>Orders: %{y:,}<extra></extra>",
        )
    )

    fig.update_layout(
        tpl["layout"],
        title="Hourly Order Distribution (Lunch: 11-14h · Dinner: 17-21h)",
        xaxis=dict(title="Hour of Day", tickangle=-45),
        yaxis=dict(title="Total Orders"),
        height=300,
    )
    return fig


def plot_menu_item_ranking(items_df: pd.DataFrame, metric: str = "net_revenue", top_n: int = 10) -> go.Figure:
    """Rank menu items by revenue, gross profit, or units sold."""
    tpl = get_plotly_template(PROJECT_KEY)
    top_items = items_df.sort_values(metric, ascending=True).tail(top_n)

    labels = {
        "net_revenue": ("Net Sales Revenue ($)", "${x:,.2f}"),
        "estimated_gross_profit": ("Estimated Gross Profit ($)", "${x:,.2f}"),
        "units_sold": ("Units Sold (Quantity)", "{x:,}"),
    }
    axis_title, fmt = labels.get(metric, ("Value", "{x}"))

    fig = go.Figure(
        go.Bar(
            y=top_items["item_name"],
            x=top_items[metric],
            orientation="h",
            marker=dict(color="#C2410C"),
            hovertemplate=f"Item: %{{y}}<br>{axis_title}: {fmt}<extra></extra>",
        )
    )

    fig.update_layout(
        tpl["layout"],
        title=f"Top {top_n} Menu Items by {axis_title}",
        xaxis_title=axis_title,
        yaxis_title="",
        height=320,
    )
    return fig


def plot_menu_scatter(items_df: pd.DataFrame) -> go.Figure:
    """Menu Engineering Matrix: Units Sold vs Estimated Gross Profit."""
    tpl = get_plotly_template(PROJECT_KEY)
    fig = go.Figure()

    cats = items_df["category"].unique()
    palette = ["#C2410C", "#2563EB", "#0F766E", "#D97706", "#7C3AED"]

    median_vol = items_df["units_sold"].median()
    median_profit = items_df["estimated_gross_profit"].median()

    for idx, cat in enumerate(cats):
        subset = items_df[items_df["category"] == cat]
        fig.add_trace(
            go.Scatter(
                x=subset["units_sold"],
                y=subset["estimated_gross_profit"],
                mode="markers+text",
                name=cat,
                text=subset["item_name"],
                textposition="top right",
                textfont=dict(size=10, color="#374151"),
                marker=dict(
                    size=subset["net_revenue"] / subset["net_revenue"].max() * 26 + 8,
                    color=palette[idx % len(palette)],
                    opacity=0.8,
                    line=dict(color="#FFFFFF", width=1),
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    + "Category: " + cat + "<br>"
                    + "Units Sold: %{x:,}<br>"
                    + "Gross Profit: $%{y:,.2f}<extra></extra>"
                ),
            )
        )

    # Reference crosshairs for Stars / Plowhorses / Puzzles / Dogs
    fig.add_vline(x=median_vol, line_dash="dot", line_color="#CBD5E1", line_width=1.5)
    fig.add_hline(y=median_profit, line_dash="dot", line_color="#CBD5E1", line_width=1.5)

    fig.update_layout(
        tpl["layout"],
        title="Menu Engineering Matrix: Volume vs. Estimated Gross Profit",
        xaxis_title="Units Sold (Popularity / Volume)",
        yaxis_title="Estimated Gross Profit ($ Contribution)",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_forecast_vs_actual(combined_df: pd.DataFrame) -> go.Figure:
    """Historical sales trend vs chronological rolling-average baseline projection."""
    tpl = get_plotly_template(PROJECT_KEY)
    fig = go.Figure()

    hist = combined_df[~combined_df["is_forecast"]]
    future = combined_df[combined_df["is_forecast"]]

    # Historical Actual
    fig.add_trace(
        go.Scatter(
            x=hist["date"],
            y=hist["revenue"],
            mode="lines",
            name="Observed Actual Sales",
            line=dict(color="#64748B", width=1.5),
            hovertemplate="Actual: $%{y:,.2f}<extra></extra>",
        )
    )

    # 7-day rolling average
    fig.add_trace(
        go.Scatter(
            x=hist["date"],
            y=hist["rolling_avg"],
            mode="lines",
            name="7-Day Rolling Trend Baseline",
            line=dict(color="#C2410C", width=2.5),
            hovertemplate="Trend Baseline: $%{y:,.2f}<extra></extra>",
        )
    )

    # Future projection
    if len(future) > 0:
        fig.add_trace(
            go.Scatter(
                x=future["date"],
                y=future["projected_revenue"],
                mode="lines+markers",
                name="Projected Baseline (Next 14 Days)",
                line=dict(color="#EA580C", width=2.2, dash="dash"),
                marker=dict(size=5, color="#EA580C"),
                hovertemplate="Forecast Date: %{x|%Y-%m-%d}<br>Projected: $%{y:,.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        tpl["layout"],
        title="Chronological Sales Trend & 14-Day Rolling Baseline Projection",
        xaxis_title="Date",
        yaxis_title="Daily Net Revenue ($)",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
