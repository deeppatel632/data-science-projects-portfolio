"""Plotly visualizations for student academic performance analysis."""

from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from shared.styles import get_plotly_template

PROJECT_KEY = "student_performance"


def plot_grade_distribution(df: pd.DataFrame, pass_threshold: int = 10) -> go.Figure:
    """Histogram of G3 final grade distribution with pass threshold marker."""
    tpl = get_plotly_template(PROJECT_KEY)
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=df["G3"],
            xbins=dict(start=-0.5, end=20.5, size=1.0),
            marker=dict(
                color=["#DC2626" if x < pass_threshold else "#0F766E" for x in range(21)],
                line=dict(color="#FFFFFF", width=1),
            ),
            hovertemplate="Grade: %{x}<br>Students: %{y:,}<extra></extra>",
        )
    )

    fig.add_vline(
        x=pass_threshold - 0.5,
        line_dash="dash",
        line_color="#475569",
        line_width=2,
        annotation_text=f"Pass Threshold ({pass_threshold}/20)",
        annotation_position="top right",
    )

    fig.update_layout(
        tpl["layout"],
        title="Final Grade (G3) Cohort Distribution (0–20 Scale)",
        xaxis=dict(title="Final Grade (G3)", dtick=2, range=[-1, 21]),
        yaxis=dict(title="Number of Students"),
        height=320,
    )
    return fig


def plot_period_grades_trajectory(df: pd.DataFrame) -> go.Figure:
    """Mean grade trajectory across Period 1 (G1), Period 2 (G2), and Final (G3)."""
    tpl = get_plotly_template(PROJECT_KEY)
    fig = go.Figure()

    # Compare students who passed G1 vs those who struggled in G1
    struggled_g1 = df[df["G1"] < 10]
    passed_g1 = df[df["G1"] >= 10]

    for subset, label, color in [
        (struggled_g1, f"Struggled in G1 (N={len(struggled_g1)})", "#DC2626"),
        (passed_g1, f"Passed in G1 (N={len(passed_g1)})", "#0F766E"),
    ]:
        means = [subset["G1"].mean(), subset["G2"].mean(), subset["G3"].mean()]
        fig.add_trace(
            go.Scatter(
                x=["Period 1 (G1)", "Period 2 (G2)", "Final (G3)"],
                y=means,
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=2.5),
                marker=dict(size=8, color=color),
                hovertemplate=f"<b>{label}</b><br>Exam: %{{x}}<br>Mean Grade: %{{y:.2f}}<extra></extra>",
            )
        )

    fig.update_layout(
        tpl["layout"],
        title="Grade Trajectory Over Academic Year by Initial Performance",
        yaxis=dict(title="Mean Grade (0-20)", range=[0, 20]),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_studytime_vs_grades(df: pd.DataFrame) -> go.Figure:
    """Boxplot displaying grade distributions across self-reported weekly study hours."""
    tpl = get_plotly_template(PROJECT_KEY)
    fig = go.Figure()

    labels = {
        1: "< 2 hours",
        2: "2 to 5 hours",
        3: "5 to 10 hours",
        4: "> 10 hours",
    }
    palette = ["#94A3B8", "#5EEAD4", "#14B8A6", "#0F766E"]

    for st_val in sorted(df["studytime"].unique()):
        subset = df[df["studytime"] == st_val]
        fig.add_trace(
            go.Box(
                y=subset["G3"],
                name=labels.get(st_val, str(st_val)),
                marker=dict(color=palette[(st_val - 1) % len(palette)]),
                boxmean=True,
                hovertemplate=f"Study: {labels.get(st_val, str(st_val))}<br>Grade: %{{y}}<extra></extra>",
            )
        )

    fig.update_layout(
        tpl["layout"],
        title="Final Grade (G3) Distribution by Weekly Study Hours",
        xaxis_title="Reported Weekly Study Time",
        yaxis_title="Final Grade (0–20)",
        height=330,
        showlegend=False,
    )
    return fig


def plot_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Heatmap of pairwise feature correlations."""
    tpl = get_plotly_template(PROJECT_KEY)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns,
            y=corr_df.index,
            colorscale=[[0, "#DC2626"], [0.5, "#F8FAFC"], [1, "#0F766E"]],
            zmin=-1.0,
            zmax=1.0,
            hoverongaps=False,
            hovertemplate="Feature 1: %{y}<br>Feature 2: %{x}<br>Correlation: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        tpl["layout"],
        title="Pairwise Pearson Correlation Matrix",
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange="reversed"),
        height=420,
    )
    return fig


def plot_regression_comparison(
    results_early: Dict[str, Any],
    results_prev: Dict[str, Any]
) -> go.Figure:
    """Grouped bar chart comparing RMSE across Early-Support vs Previous-Grades scenarios."""
    tpl = get_plotly_template(PROJECT_KEY)
    fig = go.Figure()

    model_names = list(results_early.keys())
    early_rmse = [results_early[m]["rmse"] for m in model_names]
    prev_rmse = [results_prev[m]["rmse"] for m in model_names]

    fig.add_trace(
        go.Bar(
            x=model_names,
            y=early_rmse,
            name="Early Support (G1 & G2 Excluded)",
            marker=dict(color="#64748B"),
            hovertemplate="Model: %{x}<br>Early RMSE: %{y:.2f} grades<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            x=model_names,
            y=prev_rmse,
            name="With Previous Grades (G1 & G2 Included)",
            marker=dict(color="#0F766E"),
            hovertemplate="Model: %{x}<br>With Grades RMSE: %{y:.2f} grades<extra></extra>",
        )
    )

    fig.update_layout(
        tpl["layout"],
        barmode="group",
        title="Model Error (RMSE in Grades) by Feature Scenario",
        yaxis_title="Root Mean Squared Error (Grades 0-20)",
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> go.Figure:
    """Scatter plot comparing Actual vs Predicted grades with ideal reference line."""
    tpl = get_plotly_template(PROJECT_KEY)
    fig = go.Figure()

    # Ideal line
    fig.add_trace(
        go.Scatter(
            x=[0, 20],
            y=[0, 20],
            mode="lines",
            line=dict(color="#9CA3AF", dash="dash"),
            name="Perfect Prediction (y = x)",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            marker=dict(color="#0F766E", size=7, opacity=0.7),
            name="Held-out Test Students",
            hovertemplate="Actual G3: %{x}<br>Predicted G3: %{y:.1f}<extra></extra>",
        )
    )

    fig.update_layout(
        tpl["layout"],
        title=f"Actual vs. Predicted Final Grade ({model_name})",
        xaxis=dict(title="Actual G3 Grade (0-20)", range=[-0.5, 20.5]),
        yaxis=dict(title="Predicted G3 Grade (0-20)", range=[-0.5, 20.5]),
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
