"""Plotly visualization builders for customer segmentation and purchase intent."""

from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from shared.styles import get_plotly_template

PROJECT_KEY = "customer_segmentation"


def plot_conversion_by_visitor(visitor_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart comparing conversion rate and volume by visitor type."""
    tpl = get_plotly_template(PROJECT_KEY)
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=visitor_df["VisitorType"].str.replace("_", " "),
            x=visitor_df["conversion_rate"] * 100,
            orientation="h",
            marker=dict(color="#1E40AF"),
            text=[f"{cr * 100:.1f}% ({tot:,} sessions)" for cr, tot in zip(visitor_df["conversion_rate"], visitor_df["total_sessions"])],
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>Conversion Rate: %{x:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        tpl["layout"],
        title="Purchase Conversion Rate by Visitor Type",
        xaxis_title="Conversion Rate (%)",
        yaxis_title="",
        height=260,
    )
    return fig


def plot_monthly_patterns(monthly_df: pd.DataFrame) -> go.Figure:
    """Dual-axis chart displaying monthly sessions and conversion rates."""
    tpl = get_plotly_template(PROJECT_KEY)
    fig = go.Figure()

    # Session volume bars
    fig.add_trace(
        go.Bar(
            x=monthly_df["Month"],
            y=monthly_df["sessions"],
            name="Session Volume",
            marker=dict(color="#93C5FD"),
            hovertemplate="Month: %{x}<br>Sessions: %{y:,}<extra></extra>",
            yaxis="y1",
        )
    )

    # Conversion rate line
    fig.add_trace(
        go.Scatter(
            x=monthly_df["Month"],
            y=monthly_df["conversion_rate"] * 100,
            name="Conversion Rate (%)",
            mode="lines+markers",
            line=dict(color="#1E3A8A", width=2.5),
            marker=dict(size=6, color="#1E3A8A"),
            hovertemplate="Month: %{x}<br>Conversion Rate: %{y:.1f}%<extra></extra>",
            yaxis="y2",
        )
    )

    fig.update_layout(
        tpl["layout"],
        title="Monthly Session Volume and Conversion Rate",
        xaxis=dict(title="Month", tickangle=-20),
        yaxis=dict(title="Total Sessions", showgrid=True, gridcolor="#F3F4F6"),
        yaxis2=dict(
            title="Conversion Rate (%)",
            overlaying="y",
            side="right",
            showgrid=False,
            range=[0, max(monthly_df["conversion_rate"] * 100) * 1.3],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=320,
    )
    return fig


def plot_pca_clusters(
    pca_sample: pd.DataFrame,
    centroids_pca: pd.DataFrame,
    explained_var: np.ndarray,
) -> go.Figure:
    """Scatter plot displaying session clusters projected onto first two Principal Components."""
    tpl = get_plotly_template(PROJECT_KEY)
    fig = go.Figure()

    palette = ["#2563EB", "#0D9488", "#D97706", "#7C3AED", "#DC2626", "#0891B2"]
    clusters = sorted(pca_sample["Cluster"].unique())

    for i, cl in enumerate(clusters):
        subset = pca_sample[pca_sample["Cluster"] == cl]
        color = palette[i % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=subset["PC1"],
                y=subset["PC2"],
                mode="markers",
                name=cl,
                marker=dict(size=4.5, color=color, opacity=0.55),
                hovertemplate=(
                    f"<b>{cl}</b><br>"
                    + "PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>"
                    + "Product Pages: %{customdata[0]}<br>"
                    + "Exit Rate: %{customdata[1]:.3f}<br>"
                    + "Purchased: %{customdata[2]}<extra></extra>"
                ),
                customdata=subset[["ProductRelated", "ExitRates", "Revenue"]].values,
            )
        )

    # Add centroids
    fig.add_trace(
        go.Scatter(
            x=centroids_pca["PC1"],
            y=centroids_pca["PC2"],
            mode="markers+text",
            name="Centroids",
            marker=dict(size=12, color="#111827", symbol="diamond-open", line=dict(width=2)),
            text=[f"C{i+1}" for i in range(len(centroids_pca))],
            textposition="top center",
            hoverinfo="skip",
        )
    )

    pc1_pct = explained_var[0] * 100
    pc2_pct = explained_var[1] * 100

    fig.update_layout(
        tpl["layout"],
        title="Behavioral Segments in PCA Space",
        xaxis_title=f"Principal Component 1 ({pc1_pct:.1f}% Variance)",
        yaxis_title=f"Principal Component 2 ({pc2_pct:.1f}% Variance)",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_cluster_profiles(profiles_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing normalized behavioral metrics across clusters."""
    tpl = get_plotly_template(PROJECT_KEY)
    fig = go.Figure()

    metrics = [
        ("ProductRelated", "Avg Product Pages", 1.0),
        ("BounceRates", "Bounce Rate (x100)", 100.0),
        ("ExitRates", "Exit Rate (x100)", 100.0),
        ("Purchase_Rate", "Conversion Rate (%)", 100.0),
    ]

    palette = ["#2563EB", "#0D9488", "#D97706", "#7C3AED", "#DC2626"]
    for idx, (col, label, scale) in enumerate(metrics):
        fig.add_trace(
            go.Bar(
                x=[f"Segment {c+1}" for c in profiles_df.index],
                y=profiles_df[col] * scale,
                name=label,
                marker=dict(color=palette[idx % len(palette)]),
                hovertemplate=f"%{{x}}<br>{label}: %{{y:.2f}}<extra></extra>",
            )
        )

    fig.update_layout(
        tpl["layout"],
        barmode="group",
        title="Behavioral Comparison Across Discovered Segments",
        xaxis_title="",
        yaxis_title="Metric Value",
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_roc_curve(results: Dict[str, Any]) -> go.Figure:
    """ROC curve comparison between Logistic Regression and Random Forest."""
    tpl = get_plotly_template(PROJECT_KEY)
    fig = go.Figure()

    # Diagonal reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="#9CA3AF", dash="dash"),
            name="Random Guessing (AUC = 0.50)",
            hoverinfo="skip",
        )
    )

    colors = {"Logistic Regression": "#475569", "Random Forest": "#1E40AF"}
    for name, res in results.items():
        fpr, tpr = res["roc_curve"]
        auc = res["metrics_at_50"]["roc_auc"]
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{name} (AUC = {auc:.3f})",
                line=dict(color=colors.get(name, "#2563EB"), width=2.2),
                hovertemplate=f"<b>{name}</b><br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>",
            )
        )

    fig.update_layout(
        tpl["layout"],
        title="Receiver Operating Characteristic (ROC) Comparison",
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Recall)",
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_confusion_matrix(cm: np.ndarray, threshold: float = 0.50) -> go.Figure:
    """Annotated confusion matrix visualization."""
    tpl = get_plotly_template(PROJECT_KEY)
    tn, fp, fn, tp = cm.ravel()
    z_matrix = [[tn, fp], [fn, tp]]
    text_matrix = [
        [f"True Negatives<br><b>{tn:,}</b>", f"False Positives<br><b>{fp:,}</b>"],
        [f"False Negatives<br><b>{fn:,}</b>", f"True Positives<br><b>{tp:,}</b>"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=["Predicted: No Purchase", "Predicted: Purchase"],
            y=["Actual: No Purchase", "Actual: Purchase"],
            text=text_matrix,
            texttemplate="%{text}",
            colorscale=[[0, "#EFF6FF"], [1, "#1E40AF"]],
            showscale=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        tpl["layout"],
        title=f"Confusion Matrix (Decision Threshold = {threshold:.2f})",
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
        height=280,
    )
    return fig


def plot_feature_importance(importance_series: pd.Series, top_n: int = 10) -> go.Figure:
    """Horizontal bar chart showing top predictive features."""
    tpl = get_plotly_template(PROJECT_KEY)
    top_feats = importance_series.head(top_n).sort_values(ascending=True)

    fig = go.Figure(
        go.Bar(
            y=[f.replace("num__", "").replace("cat__", "") for f in top_feats.index],
            x=top_feats.values,
            orientation="h",
            marker=dict(color="#1E40AF"),
            hovertemplate="Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        tpl["layout"],
        title=f"Top {top_n} Predictive Feature Importances (Random Forest)",
        xaxis_title="Mean Decrease in Impurity (Gini Importance)",
        yaxis_title="",
        height=320,
    )
    return fig
