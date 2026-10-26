"""Styling tokens, custom CSS injection, and Plotly templates for the portfolio.

Adheres to restrained, editorial analytical design principles:
- Light, neutral background and dark readable typography.
- Single muted project accent colors without neon, glowing, or gradient noise.
- Crisp 1px borders, subtle 6px corner radius, and compact metric cards.
"""

from typing import Any, Dict, Optional

import plotly.graph_objects as go
import streamlit as st

PROJECT_THEMES = {
    "portfolio": {
        "primary": "#1E3A8A",      # Oxford Blue
        "accent": "#2563EB",       # Royal Blue
        "light_accent": "#EFF6FF", # Very soft blue
        "border": "#E2E8F0",
        "name": "Data Science Portfolio",
    },
    "customer_segmentation": {
        "primary": "#1E40AF",      # Deep Navy / Indigo
        "accent": "#3B82F6",       # Vibrant Slate Blue
        "light_accent": "#EFF6FF",
        "border": "#DBEAFE",
        "name": "Customer Behavioral Segmentation",
    },
    "student_performance": {
        "primary": "#0F766E",      # Scholarly Teal / Sage
        "accent": "#0D9488",       # Deep Teal
        "light_accent": "#F0FDFA",
        "border": "#CCFBF1",
        "name": "Student Academic Performance",
    },
    "restaurant_analytics": {
        "primary": "#C2410C",      # Warm Terracotta / Sienna
        "accent": "#EA580C",       # Amber Orange
        "light_accent": "#FFF7ED",
        "border": "#FFEDD5",
        "name": "Restaurant Sales & Performance",
    },
}


def apply_custom_css(project_key: str = "portfolio") -> None:
    """Inject subtle, editorial CSS tailored to the active project."""
    theme = PROJECT_THEMES.get(project_key, PROJECT_THEMES["portfolio"])
    primary_color = theme["primary"]
    border_color = theme["border"]
    light_accent = theme["light_accent"]

    css = f"""
    <style>
    /* Global Typography & Background adjustments */
    html, body, [class*="css"] {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #111827;
    }}

    /* Subtle main container padding */
    .block-container {{
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }}

    /* Sidebar editorial styling */
    section[data-testid="stSidebar"] {{
        background-color: #FFFFFF;
        border-right: 1px solid #E5E7EB;
    }}
    section[data-testid="stSidebar"] .block-container {{
        padding-top: 2rem;
    }}

    /* Editorial Metric Cards */
    .editorial-card {{
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 6px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    }}
    .editorial-card-title {{
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6B7280;
        margin-bottom: 0.35rem;
    }}
    .editorial-card-value {{
        font-size: 1.65rem;
        font-weight: 700;
        color: #111827;
        line-height: 1.2;
    }}
    .editorial-card-sub {{
        font-size: 0.76rem;
        color: #4B5563;
        margin-top: 0.35rem;
    }}

    /* Editorial Section Banners / Headers */
    .project-header {{
        border-bottom: 1px solid #E5E7EB;
        padding-bottom: 1.1rem;
        margin-bottom: 1.5rem;
    }}
    .project-header-tag {{
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {primary_color};
        background-color: {light_accent};
        border: 1px solid {border_color};
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }}
    .project-header-title {{
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.25;
    }}
    .project-header-desc {{
        font-size: 0.95rem;
        color: #4B5563;
        margin-top: 0.45rem;
        line-height: 1.5;
    }}

    /* Methodology and Note Callouts */
    .editorial-callout {{
        background-color: #F8FAFC;
        border-left: 3px solid {primary_color};
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        border-radius: 0 4px 4px 0;
        padding: 0.85rem 1.1rem;
        margin: 1rem 0;
        font-size: 0.86rem;
        color: #334155;
        line-height: 1.55;
    }}
    .editorial-callout strong {{
        color: #0F172A;
    }}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        border-bottom: 1px solid #E5E7EB;
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 8px 16px;
        font-weight: 500;
        font-size: 0.88rem;
        color: #4B5563;
        border-radius: 4px 4px 0 0;
    }}
    .stTabs [aria-selected="true"] {{
        color: {primary_color} !important;
        font-weight: 600 !important;
        border-bottom: 2px solid {primary_color} !important;
    }}

    /* Table headers */
    thead tr th:first-child {{display:none}}
    tbody th {{display:none}}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_page_header(
    title: str,
    subtitle: str,
    tag: str = "DATA SCIENCE PORTFOLIO",
    project_key: str = "portfolio",
) -> None:
    """Render a clean, restrained top header."""
    apply_custom_css(project_key)
    theme = PROJECT_THEMES.get(project_key, PROJECT_THEMES["portfolio"])
    theme["primary"]
    theme["border"]
    theme["light_accent"]

    header_html = f"""
    <div class="project-header">
        <span class="project-header-tag">{tag}</span>
        <h1 class="project-header-title">{title}</h1>
        <div class="project-header-desc">{subtitle}</div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def render_metric_card(
    title: str,
    value: str,
    sub: Optional[str] = None,
) -> None:
    """Render a compact, high-legibility metric card without neon or decorative fluff."""
    sub_html = f'<div class="editorial-card-sub">{sub}</div>' if sub else ""
    card_html = f"""
    <div class="editorial-card">
        <div class="editorial-card-title">{title}</div>
        <div class="editorial-card-value">{value}</div>
        {sub_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_callout(text: str, title: str = "Analytical Methodology", project_key: str = "portfolio") -> None:
    """Render an interpretable methodology / limitation notice."""
    callout_html = f"""
    <div class="editorial-callout">
        <strong>{title}:</strong> {text}
    </div>
    """
    st.markdown(callout_html, unsafe_allow_html=True)


def get_plotly_template(project_key: str = "portfolio") -> Dict[str, Any]:
    """Return a consistent, publication-ready Plotly layout dictionary."""
    theme = PROJECT_THEMES.get(project_key, PROJECT_THEMES["portfolio"])
    accent = theme["primary"]

    return {
        "layout": go.Layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(
                family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
                size=12,
                color="#1F2937",
            ),
            title=dict(
                font=dict(size=14, color="#111827", family="-apple-system, sans-serif"),
                x=0.01,
                xanchor="left",
            ),
            margin=dict(l=45, r=25, t=40, b=40),
            xaxis=dict(
                showgrid=True,
                gridcolor="#F3F4F6",
                gridwidth=1,
                linecolor="#E5E7EB",
                zeroline=False,
                tickfont=dict(size=11, color="#4B5563"),
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="#F3F4F6",
                gridwidth=1,
                linecolor="#E5E7EB",
                zeroline=False,
                tickfont=dict(size=11, color="#4B5563"),
            ),
            legend=dict(
                font=dict(size=11, color="#374151"),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#E5E7EB",
                borderwidth=1,
            ),
            hoverlabel=dict(
                bgcolor="#111827",
                font_size=11,
                font_color="#F9FAFB",
                font_family="-apple-system, sans-serif",
            ),
            colorway=[
                accent,
                "#475569",  # Slate
                "#059669",  # Emerald
                "#D97706",  # Amber
                "#7C3AED",  # Violet
                "#DC2626",  # Crimson
                "#0891B2",  # Cyan
            ],
        )
    }
