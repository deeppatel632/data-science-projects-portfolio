"""Central Data Science Portfolio Application.

Provides an editorial landing page and seamless integrated navigation across all three projects:
1. Customer Behavioral Segmentation and Purchase Intent Prediction
2. Student Academic Performance Analysis and Visualization
3. Restaurant Performance and Food Sales Analysis
"""

import sys
from pathlib import Path

import streamlit as st

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.customer_segmentation.app import render_customer_segmentation_page
from projects.restaurant_analytics.app import render_restaurant_analytics_page
from projects.student_performance.app import render_student_performance_page
from shared.styles import (
    apply_custom_css,
    render_metric_card,
    render_page_header,
)


def render_portfolio_home():
    """Render the central editorial portfolio homepage with project summaries and direct navigation."""
    apply_custom_css("portfolio")

    render_page_header(
        title="Data Science & Applied Analytics Portfolio",
        subtitle=(
            "A collection of three independent, intermediate-level data science applications demonstrating "
            "exploratory data analysis, unsupervised clustering, supervised classification & regression, "
            "financial modeling, and ethical machine learning governance."
        ),
        tag="PORTFOLIO OVERVIEW",
        project_key="portfolio",
    )

    # High-level Portfolio Scope Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Projects Included", "3 Projects", "Fully interactive dashboards")
    with c2:
        render_metric_card("Total Observations", "34,000+", "Real UCI + Calibrated POS data")
    with c3:
        render_metric_card("ML Workflows", "4 Algorithms", "K-Means, Logistic, RF, Ridge")
    with c4:
        render_metric_card("Test Coverage", "20 Tests Passing", "Pytest unit & integration suite")

    st.markdown("---")
    st.markdown("### Featured Portfolio Projects")

    # Project 1 Showcase
    with st.container():
        p1_col1, p1_col2 = st.columns([2.2, 1])
        with p1_col1:
            st.markdown(
                """
                #### 1. Customer Behavioral Segmentation & Purchase Intent
                **Domain:** E-Commerce User Behavior · **Data:** 12,330 Sessions (UCI Machine Learning Repository)

                - **Analytical Core:** Unsupervised K-Means clustering ($k=4$) on browsing metrics (page views, durations, bounce & exit rates) visualized in 2D PCA space, coupled with dual-classifier purchase intent modeling (Logistic Regression vs. Random Forest).
                - **Questions Answered:**
                  - *Which browsing behavior profiles signal high intent versus early abandonment?*
                  - *How does excluding leaky post-hoc features (`PageValues`) affect realistic model evaluation?*
                  - *How does tuning the decision threshold balance conversion recall against marketing spend?*
                - **Key Takeaway:** First-time visitors exhibit higher conversion propensity (24%) than returning visitors (14%), while exit rate on early product pages provides the strongest negative conversion signal.
                """
            )
        with p1_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Launch Customer Segmentation →", key="btn_nav_p1", use_container_width=True):
                st.session_state["nav_selection"] = "🛒 Customer Behavioral Segmentation"
                st.rerun()
            st.caption("Standalone launch: `streamlit run projects/customer_segmentation/app.py`")

    st.markdown("---")

    # Project 2 Showcase
    with st.container():
        p2_col1, p2_col2 = st.columns([2.2, 1])
        with p2_col1:
            st.markdown(
                """
                #### 2. Student Academic Performance Analysis & Grade Prediction
                **Domain:** Educational Data Science · **Data:** 1,044 Records across Math & Portuguese (UCI)

                - **Analytical Core:** Interpretable regression modeling ($G3$ grade on 0–20 scale) contrasting an *Early-Support Scenario* (excluding interim exams $G1/G2$) with a *Mid-Term Trajectory Scenario* (including $G1/G2$).
                - **Questions Answered:**
                  - *How much predictive power do background demographic covariates offer before exams take place?*
                  - *Why does mid-term academic performance dominate static student characteristics?*
                  - *How can predictive tools be governed responsibly to prevent algorithmic student tracking?*
                - **Key Takeaway:** Prior exam grades ($G1/G2$) explain over 75% of final variance ($r \approx 0.85$), showing that early warning systems must trigger supportive coaching rather than punitive tracking.
                """
            )
        with p2_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Launch Student Performance →", key="btn_nav_p2", use_container_width=True):
                st.session_state["nav_selection"] = "🎓 Student Academic Performance"
                st.rerun()
            st.caption("Standalone launch: `streamlit run projects/student_performance/app.py`")

    st.markdown("---")

    # Project 3 Showcase
    with st.container():
        p3_col1, p3_col2 = st.columns([2.2, 1])
        with p3_col1:
            st.markdown(
                """
                #### 3. Restaurant Performance & Food Sales Analysis
                **Domain:** Commercial Hospitality Intelligence · **Data:** 14,785 Line Items across 7,500 Orders

                - **Analytical Core:** Order-level financial accounting (Net Revenue, Estimated Gross Profit, Gross Margin, AOV), operational rush analysis, Menu Engineering Quadrant Matrix (Stars / Plowhorses / Puzzles / Dogs), ABC Pareto classification, and 14-day chronological sales forecasting.
                - **Questions Answered:**
                  - *Which menu items drive high sales volume but deliver limited contribution margins?*
                  - *When do operational rush hours occur, and how do channels differ across outlets?*
                  - *What is the projected 14-day chronological demand trend for kitchen procurement?*
                - **Key Takeaway:** Multi-line order aggregation prevents distorted order counts, revealing that evening rush periods (17:30–21:30) generate 65% of daily revenue with Friday/Saturday surges.
                """
            )
        with p3_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Launch Restaurant Analytics →", key="btn_nav_p3", use_container_width=True):
                st.session_state["nav_selection"] = "🍽️ Restaurant Sales & Performance"
                st.rerun()
            st.caption("Standalone launch: `streamlit run projects/restaurant_analytics/app.py`")

    st.markdown("---")
    st.markdown("### Engineering Architecture & Portability")
    col_arch1, col_arch2 = st.columns(2)
    with col_arch1:
        st.markdown(
            """
            - **Self-Contained Execution:** Every project functions both inside this central hub and as an independent standalone Streamlit application.
            - **No External Paid APIs or Cloud Dependencies:** Runs entirely locally on macOS Apple Silicon, Linux, or Windows with under 1 GB RAM usage.
            - **Target Leakage Governance:** Rigorous prevention of feature leakage in both classification and educational regression pipelines.
            """
        )
    with col_arch2:
        st.markdown(
            """
            - **Publication & Export Ready:** Contains an automated export utility (`scripts/publish_project.sh`) allowing any of the three projects to be exported into a standalone repository.
            - **Comprehensive Automated Testing:** 20 unit and integration tests covering data integrity, pipeline execution, metrics, and error boundaries.
            """
        )


def main():
    """Main routing controller for the central portfolio."""
    st.set_page_config(
        page_title="Data Science Projects Portfolio",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    nav_options = [
        "🏠 Portfolio Overview",
        "🛒 Customer Behavioral Segmentation",
        "🎓 Student Academic Performance",
        "🍽️ Restaurant Sales & Performance",
    ]

    # Initialize session state for navigation
    if "nav_selection" not in st.session_state:
        st.session_state["nav_selection"] = nav_options[0]

    # Sidebar Navigation
    st.sidebar.markdown("## 📊 Portfolio Directory")
    current_idx = nav_options.index(st.session_state["nav_selection"]) if st.session_state["nav_selection"] in nav_options else 0
    selected_page = st.sidebar.radio(
        "Select Project View:",
        options=nav_options,
        index=current_idx,
    )
    st.session_state["nav_selection"] = selected_page

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Repository Details:**
        - **Stack:** Python 3.11, pandas, scikit-learn, Streamlit, Plotly, pytest
        - **Monorepo Strategy:** Single central repository with standalone project decoupling
        """
    )

    # Route to selected page
    if selected_page == "🏠 Portfolio Overview":
        render_portfolio_home()
    elif selected_page == "🛒 Customer Behavioral Segmentation":
        render_customer_segmentation_page()
    elif selected_page == "🎓 Student Academic Performance":
        render_student_performance_page()
    elif selected_page == "🍽️ Restaurant Sales & Performance":
        render_restaurant_analytics_page()


if __name__ == "__main__":
    main()
