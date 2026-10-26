"""Interactive Streamlit application for Customer Behavioral Segmentation and Purchase Intent Prediction."""

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
    format_number,
    format_percentage,
    safe_dataframe_download,
)

try:
    from projects.customer_segmentation.analysis import (
        compute_dataset_summary,
        compute_monthly_patterns,
        compute_visitor_conversion,
        compute_weekend_patterns,
        generate_eda_observations,
    )
    from projects.customer_segmentation.data import load_shoppers_data
    from projects.customer_segmentation.modeling import (
        perform_kmeans_clustering,
        predict_with_threshold,
        train_classification_models,
    )
    from projects.customer_segmentation.visualizations import (
        plot_cluster_profiles,
        plot_confusion_matrix,
        plot_conversion_by_visitor,
        plot_feature_importance,
        plot_monthly_patterns,
        plot_pca_clusters,
        plot_roc_curve,
    )
except ImportError:
    from analysis import (
        compute_dataset_summary,
        compute_monthly_patterns,
        compute_visitor_conversion,
        compute_weekend_patterns,
        generate_eda_observations,
    )
    from data import load_shoppers_data
    from modeling import (
        perform_kmeans_clustering,
        predict_with_threshold,
        train_classification_models,
    )
    from visualizations import (
        plot_cluster_profiles,
        plot_confusion_matrix,
        plot_conversion_by_visitor,
        plot_feature_importance,
        plot_monthly_patterns,
        plot_pca_clusters,
        plot_roc_curve,
    )

PROJECT_KEY = "customer_segmentation"


@st.cache_data
def get_cached_shoppers_data():
    """Cache loaded dataset to avoid repeated I/O."""
    return load_shoppers_data()


@st.cache_data
def get_cached_models(include_page_values: bool):
    """Cache trained classification models."""
    df, _ = get_cached_shoppers_data()
    return train_classification_models(df, include_page_values=include_page_values)


def render_customer_segmentation_page():
    """Render the complete customer segmentation and intent dashboard."""
    apply_custom_css(PROJECT_KEY)

    render_page_header(
        title="Customer Behavioral Segmentation & Purchase Intent",
        subtitle=(
            "Session-level exploratory analysis, unsupervised behavioral clustering, "
            "and leakage-free purchase intention classification on e-commerce sessions."
        ),
        tag="PROJECT 1 · E-COMMERCE ANALYTICS",
        project_key=PROJECT_KEY,
    )

    df, metadata = get_cached_shoppers_data()
    summary = compute_dataset_summary(df)

    # Data Provenance Banner
    if metadata.get("is_synthetic"):
        st.warning(
            "⚠️ Using Reproducible Synthetic Demonstration Dataset: "
            "The official UCI dataset could not be read locally. All statistics and models use "
            "synthetically generated data with matching statistical distributions."
        )
    else:
        st.caption(
            f"✓ Data Source: **{metadata['source_name']}** ({summary['total_sessions']:,} sessions) · "
            f"License: {metadata['license']}"
        )

    # Dashboard Tabs
    tab_overview, tab_eda, tab_clustering, tab_modeling, tab_findings = st.tabs([
        "1. Overview",
        "2. Exploratory Analysis",
        "3. Behavioral Segmentation",
        "4. Purchase Intent Prediction",
        "5. Findings & Limitations",
    ])

    # -------------------------------------------------------------
    # TAB 1: OVERVIEW
    # -------------------------------------------------------------
    with tab_overview:
        st.markdown("### Executive Summary & Session Metrics")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_metric_card(
                title="Total Sessions",
                value=format_number(summary["total_sessions"]),
                sub="1-year observation window",
            )
        with c2:
            render_metric_card(
                title="Purchase Conversion Rate",
                value=format_percentage(summary["conversion_rate"]),
                sub=f"{format_number(summary['purchase_count'])} purchasing sessions",
            )
        with c3:
            render_metric_card(
                title="Avg Product Views",
                value=f"{summary['avg_product_pages']:.1f} pages",
                sub=f"{summary['avg_product_duration_min']:.1f} min catalog duration",
            )
        with c4:
            render_metric_card(
                title="Avg Exit Rate",
                value=format_percentage(summary["avg_exit_rate"]),
                sub=f"Avg bounce: {format_percentage(summary['avg_bounce_rate'])}",
            )

        st.markdown("#### Dataset Scope & Quality Characteristics")
        col_scope, col_quality = st.columns(2)
        with col_scope:
            st.markdown(
                """
                - **Session-Level Observation:** Each record represents an individual browsing session rather than an identified individual.
                - **Class Imbalance:** Purchases represent approximately **15.5%** of sessions, creating a standard imbalanced classification challenge.
                - **Multi-Source Traffic:** Covers 13 distinct traffic acquisition channels across 9 geographic regions.
                """
            )
        with col_quality:
            st.markdown(
                """
                - **Missing Values:** Zero missing values across all 18 standard features in the validated dataset.
                - **Temporality:** Data spans 10 calendar months (January and April are excluded from the source dataset).
                - **Target Variable:** `Revenue` is a binary boolean indicating if the session concluded with a completed transaction.
                """
            )

        render_callout(
            text=(
                "Because sessions are anonymized and lack persistent customer keys, this dashboard models "
                "<strong>immediate session behavior</strong>, not long-term customer personalities or lifetime value."
            ),
            title="Scope Boundary",
            project_key=PROJECT_KEY,
        )

    # -------------------------------------------------------------
    # TAB 2: EXPLORATORY DATA ANALYSIS
    # -------------------------------------------------------------
    with tab_eda:
        st.markdown("### Browsing Dynamics & Conversion Patterns")
        eda_obs = generate_eda_observations(df)

        col_vis, col_month = st.columns([1, 1.2])
        with col_vis:
            visitor_df = compute_visitor_conversion(df)
            st.plotly_chart(plot_conversion_by_visitor(visitor_df), use_container_width=True)
            st.caption(f"**Observed Pattern:** {eda_obs['visitor_comparison']}")

        with col_month:
            monthly_df = compute_monthly_patterns(df)
            st.plotly_chart(plot_monthly_patterns(monthly_df), use_container_width=True)
            st.caption(f"**Seasonality Insight:** {eda_obs['seasonality']}")

        st.markdown("---")
        col_weekend, col_bounce = st.columns(2)
        with col_weekend:
            weekend_df = compute_weekend_patterns(df)
            st.markdown("#### Weekday vs. Weekend Dynamics")
            st.dataframe(
                weekend_df[["day_type", "sessions", "purchases", "conversion_rate", "avg_product_pages"]].rename(
                    columns={
                        "day_type": "Period",
                        "sessions": "Sessions",
                        "purchases": "Purchases",
                        "conversion_rate": "Conversion Rate",
                        "avg_product_pages": "Avg Product Pages",
                    }
                ).style.format({
                    "Sessions": "{:,}",
                    "Purchases": "{:,}",
                    "Conversion Rate": "{:.2%}",
                    "Avg Product Pages": "{:.1f}",
                }),
                use_container_width=True,
                hide_index=True,
            )
        with col_bounce:
            st.markdown("#### Engagement & Abandonment Metrics")
            st.markdown(
                f"""
                - **Zero-Bounce Sessions:** Make up **{(df['BounceRates'] == 0).mean() * 100:.1f}%** of all visits.
                - **Observed Conversion Impact:** {eda_obs['bounce_impact']}
                - **Duration Relationship:** Median duration on product pages is **{df['ProductRelated_Duration'].median():.0f} seconds** for non-purchasers versus **{df[df['Revenue']]['ProductRelated_Duration'].median():.0f} seconds** for purchasers.
                """
            )

    # -------------------------------------------------------------
    # TAB 3: BEHAVIORAL SEGMENTATION
    # -------------------------------------------------------------
    with tab_clustering:
        st.markdown("### Unsupervised K-Means Session Segmentation")
        st.markdown(
            "Sessions are clustered strictly using 8 numerical browsing features. "
            "**The purchase target is excluded** to prevent post-hoc bias."
        )

        col_ctrl1, col_ctrl2 = st.columns([1, 2])
        with col_ctrl1:
            k_val = st.slider("Select Number of Clusters (k)", min_value=2, max_value=6, value=4, step=1)
            st.caption("Standardized with StandardScaler; fixed random state = 42.")

        cluster_results = perform_kmeans_clustering(df, n_clusters=k_val, random_state=42)
        profiles = cluster_results["profiles"]
        pca_sample = cluster_results["pca_sample"]
        centroids_pca = cluster_results["centroids_pca"]
        explained_var = cluster_results["explained_variance"]

        with col_ctrl2:
            st.markdown("**Discovered Cluster Profiles:**")
            for c_id, desc in cluster_results["segment_descriptions"].items():
                st.markdown(f"- **{desc}**")

        st.plotly_chart(
            plot_pca_clusters(pca_sample, centroids_pca, explained_var),
            use_container_width=True,
        )

        st.plotly_chart(
            plot_cluster_profiles(profiles),
            use_container_width=True,
        )

        st.markdown("#### Segment Assignment Export")
        export_df = cluster_results["df_assigned"][
            ["Cluster", "Segment_Description", "Administrative", "ProductRelated", "BounceRates", "ExitRates", "Revenue"]
        ]
        safe_dataframe_download(
            export_df,
            filename=f"session_clusters_k{k_val}.csv",
            label=f"Download {len(export_df):,} Clustered Sessions (CSV)",
            key="download_clusters_csv",
        )

    # -------------------------------------------------------------
    # TAB 4: PURCHASE INTENT PREDICTION
    # -------------------------------------------------------------
    with tab_modeling:
        st.markdown("### Purchase-Intention Classification")

        # Educational target leakage selector
        st.markdown("#### Feature Set & Leakage Prevention")
        include_pv = st.checkbox(
            "Demonstrate Target Leakage by including 'PageValues'",
            value=False,
            help=(
                "PageValues measures the average value of a web page visited before a transaction. "
                "In practice, its calculation often relies on post-transaction event logging, "
                "which introduces target leakage and artificially inflates ROC-AUC beyond realistic capability."
            ),
        )

        if include_pv:
            st.warning(
                "⚠️ Target Leakage Active: 'PageValues' is included. Notice the inflated test metrics. "
                "In production, page value is rarely known for new unpurchased sessions."
            )
        else:
            st.info(
                "✓ Honest Modeling: 'PageValues' is excluded from features to prevent target leakage."
            )

        model_artifacts = get_cached_models(include_page_values=include_pv)
        results = model_artifacts["results"]
        y_test = model_artifacts["y_test"]

        # Comparison metrics table
        st.markdown("#### Held-Out Evaluation (80/20 Stratified Split)")
        comp_rows = []
        for m_name, m_res in results.items():
            metrics = m_res["metrics_at_50"]
            comp_rows.append({
                "Model": m_name,
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1 Score": metrics["f1"],
                "ROC-AUC": metrics["roc_auc"],
            })
        st.dataframe(
            pd.DataFrame(comp_rows).style.format({
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1 Score": "{:.3f}",
                "ROC-AUC": "{:.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        col_roc, col_feat = st.columns(2)
        with col_roc:
            st.plotly_chart(plot_roc_curve(results), use_container_width=True)
        with col_feat:
            st.plotly_chart(
                plot_feature_importance(model_artifacts["feature_importances"], top_n=8),
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("#### Interactive Decision Threshold Simulator")
        st.markdown(
            "In e-commerce conversion prediction, false negatives mean missed sales engagement, "
            "while false positives waste outreach coupons or sales agent time."
        )

        selected_model_name = st.radio(
            "Select Model for Threshold Tuning:",
            options=["Random Forest", "Logistic Regression"],
            horizontal=True,
        )
        threshold_val = st.slider(
            "Classification Decision Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.45,
            step=0.05,
            help="Probabilities greater than or equal to this threshold are classified as 'Purchase'.",
        )

        active_prob = results[selected_model_name]["y_prob"]
        thresh_metrics = predict_with_threshold(y_test, active_prob, threshold=threshold_val)

        cm_col, met_col = st.columns([1.2, 1])
        with cm_col:
            st.plotly_chart(
                plot_confusion_matrix(thresh_metrics["cm"], threshold=threshold_val),
                use_container_width=True,
            )
        with met_col:
            st.markdown(f"**Performance at Threshold = {threshold_val:.2f}:**")
            render_metric_card("Precision (PPV)", format_percentage(thresh_metrics["precision"]))
            render_metric_card("Recall (Sensitivity)", format_percentage(thresh_metrics["recall"]))
            render_metric_card("F1 Harmonic Score", f"{thresh_metrics['f1']:.3f}")
            st.caption(
                f"True Positives: {thresh_metrics['tp']:,} · False Positives: {thresh_metrics['fp']:,} · "
                f"False Negatives: {thresh_metrics['fn']:,}"
            )

    # -------------------------------------------------------------
    # TAB 5: FINDINGS & LIMITATIONS
    # -------------------------------------------------------------
    with tab_findings:
        st.markdown("### Practical Findings & Modeling Governance")

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("#### Key Analytical Findings")
            st.markdown(
                """
                1. **Exit Rate is the Strongest Negative Signal:** High exit rate in early product pages strongly discriminates non-purchasers before total duration accumulates.
                2. **New Visitors Display Higher Conversion Propensity:** Counter-intuitively, first-time visitors exhibited higher conversion rates than returning visitors, likely reflecting targeted campaign arrivals with pre-formed purchase intent.
                3. **Engagement Asymmetry:** Clusters characterized by high catalog exploration convert at 3–4x the baseline rate, whereas high-bounce sessions convert under 2%.
                """
            )
        with col_f2:
            st.markdown("#### Methodological Limitations")
            st.markdown(
                """
                1. **Session Granularity:** Identifiers are not linked across devices. A single human may browse on mobile, abandon, and purchase on desktop, appearing as two disconnected sessions.
                2. **Target Leakage Risks:** The `PageValues` feature in the UCI dataset reflects revenue credit assigned to visited pages post-transaction. Honest machine learning workflows must omit it.
                3. **Non-Causal Associations:** Correlation between product duration and purchase does not prove that forcing users to spend more time will increase conversion.
                """
            )

        render_callout(
            text=(
                "Models trained on this dataset should be deployed for <em>dynamic user experience optimization</em> "
                "(e.g., offering live chat assistance to high-intent disengaged sessions) rather than punitive access restrictions."
            ),
            title="Ethical Guidance",
            project_key=PROJECT_KEY,
        )


if __name__ == "__main__":
    st.set_page_config(
        page_title="Customer Segmentation & Purchase Intent",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_customer_segmentation_page()
