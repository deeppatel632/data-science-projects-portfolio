"""Interactive Streamlit dashboard for Student Academic Performance Analysis and Grade Prediction."""

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
)

try:
    from projects.student_performance.analysis import (
        compute_academic_overview,
        compute_correlations,
        compute_group_comparisons,
        generate_academic_observations,
    )
    from projects.student_performance.data import load_student_data
    from projects.student_performance.modeling import (
        predict_single_student,
        train_regression_models,
    )
    from projects.student_performance.visualizations import (
        plot_actual_vs_predicted,
        plot_correlation_heatmap,
        plot_grade_distribution,
        plot_period_grades_trajectory,
        plot_regression_comparison,
        plot_studytime_vs_grades,
    )
except ImportError:
    from analysis import (
        compute_academic_overview,
        compute_correlations,
        compute_group_comparisons,
        generate_academic_observations,
    )
    from data import load_student_data
    from modeling import (
        predict_single_student,
        train_regression_models,
    )
    from visualizations import (
        plot_actual_vs_predicted,
        plot_correlation_heatmap,
        plot_grade_distribution,
        plot_period_grades_trajectory,
        plot_regression_comparison,
        plot_studytime_vs_grades,
    )

PROJECT_KEY = "student_performance"


@st.cache_data
def get_cached_student_data(subject_code: str):
    """Cache loaded dataset per subject."""
    return load_student_data(subject=subject_code)


@st.cache_data
def get_cached_regression_models(subject_code: str, include_previous_grades: bool):
    """Cache trained regression pipelines per subject and scenario."""
    df, _ = get_cached_student_data(subject_code)
    return train_regression_models(df, include_previous_grades=include_previous_grades)


def render_student_performance_page():
    """Render the complete student academic performance analysis application."""
    apply_custom_css(PROJECT_KEY)

    render_page_header(
        title="Student Academic Performance Analysis & Grade Prediction",
        subtitle=(
            "Interpretable secondary school achievement modeling across Mathematics and Portuguese language cohorts, "
            "evaluating early-intervention vs mid-term grade scenarios with strict non-causal governance."
        ),
        tag="PROJECT 2 · EDUCATIONAL DATA SCIENCE",
        project_key=PROJECT_KEY,
    )

    # Subject Selector Sidebar
    st.sidebar.markdown("### Cohort Configuration")
    subject_choice = st.sidebar.selectbox(
        "Select Subject Cohort:",
        options=["Mathematics (student-mat)", "Portuguese Language (student-por)"],
        index=0,
        help="The UCI dataset contains two distinct subject cohorts with identical features but different grading scales.",
    )
    subject_code = "mat" if "Mathematics" in subject_choice else "por"
    subject_label = "Mathematics" if subject_code == "mat" else "Portuguese Language"

    df, metadata = get_cached_student_data(subject_code)
    overview = compute_academic_overview(df)

    # Data Provenance Banner
    if metadata.get("is_synthetic"):
        st.warning(
            f"⚠️ Using Reproducible Synthetic Demonstration Data for {subject_label}: "
            "The official CSV file could not be read locally. All statistics and models reflect "
            "synthetic student records with matching statistical moments and correlations."
        )
    else:
        st.caption(
            f"✓ Data Source: **{metadata['source_name']} ({subject_label})** ({overview['total_students']:,} students) · "
            f"Citation: {metadata['citation'][:80]}... · License: {metadata['license']}"
        )

    # Dashboard Tabs
    tab_overview, tab_eda, tab_prediction, tab_ethics = st.tabs([
        "1. Academic Overview",
        "2. Exploratory Analysis",
        "3. Grade Prediction & Scenarios",
        "4. Interpretation & Responsible Use",
    ])

    # -------------------------------------------------------------
    # TAB 1: ACADEMIC OVERVIEW
    # -------------------------------------------------------------
    with tab_overview:
        st.markdown(f"### Academic Achievement Summary · {subject_label}")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_metric_card(
                title="Cohort Size",
                value=format_number(overview["total_students"]),
                sub=f"Students evaluated in {subject_label}",
            )
        with c2:
            render_metric_card(
                title="Mean Final Grade (G3)",
                value=f"{overview['mean_grade']:.2f} / 20",
                sub=f"Median: {overview['median_grade']:.0f} · Std: ±{overview['std_grade']:.2f}",
            )
        with c3:
            render_metric_card(
                title="Pass Rate (≥10/20)",
                value=format_percentage(overview["pass_rate"]),
                sub=f"{overview['passed_count']} of {overview['total_students']} students passed",
            )
        with c4:
            render_metric_card(
                title="Zero-Grade Rate",
                value=format_percentage(overview["zero_grade_pct"]),
                sub=f"{overview['zero_grade_count']} non-attending / dropout grades",
            )

        st.plotly_chart(plot_grade_distribution(df, pass_threshold=10), use_container_width=True)

        st.markdown("#### Educational Grading System Context")
        col_scale, col_absences = st.columns(2)
        with col_scale:
            st.markdown(
                """
                - **Grading Scale:** Portuguese secondary schools grade from **0 to 20**, where:
                  - **10–20:** Passing grades (10–13 Sufficient, 14–15 Good, 16–17 Very Good, 18–20 Excellent).
                  - **0–9:** Failing grades requiring remediation or course repetition.
                - **The Zero-Grade Anomaly:** Records with $G3 = 0$ generally reflect unexcused exam absences, course drops, or disciplinary non-attendance rather than purely cognitive mastery scores.
                """
            )
        with col_absences:
            st.markdown(
                f"""
                - **Absence Distribution:** Cohort mean absences is **{overview['mean_absences']:.1f} days** (median: {overview['median_absences']:.0f} days).
                - **Data Completeness:** Zero missing values across all 33 demographic, lifestyle, and academic features.
                - **School Representation:** Gabriel Pereira (GP) accounts for **{(df['school'] == 'GP').mean() * 100:.1f}%** of records; Mousinho da Silveira (MS) accounts for remainder.
                """
            )

    # -------------------------------------------------------------
    # TAB 2: EXPLORATORY ANALYSIS
    # -------------------------------------------------------------
    with tab_eda:
        st.markdown("### Study Habits, Trajectory & Demographic Associations")
        obs = generate_academic_observations(df, subject=subject_label)

        col_traj, col_study = st.columns(2)
        with col_traj:
            st.plotly_chart(plot_period_grades_trajectory(df), use_container_width=True)
            st.caption(f"**Progression Finding:** {obs['prior_grades_signal']}")
        with col_study:
            st.plotly_chart(plot_studytime_vs_grades(df), use_container_width=True)
            st.caption(f"**Study Habit Finding:** {obs['study_time_signal']}")

        st.markdown("---")
        st.markdown("#### Demographic & Behavioral Factor Comparisons")
        factor_choice = st.selectbox(
            "Select Factor to Compare Across Cohort:",
            options=[
                ("failures", "Previous Academic Failures"),
                ("higher", "Desire for Higher Education (Yes/No)"),
                ("internet", "Home Internet Access (Yes/No)"),
                ("address", "Home Address (Urban vs Rural)"),
                ("sex", "Student Gender (F vs M)"),
                ("schoolsup", "Extra Educational Support (Yes/No)"),
            ],
            format_func=lambda x: x[1],
            index=0,
        )
        col_group, col_corr = st.columns([1, 1.2])
        with col_group:
            group_col = factor_choice[0]
            grouped_df = compute_group_comparisons(df, group_col)
            st.dataframe(
                grouped_df.style.format({
                    "student_count": "{:,}",
                    "share_of_cohort": "{:.1%}",
                    "mean_g3": "{:.2f}",
                    "median_g3": "{:.1f}",
                    "mean_absences": "{:.1f}",
                    "pass_rate": "{:.1%}",
                }),
                use_container_width=True,
                hide_index=True,
            )
            if group_col == "failures":
                st.caption(f"**Academic History:** {obs['failures_signal']}")
        with col_corr:
            corr_df = compute_correlations(df)
            st.plotly_chart(plot_correlation_heatmap(corr_df), use_container_width=True)

    # -------------------------------------------------------------
    # TAB 3: GRADE PREDICTION & SCENARIOS
    # -------------------------------------------------------------
    with tab_prediction:
        st.markdown("### Academic Performance Regression & Scenario Modeling")

        render_callout(
            text=(
                "Educational research distinguishes between <strong>Early Warning (Baseline)</strong> "
                "— identifying students at matriculation who may need support — versus <strong>Mid-Term Trajectory</strong> "
                "— refining forecasts after Period 1 (G1) and Period 2 (G2) exams are scored. "
                "Models must be evaluated separately under both regimes."
            ),
            title="Modeling Honesty Disclosure",
            project_key=PROJECT_KEY,
        )

        models_early = get_cached_regression_models(subject_code, include_previous_grades=False)
        models_prev = get_cached_regression_models(subject_code, include_previous_grades=True)

        st.plotly_chart(
            plot_regression_comparison(models_early["results"], models_prev["results"]),
            use_container_width=True,
        )

        # Performance table
        st.markdown("#### Holdout Test Set Evaluation (80/20 Train/Test Split)")
        table_rows = []
        for m_name in models_early["results"].keys():
            e_res = models_early["results"][m_name]
            p_res = models_prev["results"][m_name]
            table_rows.append({
                "Algorithm": m_name,
                "Early MAE": f"{e_res['mae']:.2f}",
                "Early RMSE": f"{e_res['rmse']:.2f}",
                "Early R²": f"{e_res['r2']:.2f}",
                "With G1/G2 MAE": f"{p_res['mae']:.2f}",
                "With G1/G2 RMSE": f"{p_res['rmse']:.2f}",
                "With G1/G2 R²": f"{p_res['r2']:.2f}",
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        col_actual, col_form = st.columns([1.2, 1])
        with col_actual:
            st.plotly_chart(
                plot_actual_vs_predicted(
                    models_prev["y_test"],
                    models_prev["results"]["Random Forest Regressor"]["y_pred"],
                    "Random Forest · With Previous Grades",
                ),
                use_container_width=True,
            )

        with col_form:
            st.markdown("#### Interactive Student Simulator")
            st.caption("Estimate illustrative final grade for a hypothetical student:")

            sim_scenario = st.radio(
                "Prediction Scenario:",
                options=["Early-Support (No previous grades)", "Mid-Year (With G1 & G2)"],
                horizontal=True,
            )
            sim_use_grades = "Mid-Year" in sim_scenario

            with st.form("student_simulator_form"):
                f_study = st.select_slider(
                    "Weekly Study Hours:",
                    options=[1, 2, 3, 4],
                    value=2,
                    format_func=lambda x: {1: "< 2 hrs", 2: "2–5 hrs", 3: "5–10 hrs", 4: "> 10 hrs"}[x],
                )
                f_failures = st.slider("Past Class Failures:", 0, 3, 0)
                f_absences = st.slider("School Absences (days):", 0, 40, 4)
                f_medu = st.slider("Mother's Education Level (0-4):", 0, 4, 2)
                f_higher = st.selectbox("Plans to Pursue Higher Education?", ["yes", "no"], index=0)
                f_internet = st.selectbox("Has Home Internet Access?", ["yes", "no"], index=0)

                f_g1, f_g2 = 11, 11
                if sim_use_grades:
                    c_g1, c_g2 = st.columns(2)
                    with c_g1:
                        f_g1 = st.slider("Period 1 Grade (G1):", 0, 20, 11)
                    with c_g2:
                        f_g2 = st.slider("Period 2 Grade (G2):", 0, 20, 11)

                submitted = st.form_submit_button("Compute Estimated Final Grade (G3)")

            if submitted:
                sim_student = {
                    "school": "GP", "sex": "F", "age": 16, "address": "U", "famsize": "GT3", "Pstatus": "T",
                    "Medu": f_medu, "Fedu": f_medu, "Mjob": "other", "Fjob": "other", "reason": "course",
                    "guardian": "mother", "traveltime": 1, "studytime": f_study, "failures": f_failures,
                    "schoolsup": "no", "famsup": "yes", "paid": "no", "activities": "yes", "nursery": "yes",
                    "higher": f_higher, "internet": f_internet, "romantic": "no", "famrel": 4, "freetime": 3,
                    "goout": 3, "Dalc": 1, "Walc": 2, "health": 3, "absences": f_absences,
                    "G1": f_g1, "G2": f_g2,
                }
                active_pipeline = (
                    models_prev["results"]["Random Forest Regressor"]["pipeline"]
                    if sim_use_grades
                    else models_early["results"]["Random Forest Regressor"]["pipeline"]
                )
                predicted_g3 = predict_single_student(active_pipeline, sim_student)

                st.success(f"**Estimated Final Grade: {predicted_g3:.1f} / 20**")
                pass_label = "Passing (≥ 10)" if predicted_g3 >= 10 else "At-Risk (< 10)"
                st.caption(
                    f"Status Estimate: **{pass_label}**. Note: This statistical estimate has an expected error margin of "
                    f"±{models_prev['results']['Random Forest Regressor']['mae']:.1f} grades with G1/G2 (or "
                    f"±{models_early['results']['Random Forest Regressor']['mae']:.1f} in early scenario)."
                )

    # -------------------------------------------------------------
    # TAB 4: INTERPRETATION & RESPONSIBLE USE
    # -------------------------------------------------------------
    with tab_ethics:
        st.markdown("### Methodological Interpretation & Governance")

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.markdown("#### Analytical Limitations")
            st.markdown(
                """
                1. **Localized Sample:** Data was collected from two specific secondary schools in Portugal during the 2005–2006 academic year. Educational norms, grading standards, and socioeconomic conditions may not generalize to other institutions or eras.
                2. **Small Cohort Size:** With only 395 Mathematics and 649 Portuguese students, high-order subgroup interactions cannot be estimated with statistical certainty.
                3. **Self-Reported Covariates:** Factors like alcohol consumption (`Dalc`, `Walc`) and weekly study time are self-reported survey responses subject to social desirability bias.
                """
            )
        with col_e2:
            st.markdown("#### Responsible AI & Educational Ethics")
            st.markdown(
                """
                1. **No High-Stakes Automation:** Machine learning predictions must **never** serve as the sole or primary basis for grading, disciplinary intervention, academic tracking, or university admissions.
                2. **Early Support, Not Tracking:** Early warning models should trigger constructive academic coaching and tutoring opportunities, never punitive or stigmatizing labels.
                3. **Correlation vs. Causation:** Observing that internet access correlates positively with grades does not prove that providing a modem will automatically elevate an underperforming student's test scores.
                """
            )

        render_callout(
            text=(
                "Student performance is multidimensional and influenced by health, pedagogical quality, family "
                "stress, and personal motivation. Algorithmic outputs are statistical summaries of historical patterns, "
                "not deterministic destiny."
            ),
            title="Non-Deterministic Principle",
            project_key=PROJECT_KEY,
        )


if __name__ == "__main__":
    st.set_page_config(
        page_title="Student Academic Performance",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_student_performance_page()
