"""Exploratory and correlation analysis of student academic records."""

from typing import Any, Dict

import pandas as pd

NUMERIC_CORR_COLS = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health",
    "absences", "G1", "G2", "G3"
]


def compute_academic_overview(df: pd.DataFrame, pass_threshold: int = 10) -> Dict[str, Any]:
    """Compute summary academic indicators on the 0-20 Portuguese scale."""
    total_students = len(df)
    g3 = df["G3"]
    mean_grade = float(g3.mean())
    median_grade = float(g3.median())
    std_grade = float(g3.std())

    passed_count = int((g3 >= pass_threshold).sum())
    pass_rate = (passed_count / total_students) if total_students > 0 else 0.0

    zero_grade_count = int((g3 == 0).sum())
    zero_grade_pct = (zero_grade_count / total_students) if total_students > 0 else 0.0

    mean_absences = float(df["absences"].mean())
    median_absences = float(df["absences"].median())

    return {
        "total_students": total_students,
        "mean_grade": mean_grade,
        "median_grade": median_grade,
        "std_grade": std_grade,
        "pass_threshold": pass_threshold,
        "passed_count": passed_count,
        "pass_rate": pass_rate,
        "zero_grade_count": zero_grade_count,
        "zero_grade_pct": zero_grade_pct,
        "mean_absences": mean_absences,
        "median_absences": median_absences,
    }


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate pairwise Pearson correlation across all numerical academic features."""
    cols = [c for c in NUMERIC_CORR_COLS if c in df.columns]
    return df[cols].corr()


def compute_group_comparisons(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Compute average grade and pass rate grouped by a demographic or behavioral factor."""
    grouped = df.groupby(group_col, as_index=False).agg(
        student_count=("G3", "count"),
        mean_g3=("G3", "mean"),
        median_g3=("G3", "median"),
        mean_absences=("absences", "mean"),
        pass_rate=("G3", lambda s: (s >= 10).mean()),
    )
    grouped["share_of_cohort"] = grouped["student_count"] / len(df)
    return grouped.sort_values(by="mean_g3", ascending=False)


def generate_academic_observations(df: pd.DataFrame, subject: str = "Mathematics") -> Dict[str, str]:
    """Generate dynamic, factual observations reflecting the empirical data."""
    corr = compute_correlations(df)
    g1_g3_corr = corr.loc["G1", "G3"] if "G1" in corr.columns and "G3" in corr.columns else 0.0
    g2_g3_corr = corr.loc["G2", "G3"] if "G2" in corr.columns and "G3" in corr.columns else 0.0
    fail_g3_corr = corr.loc["failures", "G3"] if "failures" in corr.columns and "G3" in corr.columns else 0.0

    study_summary = compute_group_comparisons(df, "studytime")
    low_study = study_summary[study_summary["studytime"] == 1]
    high_study = study_summary[study_summary["studytime"] >= 3]

    low_study_mean = low_study["mean_g3"].values[0] if len(low_study) > 0 else 0.0
    high_study_mean = high_study["mean_g3"].mean() if len(high_study) > 0 else 0.0

    failures_summary = compute_group_comparisons(df, "failures")
    no_fail = failures_summary[failures_summary["failures"] == 0]
    has_fail = failures_summary[failures_summary["failures"] > 0]
    no_fail_mean = no_fail["mean_g3"].values[0] if len(no_fail) > 0 else 0.0
    has_fail_mean = has_fail["mean_g3"].mean() if len(has_fail) > 0 else 0.0

    return {
        "prior_grades_signal": (
            f"Prior period grades correlate very strongly with final achievement: "
            f"Pearson r(G1, G3) = {g1_g3_corr:.2f} and r(G2, G3) = {g2_g3_corr:.2f}. "
            f"Once mid-term performance is known, static background features add limited predictive power."
        ),
        "study_time_signal": (
            f"Students reporting <2 hours/week of study averaged {low_study_mean:.1f} on G3, "
            f"compared to {high_study_mean:.1f} for students reporting 5+ hours/week."
        ),
        "failures_signal": (
            f"Previous academic failures exhibit a strong inverse association (r = {fail_g3_corr:.2f}). "
            f"Students with zero past failures averaged {no_fail_mean:.1f} on G3, while those with at least "
            f"one past failure averaged {has_fail_mean:.1f}."
        ),
    }
