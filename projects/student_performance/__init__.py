try:
    from .analysis import (
        compute_academic_overview,
        compute_correlations,
        compute_group_comparisons,
        generate_academic_observations,
    )
    from .data import (
        generate_synthetic_student_data,
        get_student_metadata,
        get_subject_dataset,
        load_student_data,
    )
    from .modeling import (
        REGRESSION_FEATURES_EARLY,
        REGRESSION_FEATURES_PREVIOUS,
        predict_single_student,
        train_regression_models,
    )
    from .visualizations import (
        plot_actual_vs_predicted,
        plot_correlation_heatmap,
        plot_grade_distribution,
        plot_period_grades_trajectory,
        plot_regression_comparison,
        plot_studytime_vs_grades,
    )
except (ImportError, ValueError):
    from analysis import (
        compute_academic_overview,
        compute_correlations,
        compute_group_comparisons,
        generate_academic_observations,
    )
    from data import (
        generate_synthetic_student_data,
        get_student_metadata,
        get_subject_dataset,
        load_student_data,
    )
    from modeling import (
        REGRESSION_FEATURES_EARLY,
        REGRESSION_FEATURES_PREVIOUS,
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

__all__ = [
    "load_student_data",
    "get_subject_dataset",
    "generate_synthetic_student_data",
    "get_student_metadata",
    "compute_academic_overview",
    "compute_correlations",
    "compute_group_comparisons",
    "generate_academic_observations",
    "train_regression_models",
    "predict_single_student",
    "REGRESSION_FEATURES_EARLY",
    "REGRESSION_FEATURES_PREVIOUS",
    "plot_grade_distribution",
    "plot_period_grades_trajectory",
    "plot_studytime_vs_grades",
    "plot_correlation_heatmap",
    "plot_regression_comparison",
    "plot_actual_vs_predicted",
]
