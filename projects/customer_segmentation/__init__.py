try:
    from .analysis import (
        compute_dataset_summary,
        compute_monthly_patterns,
        compute_visitor_conversion,
    )
    from .data import generate_synthetic_shoppers_data, get_dataset_metadata, load_shoppers_data
    from .modeling import (
        perform_kmeans_clustering,
        predict_with_threshold,
        train_classification_models,
    )
    from .visualizations import (
        plot_cluster_profiles,
        plot_confusion_matrix,
        plot_conversion_by_visitor,
        plot_feature_importance,
        plot_monthly_patterns,
        plot_pca_clusters,
        plot_roc_curve,
    )
except (ImportError, ValueError):
    from analysis import (
        compute_dataset_summary,
        compute_monthly_patterns,
        compute_visitor_conversion,
    )
    from data import generate_synthetic_shoppers_data, get_dataset_metadata, load_shoppers_data
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

__all__ = [
    "load_shoppers_data",
    "get_dataset_metadata",
    "generate_synthetic_shoppers_data",
    "compute_dataset_summary",
    "compute_visitor_conversion",
    "compute_monthly_patterns",
    "perform_kmeans_clustering",
    "train_classification_models",
    "predict_with_threshold",
    "plot_conversion_by_visitor",
    "plot_monthly_patterns",
    "plot_pca_clusters",
    "plot_cluster_profiles",
    "plot_roc_curve",
    "plot_confusion_matrix",
    "plot_feature_importance",
]
