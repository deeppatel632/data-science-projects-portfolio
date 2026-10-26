try:
    from .analysis import (
        compute_abc_analysis,
        compute_business_kpis,
        compute_channel_sales,
        compute_hourly_sales,
        compute_menu_item_performance,
        compute_monthly_sales,
        compute_outlet_sales,
        compute_sales_forecast,
        compute_weekday_sales,
        generate_operational_insights,
    )
    from .data import (
        REQUIRED_COLUMNS,
        generate_synthetic_restaurant_data,
        load_restaurant_data,
        validate_restaurant_dataframe,
    )
    from .visualizations import (
        plot_channel_share,
        plot_forecast_vs_actual,
        plot_hourly_distribution,
        plot_menu_item_ranking,
        plot_menu_scatter,
        plot_outlet_breakdown,
        plot_sales_trend,
    )
except (ImportError, ValueError):
    from analysis import (
        compute_abc_analysis,
        compute_business_kpis,
        compute_channel_sales,
        compute_hourly_sales,
        compute_menu_item_performance,
        compute_monthly_sales,
        compute_outlet_sales,
        compute_sales_forecast,
        compute_weekday_sales,
        generate_operational_insights,
    )
    from data import (
        REQUIRED_COLUMNS,
        generate_synthetic_restaurant_data,
        load_restaurant_data,
        validate_restaurant_dataframe,
    )
    from visualizations import (
        plot_channel_share,
        plot_forecast_vs_actual,
        plot_hourly_distribution,
        plot_menu_item_ranking,
        plot_menu_scatter,
        plot_outlet_breakdown,
        plot_sales_trend,
    )

__all__ = [
    "load_restaurant_data",
    "generate_synthetic_restaurant_data",
    "validate_restaurant_dataframe",
    "REQUIRED_COLUMNS",
    "compute_business_kpis",
    "compute_outlet_sales",
    "compute_channel_sales",
    "compute_hourly_sales",
    "compute_weekday_sales",
    "compute_monthly_sales",
    "compute_menu_item_performance",
    "compute_abc_analysis",
    "generate_operational_insights",
    "compute_sales_forecast",
    "plot_sales_trend",
    "plot_outlet_breakdown",
    "plot_channel_share",
    "plot_hourly_distribution",
    "plot_menu_item_ranking",
    "plot_menu_scatter",
    "plot_forecast_vs_actual",
]
