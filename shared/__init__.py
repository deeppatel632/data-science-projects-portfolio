"""Shared package containing cross-project styles, theme helpers, and path utilities."""

from .styles import (
    PROJECT_THEMES,
    apply_custom_css,
    get_plotly_template,
    render_callout,
    render_metric_card,
    render_page_header,
)
from .utilities import (
    format_currency,
    format_number,
    format_percentage,
    resolve_data_path,
    safe_dataframe_download,
)

__all__ = [
    "apply_custom_css",
    "render_page_header",
    "render_metric_card",
    "render_callout",
    "get_plotly_template",
    "PROJECT_THEMES",
    "resolve_data_path",
    "format_currency",
    "format_number",
    "format_percentage",
    "safe_dataframe_download",
]
