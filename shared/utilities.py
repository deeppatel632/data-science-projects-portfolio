"""Cross-project utility functions for path resolution, formatting, and downloads."""

from pathlib import Path
from typing import Optional, Union

import pandas as pd
import streamlit as st


def resolve_data_path(
    relative_path: Union[str, Path],
    calling_file: Optional[str] = None
) -> Path:
    """Resolve a data file path flexibly across root, project-local, and exported environments.

    Search order:
    1. Direct path as provided (if exists)
    2. Path relative to current working directory
    3. Path relative to calling_file directory (if provided)
    4. Path relative to repository root (detected via markers like requirements.txt)
    """
    path_obj = Path(relative_path)
    if path_obj.is_absolute() and path_obj.exists():
        return path_obj

    # Check cwd
    cwd_path = Path.cwd() / path_obj
    if cwd_path.exists():
        return cwd_path.resolve()

    # Check calling file directory
    if calling_file:
        calling_dir = Path(calling_file).resolve().parent
        caller_path = calling_dir / path_obj
        if caller_path.exists():
            return caller_path.resolve()
        # Also check calling_dir/data/...
        caller_data = calling_dir / "data" / path_obj.name
        if caller_data.exists():
            return caller_data.resolve()

    # Search upwards from cwd or calling_file for repo root
    start_dir = Path(calling_file).resolve().parent if calling_file else Path.cwd()
    curr = start_dir
    for _ in range(5):
        candidate = curr / path_obj
        if candidate.exists():
            return candidate.resolve()
        if (curr / "requirements.txt").exists() or (curr / ".git").exists():
            candidate = curr / path_obj
            if candidate.exists():
                return candidate.resolve()
            break
        if curr.parent == curr:
            break
        curr = curr.parent

    # Return cwd_path as default target
    return cwd_path


def format_currency(val: float) -> str:
    """Format floating point numbers as clean currency strings."""
    if pd.isna(val):
        return "$0.00"
    if abs(val) >= 1_000_000:
        return f"${val / 1_000_000:,.2f}M"
    elif abs(val) >= 10_000:
        return f"${val:,.0f}"
    else:
        return f"${val:,.2f}"


def format_number(val: float, decimals: int = 0) -> str:
    """Format numeric values with thousands separators."""
    if pd.isna(val):
        return "0"
    if decimals == 0:
        return f"{int(round(val)):,}"
    return f"{val:,.{decimals}f}"


def format_percentage(val: float, decimals: int = 1) -> str:
    """Format decimal or percentage values cleanly."""
    if pd.isna(val):
        return "0.0%"
    # If val is between 0 and 1, assume ratio unless > 1
    pct = val * 100 if abs(val) <= 1.0 else val
    return f"{pct:.{decimals}f}%"


def safe_dataframe_download(
    df: pd.DataFrame,
    filename: str,
    label: str = "Download CSV",
    key: Optional[str] = None
) -> None:
    """Provide a reliable CSV download button for Streamlit."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        key=key,
        help="Export table data directly to CSV."
    )
