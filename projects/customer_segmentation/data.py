"""Data loading, validation, and reproducible synthetic fallback for Online Shoppers Intention dataset."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from shared.utilities import resolve_data_path

EXPECTED_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
    "Revenue",
]

UCI_METADATA = {
    "source_name": "UCI Online Shoppers Purchasing Intention Dataset",
    "source_url": "https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset",
    "citation": "Sakar, C.O., Polat, S.O., Katircioglu, M. et al. (2019). Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks. Neural Computing and Applications, 31, 6893–6908.",
    "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
    "description": "12,330 e-commerce sessions over a 1-year period to avoid holiday campaign bias.",
}


def get_dataset_metadata() -> Dict[str, Any]:
    """Return static metadata and citation for the UCI dataset."""
    return dict(UCI_METADATA)


def generate_synthetic_shoppers_data(n_samples: int = 12330, seed: int = 42) -> pd.DataFrame:
    """Generate a reproducible, realistically distributed synthetic fallback dataset."""
    rng = np.random.default_rng(seed)

    # Visitor type distribution: ~85% Returning, ~14% New, ~1% Other
    visitor_types = rng.choice(["Returning_Visitor", "New_Visitor", "Other"], size=n_samples, p=[0.85, 0.14, 0.01])

    # Months distribution with e-commerce seasonal weighting (Nov, May, Dec peaks)
    months_list = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_weights = [0.015, 0.155, 0.27, 0.025, 0.035, 0.035, 0.04, 0.045, 0.245, 0.135]
    months = rng.choice(months_list, size=n_samples, p=month_weights)

    # Weekend binary
    weekend = rng.choice([False, True], size=n_samples, p=[0.77, 0.23])

    # Browsing counts (right-skewed Poisson / Negative Binomial style)
    admin = rng.poisson(lam=2.3, size=n_samples)
    admin_dur = admin * rng.exponential(scale=35.0, size=n_samples)

    info = rng.poisson(lam=0.5, size=n_samples)
    info_dur = info * rng.exponential(scale=65.0, size=n_samples)

    prod = np.maximum(1, rng.poisson(lam=31.0, size=n_samples))
    prod_dur = prod * rng.exponential(scale=38.0, size=n_samples)

    # Bounce & exit rates (0.0 to 0.2 mostly)
    bounce = np.clip(rng.beta(a=0.5, b=15.0, size=n_samples), 0.0, 0.2)
    exit_rate = np.clip(bounce + rng.uniform(0.01, 0.06, size=n_samples), 0.0, 0.2)

    # Special day (proximity to Mother's day, Valentine's, etc.)
    special_day = rng.choice([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], size=n_samples, p=[0.89, 0.02, 0.02, 0.03, 0.02, 0.02])

    # Page values (many 0, some positive)
    has_pv = rng.choice([0, 1], size=n_samples, p=[0.78, 0.22])
    page_values = has_pv * rng.exponential(scale=25.0, size=n_samples)

    # Technical attributes
    os_sys = rng.choice([1, 2, 3, 4, 8], size=n_samples, p=[0.21, 0.54, 0.20, 0.03, 0.02])
    browser = rng.choice([1, 2, 4, 5, 10], size=n_samples, p=[0.19, 0.65, 0.06, 0.04, 0.06])
    region = rng.choice(range(1, 10), size=n_samples)
    traffic_type = rng.choice(range(1, 14), size=n_samples)

    # Latent conversion probability: higher for high page values, high product visits, low bounce
    logit = (
        -2.5
        + 0.05 * np.log1p(prod)
        - 4.0 * bounce
        - 3.0 * exit_rate
        + 0.12 * page_values
        + 0.4 * (visitor_types == "New_Visitor")
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    revenue = rng.uniform(0, 1, size=n_samples) < prob

    df = pd.DataFrame({
        "Administrative": admin,
        "Administrative_Duration": admin_dur,
        "Informational": info,
        "Informational_Duration": info_dur,
        "ProductRelated": prod,
        "ProductRelated_Duration": prod_dur,
        "BounceRates": bounce,
        "ExitRates": exit_rate,
        "PageValues": page_values,
        "SpecialDay": special_day,
        "Month": months,
        "OperatingSystems": os_sys,
        "Browser": browser,
        "Region": region,
        "TrafficType": traffic_type,
        "VisitorType": visitor_types,
        "Weekend": weekend,
        "Revenue": revenue,
    })
    return df


def validate_shoppers_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate that the dataset meets required schema constraints."""
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"
    if len(df) < 100:
        return False, f"Dataset has too few records ({len(df)} rows)."
    if "Revenue" not in df.columns:
        return False, "Target column 'Revenue' is missing."
    return True, "Valid dataset."


def load_shoppers_data(
    file_path: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load the shoppers dataset from CSV or create synthetic fallback if unavailable."""
    resolved_path = resolve_data_path(
        file_path or "projects/customer_segmentation/data/online_shoppers_intention.csv",
        calling_file=__file__
    )

    if resolved_path.exists():
        try:
            df = pd.read_csv(resolved_path)
            is_valid, msg = validate_shoppers_dataframe(df)
            if is_valid:
                # Standardize boolean types
                df["Revenue"] = df["Revenue"].astype(bool)
                df["Weekend"] = df["Weekend"].astype(bool)
                metadata = {
                    **UCI_METADATA,
                    "is_synthetic": False,
                    "record_count": len(df),
                    "file_path": str(resolved_path),
                }
                return df, metadata
        except Exception:
            pass  # Fall through to synthetic fallback

    # Fallback to deterministic synthetic dataset
    df_synthetic = generate_synthetic_shoppers_data(n_samples=12330, seed=42)
    metadata = {
        **UCI_METADATA,
        "is_synthetic": True,
        "record_count": len(df_synthetic),
        "file_path": "SYNTHETIC_GENERATED",
        "notice": "Official UCI file was unreadable or missing. Generated deterministic synthetic dataset with identical schema.",
    }
    return df_synthetic, metadata
