"""Data generation, validation, and loading for restaurant sales and performance."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from shared.utilities import resolve_data_path

REQUIRED_COLUMNS = [
    "order_id",
    "line_item_id",
    "order_timestamp",
    "outlet",
    "ordering_channel",
    "payment_method",
    "item_name",
    "category",
    "quantity",
    "unit_price",
    "unit_cost",
    "discount_rate",
    "net_revenue",
    "estimated_gross_profit",
]

MENU_CATALOG = [
    # Mains
    {"item_name": "Artisan Burger", "category": "Mains", "unit_price": 16.50, "unit_cost": 5.20, "weight": 0.14},
    {"item_name": "Ribeye Steak", "category": "Mains", "unit_price": 34.00, "unit_cost": 13.50, "weight": 0.08},
    {"item_name": "Grilled Salmon", "category": "Mains", "unit_price": 26.00, "unit_cost": 10.00, "weight": 0.09},
    {"item_name": "Margherita Pizza", "category": "Mains", "unit_price": 15.00, "unit_cost": 3.80, "weight": 0.12},
    {"item_name": "Pasta Primavera", "category": "Mains", "unit_price": 17.50, "unit_cost": 4.50, "weight": 0.07},
    {"item_name": "Chicken Tikka Bowl", "category": "Mains", "unit_price": 18.00, "unit_cost": 5.10, "weight": 0.08},
    # Appetizers
    {"item_name": "Truffle Fries", "category": "Appetizers", "unit_price": 9.50, "unit_cost": 2.10, "weight": 0.11},
    {"item_name": "Crispy Calamari", "category": "Appetizers", "unit_price": 13.00, "unit_cost": 4.20, "weight": 0.06},
    {"item_name": "Buffalo Wings", "category": "Appetizers", "unit_price": 12.50, "unit_cost": 3.90, "weight": 0.08},
    {"item_name": "Caesar Salad", "category": "Appetizers", "unit_price": 11.00, "unit_cost": 2.80, "weight": 0.07},
    # Desserts
    {"item_name": "Chocolate Lava Cake", "category": "Desserts", "unit_price": 8.50, "unit_cost": 2.20, "weight": 0.06},
    {"item_name": "Tiramisu", "category": "Desserts", "unit_price": 9.00, "unit_cost": 2.40, "weight": 0.05},
    {"item_name": "New York Cheesecake", "category": "Desserts", "unit_price": 8.00, "unit_cost": 2.10, "weight": 0.05},
    # Beverages
    {"item_name": "Craft Draft Beer", "category": "Beverages", "unit_price": 7.50, "unit_cost": 1.70, "weight": 0.11},
    {"item_name": "House Red Wine", "category": "Beverages", "unit_price": 10.00, "unit_cost": 2.80, "weight": 0.07},
    {"item_name": "Artisanal Lemonade", "category": "Beverages", "unit_price": 4.50, "unit_cost": 0.80, "weight": 0.09},
    {"item_name": "Espresso Double", "category": "Beverages", "unit_price": 4.00, "unit_cost": 0.50, "weight": 0.08},
    {"item_name": "Sparkling Water", "category": "Beverages", "unit_price": 3.50, "unit_cost": 0.40, "weight": 0.07},
    # Sides
    {"item_name": "Garlic Bread", "category": "Sides", "unit_price": 6.00, "unit_cost": 1.20, "weight": 0.07},
    {"item_name": "Sweet Potato Fries", "category": "Sides", "unit_price": 7.00, "unit_cost": 1.60, "weight": 0.06},
]


def generate_synthetic_restaurant_data(
    n_orders: int = 7500,
    seed: int = 42,
    start_date: str = "2025-01-01",
    end_date: str = "2025-12-31"
) -> pd.DataFrame:
    """Generate a realistic, multi-outlet synthetic restaurant order dataset."""
    rng = np.random.default_rng(seed)

    outlets = ["Downtown Bistro", "Uptown Grill", "Westside Cantina", "Suburban Cafe"]
    outlet_weights = [0.36, 0.28, 0.21, 0.15]

    channels = ["Dine-in", "Takeout", "Delivery", "Mobile App"]
    channel_weights = [0.45, 0.22, 0.21, 0.12]

    payments = ["Credit Card", "Digital Wallet", "Cash", "Gift Card"]
    payment_weights = [0.62, 0.24, 0.10, 0.04]

    # Generate dates across calendar year
    dates_range = pd.date_range(start=start_date, end=end_date, freq="D")
    # Add day-of-week weight (Fridays & Saturdays have higher volume)
    day_weights = np.array([1.3 if d.weekday() in [4, 5] else 0.9 for d in dates_range])
    day_weights /= day_weights.sum()

    order_dates = rng.choice(dates_range, size=n_orders, p=day_weights)

    # Realistic meal hours: Lunch rush (11:30 - 14:00), Dinner rush (17:30 - 21:30)
    hours_dist = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.02, # 0-10
        0.11, 0.16, 0.12, 0.04, 0.03, 0.03, # 11-16 Lunch peak
        0.08, 0.15, 0.14, 0.08, 0.03, 0.01, 0.0 # 17-23 Dinner peak
    ]
    hours_dist = np.array(hours_dist) / sum(hours_dist)

    items_list = [item["item_name"] for item in MENU_CATALOG]
    items_weights = np.array([item["weight"] for item in MENU_CATALOG])
    items_weights /= items_weights.sum()
    catalog_by_name = {item["item_name"]: item for item in MENU_CATALOG}

    rows = []
    order_counter = 10001

    for i in range(n_orders):
        order_id = f"ORD-2025-{order_counter}"
        order_counter += 1

        chosen_date = pd.Timestamp(order_dates[i])
        chosen_hour = int(rng.choice(range(24), p=hours_dist))
        chosen_minute = int(rng.choice(range(60)))
        timestamp = pd.Timestamp(
            year=chosen_date.year,
            month=chosen_date.month,
            day=chosen_date.day,
            hour=chosen_hour,
            minute=chosen_minute,
            second=int(rng.choice(range(60)))
        )

        outlet = rng.choice(outlets, p=outlet_weights)
        channel = rng.choice(channels, p=channel_weights)
        payment = rng.choice(payments, p=payment_weights)

        # Order-level discount rate: 80% have 0% discount, remainder have 5-20%
        discount_rate = float(rng.choice([0.0, 0.05, 0.10, 0.15, 0.20], p=[0.80, 0.08, 0.06, 0.04, 0.02]))

        # Number of line items in order: 1 to 4
        num_items = int(rng.choice([1, 2, 3, 4], p=[0.38, 0.35, 0.18, 0.09]))
        chosen_items = rng.choice(items_list, size=num_items, replace=False, p=items_weights)

        for line_num, item_name in enumerate(chosen_items, start=1):
            info = catalog_by_name[item_name]
            quantity = int(rng.choice([1, 2, 3], p=[0.78, 0.18, 0.04]))
            unit_price = float(info["unit_price"])
            unit_cost = float(info["unit_cost"])

            gross_sales = round(quantity * unit_price, 2)
            discount_amount = round(gross_sales * discount_rate, 2)
            net_revenue = round(gross_sales - discount_amount, 2)
            cogs = round(quantity * unit_cost, 2)
            estimated_gross_profit = round(net_revenue - cogs, 2)
            gross_margin = (estimated_gross_profit / net_revenue) if net_revenue > 0 else 0.0

            rows.append({
                "order_id": order_id,
                "line_item_id": f"{order_id}-{line_num}",
                "order_timestamp": timestamp,
                "outlet": outlet,
                "ordering_channel": channel,
                "payment_method": payment,
                "item_name": item_name,
                "category": info["category"],
                "quantity": quantity,
                "unit_price": unit_price,
                "unit_cost": unit_cost,
                "discount_rate": discount_rate,
                "gross_sales": gross_sales,
                "discount_amount": discount_amount,
                "net_revenue": net_revenue,
                "cogs": cogs,
                "estimated_gross_profit": estimated_gross_profit,
                "gross_margin": gross_margin,
            })

    df = pd.DataFrame(rows)
    df["order_timestamp"] = pd.to_datetime(df["order_timestamp"])
    df["date"] = df["order_timestamp"].dt.date
    df["hour"] = df["order_timestamp"].dt.hour
    df["day_of_week"] = df["order_timestamp"].dt.day_name()
    return df


def validate_restaurant_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate uploaded restaurant transaction dataframe against schema rules."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"

    if len(df) < 10:
        return False, f"Dataset contains too few records ({len(df)} rows)."

    # Check order_timestamp parsability
    try:
        pd.to_datetime(df["order_timestamp"])
    except Exception as e:
        return False, f"Failed to parse 'order_timestamp' into valid dates: {e}"

    # Numerical validation
    numeric_checks = ["quantity", "unit_price", "unit_cost", "net_revenue", "estimated_gross_profit"]
    for col in numeric_checks:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, f"Column '{col}' must be numeric."
        if (df[col] < 0).any() and col in ["quantity", "unit_price", "unit_cost"]:
            return False, f"Column '{col}' contains negative values, which are invalid."

    return True, "Valid restaurant dataset."


def load_restaurant_data(
    file_path: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load default sample dataset from CSV, generating it deterministically if not present."""
    default_csv = "projects/restaurant_analytics/data/restaurant_sales_sample.csv"
    resolved_path = resolve_data_path(file_path or default_csv, calling_file=__file__)

    if resolved_path.exists():
        try:
            df = pd.read_csv(resolved_path)
            is_valid, msg = validate_restaurant_dataframe(df)
            if is_valid:
                df["order_timestamp"] = pd.to_datetime(df["order_timestamp"])
                df["date"] = df["order_timestamp"].dt.date
                df["hour"] = df["order_timestamp"].dt.hour
                df["day_of_week"] = df["order_timestamp"].dt.day_name()
                metadata = {
                    "source_name": "Deterministic Synthetic Restaurant Demonstration Data",
                    "is_synthetic": True,
                    "record_count": len(df),
                    "file_path": str(resolved_path),
                    "notice": "Simulated multi-outlet restaurant transaction records generated with calibrated realistic distributions.",
                }
                return df, metadata
        except Exception:
            pass

    # Generate and bundle sample CSV
    df = generate_synthetic_restaurant_data(n_orders=7500, seed=42)
    try:
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(resolved_path, index=False)
    except Exception:
        pass

    metadata = {
        "source_name": "Deterministic Synthetic Restaurant Demonstration Data",
        "is_synthetic": True,
        "record_count": len(df),
        "file_path": str(resolved_path),
        "notice": "Simulated multi-outlet restaurant transaction records generated with calibrated realistic distributions.",
    }
    return df, metadata
