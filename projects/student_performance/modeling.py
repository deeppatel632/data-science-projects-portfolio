"""Regression modeling pipelines and interactive grade estimation for Student Performance."""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Base feature sets
CATEGORICAL_FEATURES = [
    "school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob",
    "reason", "guardian", "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic"
]

NUMERIC_FEATURES_EARLY = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"
]

NUMERIC_FEATURES_PREVIOUS = NUMERIC_FEATURES_EARLY + ["G1", "G2"]

REGRESSION_FEATURES_EARLY = NUMERIC_FEATURES_EARLY + CATEGORICAL_FEATURES
REGRESSION_FEATURES_PREVIOUS = NUMERIC_FEATURES_PREVIOUS + CATEGORICAL_FEATURES


def build_preprocessor(include_previous_grades: bool = False) -> Tuple[ColumnTransformer, List[str]]:
    """Build a ColumnTransformer applying scaling to numerics and one-hot encoding to categoricals."""
    num_cols = list(NUMERIC_FEATURES_PREVIOUS if include_previous_grades else NUMERIC_FEATURES_EARLY)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor, num_cols + CATEGORICAL_FEATURES


def train_regression_models(
    df: pd.DataFrame,
    include_previous_grades: bool = False,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train Dummy Baseline, Ridge Linear Regression, and Random Forest Regressor."""
    preprocessor, feature_cols = build_preprocessor(include_previous_grades=include_previous_grades)

    X = df[feature_cols].copy()
    y = df["G3"].values.astype(float)

    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state
    )

    models = {
        "Dummy Baseline (Mean)": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", DummyRegressor(strategy="mean")),
        ]),
        "Linear Regression (Ridge)": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=1.0, random_state=random_state)),
        ]),
        "Random Forest Regressor": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=100,
                max_depth=7,
                random_state=random_state,
                n_jobs=-1
            )),
        ]),
    }

    results = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        # Grade scale is bounded between 0 and 20
        y_pred_clipped = np.clip(y_pred, 0, 20)

        mae = mean_absolute_error(y_test, y_pred_clipped)
        rmse = root_mean_squared_error(y_test, y_pred_clipped)
        r2 = r2_score(y_test, y_pred_clipped)

        # 5-fold cross-validation on training set for stability verification
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error")
        cv_rmse = -float(np.mean(cv_scores))

        results[name] = {
            "pipeline": pipe,
            "y_pred": y_pred_clipped,
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "cv_rmse": float(cv_rmse),
        }

    # Extract feature importances from Random Forest
    rf_pipe = models["Random Forest Regressor"]
    rf_reg = rf_pipe.named_steps["regressor"]
    preproc_fitted = rf_pipe.named_steps["preprocessor"]

    cat_feature_names = preproc_fitted.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    all_names = np.concatenate([
        preproc_fitted.transformers_[0][2],
        cat_feature_names
    ])
    importances = pd.Series(rf_reg.feature_importances_, index=all_names).sort_values(ascending=False)

    return {
        "results": results,
        "y_test": y_test,
        "feature_importances": importances,
        "include_previous_grades": include_previous_grades,
        "test_size": len(y_test),
        "models": models,
    }


def predict_single_student(
    pipeline: Pipeline,
    student_dict: Dict[str, Any],
) -> float:
    """Predict estimated final grade G3 for a single student profile."""
    df_student = pd.DataFrame([student_dict])
    raw_pred = pipeline.predict(df_student)[0]
    return float(np.clip(raw_pred, 0.0, 20.0))
