"""Data loader, validator, and synthetic fallback for UCI Student Performance dataset."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from shared.utilities import resolve_data_path

EXPECTED_COLUMNS = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    "traveltime", "studytime", "failures", "schoolsup", "famsup",
    "paid", "activities", "nursery", "higher", "internet",
    "romantic", "famrel", "freetime", "goout", "Dalc", "Walc",
    "health", "absences", "G1", "G2", "G3"
]

STUDENT_METADATA = {
    "source_name": "UCI Student Performance Dataset",
    "source_url": "https://archive.ics.uci.edu/dataset/320/student+performance",
    "citation": "Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school student performance. In Proceedings of 5th Annual Future Business Technology Conference (FUBUTEC 2008), pp. 5-12, Porto, Portugal.",
    "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
    "description": "Secondary school student achievement in Mathematics (395 records) and Portuguese language (649 records) from two Portuguese schools (Gabriel Pereira and Mousinho da Silveira).",
}


def get_student_metadata() -> Dict[str, Any]:
    """Return static metadata and citation for the UCI Student dataset."""
    return dict(STUDENT_METADATA)


def generate_synthetic_student_data(subject: str = "mat", n_samples: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    """Generate deterministic synthetic student records with matching feature correlations."""
    rng = np.random.default_rng(seed)
    n = n_samples if n_samples else (395 if subject == "mat" else 649)

    schools = rng.choice(["GP", "MS"], size=n, p=[0.88, 0.12])
    sex = rng.choice(["F", "M"], size=n, p=[0.52, 0.48])
    age = rng.choice([15, 16, 17, 18, 19, 20, 21, 22], size=n, p=[0.21, 0.27, 0.25, 0.20, 0.05, 0.01, 0.005, 0.005])
    address = rng.choice(["U", "R"], size=n, p=[0.78, 0.22])
    famsize = rng.choice(["GT3", "LE3"], size=n, p=[0.71, 0.29])
    pstatus = rng.choice(["T", "A"], size=n, p=[0.89, 0.11])

    medu = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.01, 0.20, 0.26, 0.25, 0.28])
    fedu = np.clip(medu + rng.choice([-1, 0, 1], size=n, p=[0.25, 0.50, 0.25]), 0, 4)

    mjob = rng.choice(["at_home", "health", "other", "services", "teacher"], size=n, p=[0.15, 0.09, 0.36, 0.26, 0.14])
    fjob = rng.choice(["at_home", "health", "other", "services", "teacher"], size=n, p=[0.05, 0.05, 0.55, 0.28, 0.07])
    reason = rng.choice(["course", "home", "other", "reputation"], size=n, p=[0.37, 0.27, 0.09, 0.27])
    guardian = rng.choice(["mother", "father", "other"], size=n, p=[0.70, 0.23, 0.07])

    traveltime = rng.choice([1, 2, 3, 4], size=n, p=[0.65, 0.27, 0.06, 0.02])
    studytime = rng.choice([1, 2, 3, 4], size=n, p=[0.27, 0.50, 0.17, 0.06])
    failures = rng.choice([0, 1, 2, 3], size=n, p=[0.80, 0.13, 0.04, 0.03])

    schoolsup = rng.choice(["yes", "no"], size=n, p=[0.13, 0.87])
    famsup = rng.choice(["yes", "no"], size=n, p=[0.61, 0.39])
    paid = rng.choice(["yes", "no"], size=n, p=[0.46 if subject == "mat" else 0.10, 0.54 if subject == "mat" else 0.90])
    activities = rng.choice(["yes", "no"], size=n, p=[0.51, 0.49])
    nursery = rng.choice(["yes", "no"], size=n, p=[0.79, 0.21])
    higher = rng.choice(["yes", "no"], size=n, p=[0.95, 0.05])
    internet = rng.choice(["yes", "no"], size=n, p=[0.83, 0.17])
    romantic = rng.choice(["yes", "no"], size=n, p=[0.33, 0.67])

    famrel = np.clip(rng.choice([1, 2, 3, 4, 5], size=n, p=[0.02, 0.04, 0.17, 0.50, 0.27]), 1, 5)
    freetime = np.clip(rng.choice([1, 2, 3, 4, 5], size=n, p=[0.05, 0.16, 0.40, 0.29, 0.10]), 1, 5)
    goout = np.clip(rng.choice([1, 2, 3, 4, 5], size=n, p=[0.06, 0.26, 0.33, 0.22, 0.13]), 1, 5)
    dalc = np.clip(rng.choice([1, 2, 3, 4, 5], size=n, p=[0.70, 0.19, 0.07, 0.02, 0.02]), 1, 5)
    walc = np.clip(rng.choice([1, 2, 3, 4, 5], size=n, p=[0.40, 0.22, 0.20, 0.13, 0.05]), 1, 5)
    health = np.clip(rng.choice([1, 2, 3, 4, 5], size=n, p=[0.12, 0.11, 0.23, 0.17, 0.37]), 1, 5)
    absences = np.clip(rng.negative_binomial(n=1.5, p=0.2, size=n), 0, 75)

    # Grades simulation with realistic Portuguese educational grading
    base_mean = 10.4 if subject == "mat" else 11.9
    latent_ability = (
        base_mean
        + 0.8 * studytime
        - 1.3 * failures
        + 0.4 * medu
        - 0.05 * absences
        - 0.3 * (walc >= 4)
        + rng.normal(0, 2.2, size=n)
    )

    g1 = np.clip(np.round(latent_ability + rng.normal(0, 1.2, size=n)), 0, 20).astype(int)
    g2 = np.clip(np.round(0.85 * g1 + 0.15 * latent_ability + rng.normal(0, 1.0, size=n)), 0, 20).astype(int)
    g3 = np.clip(np.round(0.90 * g2 + 0.10 * latent_ability + rng.normal(0, 0.9, size=n)), 0, 20).astype(int)

    # Some zero-outdropouts
    dropout_mask = (rng.uniform(0, 1, size=n) < 0.08) & (failures > 0)
    g3[dropout_mask] = 0

    return pd.DataFrame({
        "school": schools, "sex": sex, "age": age, "address": address,
        "famsize": famsize, "Pstatus": pstatus, "Medu": medu, "Fedu": fedu,
        "Mjob": mjob, "Fjob": fjob, "reason": reason, "guardian": guardian,
        "traveltime": traveltime, "studytime": studytime, "failures": failures,
        "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
        "activities": activities, "nursery": nursery, "higher": higher,
        "internet": internet, "romantic": romantic, "famrel": famrel,
        "freetime": freetime, "goout": goout, "Dalc": dalc, "Walc": walc,
        "health": health, "absences": absences, "G1": g1, "G2": g2, "G3": g3
    })


def load_student_data(
    subject: str = "mat"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load mathematics ('mat') or Portuguese ('por') dataset."""
    filename = f"student-{subject.lower()}.csv"
    resolved_path = resolve_data_path(
        f"projects/student_performance/data/{filename}",
        calling_file=__file__
    )

    if resolved_path.exists():
        try:
            # Source UCI files use semicolon delimiters
            df = pd.read_csv(resolved_path, sep=";")
            # Fallback check if it was comma separated
            if len(df.columns) == 1:
                df = pd.read_csv(resolved_path, sep=",")

            if all(col in df.columns for col in ["school", "G1", "G2", "G3"]):
                metadata = {
                    **STUDENT_METADATA,
                    "subject": "Mathematics" if subject == "mat" else "Portuguese Language",
                    "is_synthetic": False,
                    "record_count": len(df),
                    "file_path": str(resolved_path),
                }
                return df, metadata
        except Exception:
            pass

    # Deterministic synthetic fallback
    df_synthetic = generate_synthetic_student_data(subject=subject, seed=42)
    metadata = {
        **STUDENT_METADATA,
        "subject": "Mathematics" if subject == "mat" else "Portuguese Language",
        "is_synthetic": True,
        "record_count": len(df_synthetic),
        "file_path": "SYNTHETIC_GENERATED",
        "notice": f"Official UCI file {filename} could not be loaded. Used deterministic synthetic fallback.",
    }
    return df_synthetic, metadata


def get_subject_dataset(subject: str = "mat") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Convenience accessor for either subject."""
    return load_student_data(subject=subject)
