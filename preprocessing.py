"""
Water Quality and Potability Prediction
Shared preprocessing module.

Used by:
    - water_quality_prediction.py (model training and evaluation)
    - app.py (the web interface)

Run on its own to check the data split:
    python preprocessing.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ----- Project settings (shared across the whole project) -----
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "water_potability.csv"

TARGET = "Potability"

# The eight input parameters from the project brief (Option A).
# Turbidity is kept in the CSV for EDA but is not used for modelling.
FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines",
    "Sulfate", "Conductivity", "Organic_carbon", "Trihalomethanes",
]

RANDOM_STATE = 42   # fixed seed so the split is reproducible
TEST_SIZE = 0.20    # 20% for testing, 80% for training


def load_raw_data():
    """Read the CSV file into a DataFrame."""
    return pd.read_csv(DATA_PATH)


def clean_data(df):
    """
    Apply row-level cleaning rules that do not learn from the data.
    Returns the cleaned DataFrame and a dictionary of counts for reporting.
    """
    report = {"rows_loaded": len(df)}

    # 1. Remove exact duplicate rows
    report["duplicates_removed"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # 2. Treat pH = 0 as missing (physically implausible for drinking water)
    report["zero_ph_set_to_missing"] = int((df["ph"] == 0).sum())
    df = df.copy()
    df["ph"] = df["ph"].replace(0, np.nan)

    # 3. Keep only the features used for modelling, plus the target
    df = df[FEATURES + [TARGET]]
    report["rows_after_cleaning"] = len(df)

    return df, report


def split_data(df):
    """
    Split into training (80%) and testing (20%) sets.
    stratify=y keeps the same Potable / Not Potable ratio in both sets.
    """
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return X_train, X_test, y_train, y_test


def make_preprocessor():
    """
    Preprocessing steps, packaged as a pipeline:
      1. SimpleImputer: fills missing values with the column median.
      2. StandardScaler: rescales each feature to mean 0 and standard deviation 1.

    The pipeline is fitted on training data only. Inside a larger model
    pipeline, fit() is called on X_train, so the medians and scaling
    statistics never see the test set.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def main():
    raw = load_raw_data()
    df, report = clean_data(raw)

    print("=" * 60)
    print("CLEANING REPORT")
    print("=" * 60)
    for key, value in report.items():
        print(f"{key:<28}: {value}")

    X_train, X_test, y_train, y_test = split_data(df)

    print("\n" + "=" * 60)
    print("TRAIN-TEST SPLIT")
    print("=" * 60)
    print(f"Training set : {X_train.shape[0]} rows, {X_train.shape[1]} features")
    print(f"Testing set  : {X_test.shape[0]} rows, {X_test.shape[1]} features")

    print("\nClass counts and percentages:")
    summary = pd.DataFrame({
        "Train count": y_train.value_counts().sort_index(),
        "Train %": (y_train.value_counts(normalize=True).sort_index() * 100).round(2),
        "Test count": y_test.value_counts().sort_index(),
        "Test %": (y_test.value_counts(normalize=True).sort_index() * 100).round(2),
    })
    print(summary.to_string())

    print("\nMissing values in training set (before imputation):")
    print(X_train.isnull().sum()[lambda s: s > 0].to_string())

    # Fit the preprocessing pipeline on TRAINING data only
    prep = make_preprocessor()
    prep.fit(X_train)

    print("\n" + "=" * 60)
    print("MEDIANS LEARNED FROM TRAINING DATA ONLY")
    print("=" * 60)
    medians = pd.Series(prep.named_steps["imputer"].statistics_, index=FEATURES)
    print(medians.round(3).to_string())

    # Transform both sets using the training-learned values
    X_train_ready = prep.transform(X_train)
    X_test_ready = prep.transform(X_test)

    print("\nAfter preprocessing:")
    print(f"Training array shape : {X_train_ready.shape}")
    print(f"Testing array shape  : {X_test_ready.shape}")
    print(f"Missing values left  : {np.isnan(X_train_ready).sum() + np.isnan(X_test_ready).sum()}")


if __name__ == "__main__":
    main()