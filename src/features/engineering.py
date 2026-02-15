"""
Feature engineering pipeline for fraud detection.
Transforms raw transaction data into model-ready features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def create_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from raw transaction data.

    Key fraud signals in financial transactions:
    - Balance discrepancies (money appears/disappears)
    - Large amounts relative to account balance
    - Unusual transaction timing patterns
    """
    df = df.copy()

    # --- Balance-based features (strong fraud indicators) ---

    # Error in origin balance: expected vs actual
    # If newbalanceOrig != oldbalanceOrg - amount, something is off
    df["orig_balance_error"] = (
        df["newbalanceOrig"] - (df["oldbalanceOrg"] - df["amount"])
    )

    # Error in destination balance
    df["dest_balance_error"] = (
        df["newbalanceDest"] - (df["oldbalanceDest"] + df["amount"])
    )

    # Did the origin account get completely drained?
    df["orig_zeroed"] = (df["newbalanceOrig"] == 0).astype(int)

    # Ratio of transaction amount to origin balance (avoid div by zero)
    df["amount_to_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

    # --- Amount-based features ---

    # Log-transformed amount (reduces skewness)
    df["log_amount"] = np.log1p(df["amount"])

    # --- Time-based features ---

    # Hour of day (step is in hours, 24-hour cycle)
    df["hour_of_day"] = df["step"] % 24

    # Day of simulation
    df["day"] = df["step"] // 24

    # Is it during off-hours? (midnight to 6am)
    df["is_off_hours"] = ((df["hour_of_day"] >= 0) & (df["hour_of_day"] < 6)).astype(int)

    return df


def encode_categorical(df: pd.DataFrame, col: str = "type") -> tuple[pd.DataFrame, LabelEncoder]:
    """One-hot encode the transaction type column."""
    df = df.copy()
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
    df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
    # Ensure all dummy columns are int
    for c in dummies.columns:
        df[c] = df[c].astype(int)
    return df


def get_feature_names() -> list[str]:
    """Return the list of engineered feature names (for reference)."""
    return [
        "step", "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "orig_balance_error", "dest_balance_error",
        "orig_zeroed", "amount_to_balance_ratio",
        "log_amount", "hour_of_day", "day", "is_off_hours",
        "type_CASH_OUT", "type_TRANSFER",
    ]


def run_feature_pipeline(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply full feature engineering to train and test sets."""
    print("Engineering features...")

    X_train = create_transaction_features(X_train)
    X_test = create_transaction_features(X_test)

    X_train = encode_categorical(X_train)
    X_test = encode_categorical(X_test)

    # Ensure both have the same columns (handle edge cases)
    common_cols = sorted(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    print(f"Feature count: {len(common_cols)}")
    print(f"Features: {common_cols}")

    return X_train, X_test


if __name__ == "__main__":
    # Quick test with dummy data
    sample = pd.DataFrame({
        "step": [1, 25, 50],
        "type": ["TRANSFER", "CASH_OUT", "TRANSFER"],
        "amount": [10000, 500, 200000],
        "oldbalanceOrg": [10000, 1000, 200000],
        "newbalanceOrig": [0, 500, 0],
        "oldbalanceDest": [0, 5000, 100000],
        "newbalanceDest": [10000, 5500, 300000],
    })
    result = create_transaction_features(sample)
    print(result.head())