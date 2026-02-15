"""
Data loading, cleaning, and splitting pipeline for fraud detection.
Uses the PaySim synthetic financial dataset from Kaggle.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


# PaySim dataset column reference:
# step: unit of time (1 step = 1 hour), total 744 steps (30 days)
# type: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER
# amount: transaction amount
# nameOrig: origin account
# oldbalanceOrg: balance before transaction (origin)
# newbalanceOrig: balance after transaction (origin)
# nameDest: destination account
# oldbalanceDest: balance before transaction (destination)
# newbalanceDest: balance after transaction (destination)
# isFraud: target variable (1 = fraud, 0 = legit)
# isFlaggedFraud: flagged by naive rule-based system

RAW_DATA_PATH = Path("data/raw/transactions.csv")
PROCESSED_DIR = Path("data/processed")


def load_data(filepath: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw transaction data from CSV."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            "Download from: https://www.kaggle.com/datasets/ealaxi/paysim1"
        )
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} transactions | Fraud: {df['isFraud'].sum():,} ({df['isFraud'].mean():.4%})")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the raw dataset."""
    df = df.copy()

    # Drop columns that leak info or aren't useful for modeling
    drop_cols = ["nameOrig", "nameDest", "isFlaggedFraud"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Remove duplicates
    n_before = len(df)
    df = df.drop_duplicates()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"Dropped {n_dropped:,} duplicate rows")

    # Validate no nulls
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"Warning: Found null values:\n{null_counts[null_counts > 0]}")
        df = df.dropna()

    # Filter to transaction types where fraud actually occurs
    # In PaySim, fraud only happens in TRANSFER and CASH_OUT
    fraud_types = df[df["isFraud"] == 1]["type"].unique()
    df = df[df["type"].isin(fraud_types)]
    print(f"Filtered to fraud-prone types {list(fraud_types)}: {len(df):,} transactions")

    return df


def split_data(
    df: pd.DataFrame,
    target_col: str = "isFraud",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train/test sets with stratification."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Train fraud rate: {y_train.mean():.4%} | Test fraud rate: {y_test.mean():.4%}")

    return X_train, X_test, y_train, y_test


def save_processed(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str | Path = PROCESSED_DIR,
) -> None:
    """Save processed splits to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    train.to_csv(output_dir / "train.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)
    print(f"Saved processed data to {output_dir}/")


def run_pipeline(filepath: str | Path = RAW_DATA_PATH) -> tuple:
    """Execute the full preprocessing pipeline."""
    print("=" * 50)
    print("PREPROCESSING PIPELINE")
    print("=" * 50)

    df = load_data(filepath)
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    save_processed(X_train, X_test, y_train, y_test)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_pipeline()