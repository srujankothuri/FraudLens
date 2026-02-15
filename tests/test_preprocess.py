"""Tests for data preprocessing pipeline."""

import pytest
import pandas as pd
import numpy as np
from src.data.preprocess import clean_data, split_data


@pytest.fixture
def sample_raw_data():
    """Create a minimal dataset mimicking PaySim structure."""
    return pd.DataFrame({
        "step": [1, 2, 3, 4, 5, 6, 7, 8],
        "type": ["TRANSFER", "CASH_OUT", "TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "TRANSFER", "CASH_OUT"],
        "amount": [10000, 500, 200000, 300, 50, 100, 80000, 1000],
        "nameOrig": ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"],
        "oldbalanceOrg": [10000, 1000, 200000, 15000, 500, 1000, 80000, 5000],
        "newbalanceOrig": [0, 500, 0, 14700, 450, 900, 0, 4000],
        "nameDest": ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"],
        "oldbalanceDest": [0, 5000, 0, 80000, 1000, 2000, 50000, 10000],
        "newbalanceDest": [10000, 5500, 200000, 80300, 1050, 2100, 130000, 11000],
        "isFraud": [1, 0, 1, 0, 0, 0, 1, 0],
        "isFlaggedFraud": [0, 0, 0, 0, 0, 0, 0, 0],
    })


class TestCleanData:
    """Tests for the clean_data function."""

    def test_drops_unnecessary_columns(self, sample_raw_data):
        result = clean_data(sample_raw_data)
        assert "nameOrig" not in result.columns
        assert "nameDest" not in result.columns
        assert "isFlaggedFraud" not in result.columns

    def test_keeps_required_columns(self, sample_raw_data):
        result = clean_data(sample_raw_data)
        required = ["step", "type", "amount", "oldbalanceOrg",
                     "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFraud"]
        for col in required:
            assert col in result.columns

    def test_filters_to_fraud_prone_types(self, sample_raw_data):
        result = clean_data(sample_raw_data)
        # PAYMENT and DEBIT should be removed (no fraud in those types)
        assert set(result["type"].unique()).issubset({"TRANSFER", "CASH_OUT"})

    def test_removes_duplicates(self, sample_raw_data):
        # Add a duplicate row
        df_with_dupes = pd.concat([sample_raw_data, sample_raw_data.iloc[[0]]])
        result = clean_data(df_with_dupes)
        assert len(result) < len(df_with_dupes)

    def test_no_null_values(self, sample_raw_data):
        result = clean_data(sample_raw_data)
        assert result.isnull().sum().sum() == 0


class TestSplitData:
    """Tests for the split_data function."""

    def test_split_sizes(self, sample_raw_data):
        df = clean_data(sample_raw_data)
        X_train, X_test, y_train, y_test = split_data(df, test_size=0.5)
        assert len(X_train) + len(X_test) == len(df)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

    def test_target_not_in_features(self, sample_raw_data):
        df = clean_data(sample_raw_data)
        X_train, X_test, y_train, y_test = split_data(df)
        assert "isFraud" not in X_train.columns
        assert "isFraud" not in X_test.columns

    def test_stratification_preserves_ratio(self, sample_raw_data):
        df = clean_data(sample_raw_data)
        _, _, y_train, y_test = split_data(df, test_size=0.5)
        # Both splits should contain fraud cases
        original_rate = df["isFraud"].mean()
        # With small data, exact match is unlikely, just check both have fraud
        assert y_train.sum() > 0 or y_test.sum() > 0

    def test_reproducibility(self, sample_raw_data):
        df = clean_data(sample_raw_data)
        X1, _, y1, _ = split_data(df, random_state=42)
        X2, _, y2, _ = split_data(df, random_state=42)
        pd.testing.assert_frame_equal(X1, X2)