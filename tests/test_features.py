"""Tests for feature engineering pipeline."""

import pytest
import pandas as pd
import numpy as np
from src.features.engineering import create_transaction_features, encode_categorical


@pytest.fixture
def sample_transactions():
    """Sample transactions with known expected feature values."""
    return pd.DataFrame({
        "step": [1, 156, 3],
        "type": ["TRANSFER", "CASH_OUT", "TRANSFER"],
        "amount": [10000, 500, 200000],
        "oldbalanceOrg": [10000, 1000, 200000],
        "newbalanceOrig": [0, 500, 0],
        "oldbalanceDest": [0, 5000, 100000],
        "newbalanceDest": [10000, 5500, 300000],
    })


class TestCreateTransactionFeatures:
    """Tests for feature creation logic."""

    def test_orig_balance_error(self, sample_transactions):
        result = create_transaction_features(sample_transactions)
        # Row 0: newbalanceOrig(0) - (oldbalanceOrg(10000) - amount(10000)) = 0
        assert result["orig_balance_error"].iloc[0] == 0
        # Row 1: newbalanceOrig(500) - (oldbalanceOrg(1000) - amount(500)) = 0
        assert result["orig_balance_error"].iloc[1] == 0

    def test_dest_balance_error(self, sample_transactions):
        result = create_transaction_features(sample_transactions)
        # Row 0: newbalanceDest(10000) - (oldbalanceDest(0) + amount(10000)) = 0
        assert result["dest_balance_error"].iloc[0] == 0

    def test_orig_zeroed(self, sample_transactions):
        result = create_transaction_features(sample_transactions)
        # Row 0: newbalanceOrig=0 → 1, Row 1: newbalanceOrig=500 → 0
        assert result["orig_zeroed"].iloc[0] == 1
        assert result["orig_zeroed"].iloc[1] == 0

    def test_amount_to_balance_ratio(self, sample_transactions):
        result = create_transaction_features(sample_transactions)
        # Row 0: 10000 / (10000 + 1) ≈ 0.9999
        assert result["amount_to_balance_ratio"].iloc[0] == pytest.approx(0.9999, rel=1e-2)

    def test_log_amount(self, sample_transactions):
        result = create_transaction_features(sample_transactions)
        # log1p(10000) ≈ 9.2104
        assert result["log_amount"].iloc[0] == pytest.approx(np.log1p(10000), rel=1e-4)

    def test_hour_of_day(self, sample_transactions):
        result = create_transaction_features(sample_transactions)
        # step=1 → hour=1, step=156 → 156%24=12, step=3 → hour=3
        assert result["hour_of_day"].iloc[0] == 1
        assert result["hour_of_day"].iloc[1] == 12
        assert result["hour_of_day"].iloc[2] == 3

    def test_is_off_hours(self, sample_transactions):
        result = create_transaction_features(sample_transactions)
        # hour=1 → off hours (1), hour=12 → not off hours (0), hour=3 → off hours (1)
        assert result["is_off_hours"].iloc[0] == 1
        assert result["is_off_hours"].iloc[1] == 0
        assert result["is_off_hours"].iloc[2] == 1

    def test_day_calculation(self, sample_transactions):
        result = create_transaction_features(sample_transactions)
        # step=1 → day 0, step=156 → day 6, step=3 → day 0
        assert result["day"].iloc[0] == 0
        assert result["day"].iloc[1] == 6

    def test_no_null_features(self, sample_transactions):
        result = create_transaction_features(sample_transactions)
        assert result.isnull().sum().sum() == 0


class TestEncodeCategorical:
    """Tests for one-hot encoding."""

    def test_removes_original_type_column(self, sample_transactions):
        result = encode_categorical(sample_transactions)
        assert "type" not in result.columns

    def test_creates_dummy_columns(self, sample_transactions):
        result = encode_categorical(sample_transactions)
        assert "type_TRANSFER" in result.columns
        assert "type_CASH_OUT" in result.columns

    def test_dummy_values_are_binary(self, sample_transactions):
        result = encode_categorical(sample_transactions)
        assert set(result["type_TRANSFER"].unique()).issubset({0, 1})
        assert set(result["type_CASH_OUT"].unique()).issubset({0, 1})

    def test_dummies_are_mutually_exclusive(self, sample_transactions):
        result = encode_categorical(sample_transactions)
        # Each row should have exactly one type=1
        type_cols = [c for c in result.columns if c.startswith("type_")]
        row_sums = result[type_cols].sum(axis=1)
        assert (row_sums == 1).all()