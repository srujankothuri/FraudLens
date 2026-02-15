"""
Prediction + SHAP explanation module.
This is the USP — every prediction comes with a human-readable explanation
of WHY the model flagged (or cleared) a transaction.
"""

import shap
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.models.train import MODEL_PATH, load_model
from src.features.engineering import create_transaction_features, encode_categorical



# Features in exact order the model was trained on
TRAINING_FEATURES = [
    "amount", "amount_to_balance_ratio", "day", "dest_balance_error",
    "hour_of_day", "is_off_hours", "log_amount", "newbalanceDest",
    "newbalanceOrig", "oldbalanceDest", "oldbalanceOrg", "orig_balance_error",
    "orig_zeroed", "step", "type_CASH_OUT", "type_TRANSFER",
]


def prepare_single_transaction(transaction: dict) -> pd.DataFrame:
    """
    Prepare a single transaction dict for prediction.
    Applies the same feature engineering as training and ensures
    column order + presence matches what the model expects.
    """
    df = pd.DataFrame([transaction])
    df = create_transaction_features(df)
    df = encode_categorical(df)

    # Add any missing dummy columns (e.g., type_CASH_OUT when txn is TRANSFER)
    for col in TRAINING_FEATURES:
        if col not in df.columns:
            df[col] = 0

    # Ensure exact column order from training
    df = df[TRAINING_FEATURES]
    return df


def get_shap_explainer(model):
    """
    Create a SHAP TreeExplainer for the XGBoost model.

    We use TreeExplainer because:
    - It's exact (not approximate) for tree-based models
    - Much faster than KernelExplainer
    - Works directly with XGBoost's internal structure
    """
    # Extract the XGBoost classifier from the imblearn pipeline
    xgb_model = model.named_steps["classifier"]
    explainer = shap.TreeExplainer(xgb_model)
    return explainer


def predict_with_explanation(
    transaction: dict,
    model=None,
    explainer=None,
) -> dict:
    """
    Make a prediction and generate SHAP explanation for a single transaction.

    Returns:
        dict with:
        - prediction: 0 (legit) or 1 (fraud)
        - fraud_probability: float between 0 and 1
        - risk_level: "LOW", "MEDIUM", or "HIGH"
        - shap_values: per-feature SHAP contributions
        - top_reasons: top features pushing toward fraud (human-readable)
    """
    if model is None:
        model = load_model()
    if explainer is None:
        explainer = get_shap_explainer(model)

    # Prepare features
    X = prepare_single_transaction(transaction)

    # Predict
    prediction = int(model.predict(X)[0])
    fraud_prob = float(model.predict_proba(X)[0][1])

    # Risk level
    if fraud_prob < 0.3:
        risk_level = "LOW"
    elif fraud_prob < 0.7:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    # SHAP explanation
    shap_values = explainer.shap_values(X)
    base_value = float(explainer.expected_value)

    # Build per-feature explanation
    feature_names = X.columns.tolist()
    shap_dict = {
        name: round(float(val), 4)
        for name, val in zip(feature_names, shap_values[0])
    }

    # Top reasons pushing toward fraud (positive SHAP = pushes toward fraud)
    sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_reasons = []
    for feat_name, shap_val in sorted_features[:5]:
        direction = "increases" if shap_val > 0 else "decreases"
        feat_value = float(X[feat_name].iloc[0])
        top_reasons.append({
            "feature": feat_name,
            "value": round(feat_value, 2),
            "shap_value": shap_val,
            "direction": direction,
            "impact": f"{feat_name}={feat_value:.2f} {direction} fraud risk by {abs(shap_val):.4f}",
        })

    return {
        "prediction": prediction,
        "prediction_label": "FRAUD" if prediction == 1 else "LEGIT",
        "fraud_probability": round(fraud_prob, 4),
        "risk_level": risk_level,
        "base_value": base_value,
        "shap_values": shap_dict,
        "feature_values": {col: round(float(X[col].iloc[0]), 4) for col in feature_names},
        "top_reasons": top_reasons,
    }


def predict_batch(
    transactions: list[dict],
    model=None,
) -> list[dict]:
    """Predict and explain a batch of transactions."""
    if model is None:
        model = load_model()
    explainer = get_shap_explainer(model)

    results = []
    for txn in transactions:
        result = predict_with_explanation(txn, model=model, explainer=explainer)
        results.append(result)
    return results


def predict_batch_df(
    df: pd.DataFrame,
    model=None,
) -> pd.DataFrame:
    """
    Score an entire DataFrame of transactions.
    Returns the original data with prediction columns appended.
    Optimized for batch — computes SHAP for all rows at once.
    """
    if model is None:
        model = load_model()

    required_cols = ["step", "type", "amount", "oldbalanceOrg",
                     "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Filter to valid transaction types
    valid_mask = df["type"].isin(["TRANSFER", "CASH_OUT"])
    df_valid = df[valid_mask].copy()
    df_invalid = df[~valid_mask].copy()

    if len(df_valid) == 0:
        raise ValueError("No TRANSFER or CASH_OUT transactions found in the data.")

    # Feature engineering for all rows at once
    from src.features.engineering import create_transaction_features, encode_categorical
    X = create_transaction_features(df_valid[required_cols])
    X = encode_categorical(X)

    for col in TRAINING_FEATURES:
        if col not in X.columns:
            X[col] = 0
    X = X[TRAINING_FEATURES]

    # Batch predict
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    # Batch SHAP
    explainer = get_shap_explainer(model)
    shap_values = explainer.shap_values(X)

    # Find top risk feature for each row
    top_features = []
    for i in range(len(X)):
        feat_idx = int(np.argmax(np.abs(shap_values[i])))
        top_features.append(TRAINING_FEATURES[feat_idx])

    # Build result DataFrame
    df_valid["prediction"] = predictions
    df_valid["prediction_label"] = ["FRAUD" if p == 1 else "LEGIT" for p in predictions]
    df_valid["fraud_probability"] = [round(p, 4) for p in probabilities]
    df_valid["risk_level"] = [
        "HIGH" if p >= 0.7 else "MEDIUM" if p >= 0.3 else "LOW"
        for p in probabilities
    ]
    df_valid["top_risk_feature"] = top_features

    # Add invalid rows back with "SKIPPED" label
    if len(df_invalid) > 0:
        df_invalid["prediction"] = -1
        df_invalid["prediction_label"] = "SKIPPED"
        df_invalid["fraud_probability"] = None
        df_invalid["risk_level"] = "N/A"
        df_invalid["top_risk_feature"] = "N/A (invalid type)"
        df_valid = pd.concat([df_valid, df_invalid], ignore_index=True)

    return df_valid


# --- SHAP visualization helpers (used by Streamlit app) ---

def get_shap_waterfall_data(explanation_result: dict) -> dict:
    """
    Format SHAP data for waterfall plot rendering.
    Returns base_value, feature names, SHAP values, and feature values.
    """
    shap_vals = explanation_result["shap_values"]
    feat_vals = explanation_result["feature_values"]

    # Sort by absolute SHAP value
    sorted_feats = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)

    return {
        "base_value": explanation_result["base_value"],
        "features": [f[0] for f in sorted_feats],
        "shap_values": [f[1] for f in sorted_feats],
        "feature_values": [feat_vals[f[0]] for f in sorted_feats],
    }


if __name__ == "__main__":
    # Quick test with a sample transaction
    sample_txn = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 200000,
        "oldbalanceOrg": 200000,
        "newbalanceOrig": 0,
        "oldbalanceDest": 0,
        "newbalanceDest": 200000,
    }

    print("Testing prediction with explanation...")
    result = predict_with_explanation(sample_txn)
    print(f"\nPrediction: {result['prediction_label']}")
    print(f"Fraud Probability: {result['fraud_probability']}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"\nTop Reasons:")
    for reason in result["top_reasons"]:
        print(f"  → {reason['impact']}")