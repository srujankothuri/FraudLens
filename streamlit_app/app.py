"""
FraudLens — Streamlit Dashboard
Interactive fraud detection with SHAP explainability.
"""

import sys
from pathlib import Path

# Add project root to path so we can import src modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import shap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from src.models.train import load_model
from src.models.predict import (
    predict_with_explanation,
    get_shap_explainer,
    prepare_single_transaction,
    TRAINING_FEATURES,
)

# --- Page Config ---
st.set_page_config(
    page_title="FraudLens",
    page_icon="🔍",
    layout="wide",
)

# --- Load Model (cached so it only loads once) ---
@st.cache_resource
def init_model():
    model = load_model()
    explainer = get_shap_explainer(model)
    return model, explainer

model, explainer = init_model()


# --- Header ---
st.title("🔍 FraudLens")
st.markdown("**Explainable Credit Card Fraud Detection** — See *why* a transaction is flagged, not just *if*.")
st.divider()

# --- Sidebar: Transaction Input ---
st.sidebar.header("📝 Transaction Details")
st.sidebar.markdown("Enter transaction info below:")

txn_type = st.sidebar.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT"])
step = st.sidebar.slider("Hour (step)", min_value=1, max_value=744, value=1, help="Hour of the month (1-744)")
amount = st.sidebar.number_input("Amount ($)", min_value=0.01, value=200000.0, step=1000.0)
oldbalanceOrg = st.sidebar.number_input("Origin Balance (Before)", min_value=0.0, value=200000.0, step=1000.0)
newbalanceOrig = st.sidebar.number_input("Origin Balance (After)", min_value=0.0, value=0.0, step=1000.0)
oldbalanceDest = st.sidebar.number_input("Destination Balance (Before)", min_value=0.0, value=0.0, step=1000.0)
newbalanceDest = st.sidebar.number_input("Destination Balance (After)", min_value=0.0, value=200000.0, step=1000.0)

analyze_btn = st.sidebar.button("🔍 Analyze Transaction", use_container_width=True, type="primary")

# --- Preset Examples ---
st.sidebar.divider()
st.sidebar.markdown("**Quick Examples:**")

col_s1, col_s2 = st.sidebar.columns(2)
suspicious_btn = col_s1.button("⚠️ Suspicious", use_container_width=True)
legit_btn = col_s2.button("✅ Legit", use_container_width=True)

if suspicious_btn:
    txn_type = "TRANSFER"
    step, amount = 2, 350000
    oldbalanceOrg, newbalanceOrig = 350000, 0
    oldbalanceDest, newbalanceDest = 0, 350000
    analyze_btn = True

if legit_btn:
    txn_type = "CASH_OUT"
    step, amount = 12, 500
    oldbalanceOrg, newbalanceOrig = 10000, 9500
    oldbalanceDest, newbalanceDest = 50000, 50500
    analyze_btn = True


# --- Main Content ---
if analyze_btn:
    transaction = {
        "step": step,
        "type": txn_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
    }

    with st.spinner("Analyzing transaction..."):
        result = predict_with_explanation(transaction, model=model, explainer=explainer)

    # --- Result Banner ---
    if result["risk_level"] == "HIGH":
        st.error(f"🚨 **{result['prediction_label']}** — Fraud Probability: **{result['fraud_probability']:.1%}**")
    elif result["risk_level"] == "MEDIUM":
        st.warning(f"⚠️ **{result['prediction_label']}** — Fraud Probability: **{result['fraud_probability']:.1%}**")
    else:
        st.success(f"✅ **{result['prediction_label']}** — Fraud Probability: **{result['fraud_probability']:.1%}**")

    # --- Metrics Row ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Prediction", result["prediction_label"])
    col2.metric("Fraud Probability", f"{result['fraud_probability']:.1%}")
    col3.metric("Risk Level", result["risk_level"])

    st.divider()

    # --- Two-column layout: SHAP Waterfall + Top Reasons ---
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.subheader("📊 SHAP Waterfall Plot")
        st.caption("Each bar shows how a feature pushes the prediction toward FRAUD (red) or LEGIT (blue).")

        # Build SHAP Explanation object for waterfall plot
        X = prepare_single_transaction(transaction)
        shap_values = explainer.shap_values(X)

        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X.values[0],
            feature_names=X.columns.tolist(),
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        shap.plots.waterfall(explanation, max_display=12, show=False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with right_col:
        st.subheader("🔑 Top Risk Factors")
        for i, reason in enumerate(result["top_reasons"], 1):
            icon = "🔴" if reason["shap_value"] > 0 else "🔵"
            st.markdown(f"{icon} **{reason['feature']}** = `{reason['value']}`")
            direction = "↑ Increases" if reason["shap_value"] > 0 else "↓ Decreases"
            st.caption(f"   {direction} fraud risk by {abs(reason['shap_value']):.4f}")

    st.divider()

    # --- SHAP Bar Chart (interactive with Plotly) ---
    st.subheader("📈 Feature Impact (Interactive)")

    shap_df = pd.DataFrame({
        "Feature": list(result["shap_values"].keys()),
        "SHAP Value": list(result["shap_values"].values()),
    }).sort_values("SHAP Value", key=abs, ascending=True)

    colors = ["#e74c3c" if v > 0 else "#3498db" for v in shap_df["SHAP Value"]]

    fig = go.Figure(go.Bar(
        x=shap_df["SHAP Value"],
        y=shap_df["Feature"],
        orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>SHAP Value: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="SHAP Value (→ fraud | ← legit)",
        yaxis_title="",
        height=450,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Raw Data (collapsible) ---
    with st.expander("🔧 Raw Prediction Data"):
        st.json(result)

else:
    # --- Landing state ---
    st.markdown("""
    ### How it works
    1. **Enter** a transaction's details in the sidebar
    2. **Click** "Analyze Transaction"
    3. **See** the fraud prediction along with a full SHAP explanation of *why*

    Or try the **Quick Examples** to see a suspicious vs legitimate transaction.

    ---

    #### Why Explainability Matters
    In financial services, it's not enough to flag a transaction as fraud.
    Regulators and analysts need to understand **why** the model made that decision.
    FraudLens uses **SHAP (SHapley Additive exPlanations)** to provide transparent,
    per-feature explanations for every single prediction.
    """)