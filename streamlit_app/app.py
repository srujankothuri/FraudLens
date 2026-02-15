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
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from src.models.train import load_model
from src.models.predict import (
    predict_with_explanation,
    predict_batch_df,
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

# --- Tabs for single vs batch ---
tab_single, tab_batch = st.tabs(["🔎 Single Transaction", "📁 Batch Upload"])

# ====================================================
# TAB 1: SINGLE TRANSACTION ANALYSIS
# ====================================================
with tab_single:
    # --- Sidebar: Transaction Input ---
    st.sidebar.header("📝 Transaction Details")
    st.sidebar.markdown("Enter transaction info below:")

    txn_type = st.sidebar.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT"])

    # User-friendly time input instead of raw "step"
    st.sidebar.markdown("**Transaction Time**")
    day = st.sidebar.number_input("Day of Month", min_value=1, max_value=30, value=1)
    hour = st.sidebar.slider("Hour of Day", min_value=0, max_value=23, value=2, format="%d:00")
    step = (day - 1) * 24 + hour  # Convert to step internally

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
        day, hour = 1, 2
        step = (day - 1) * 24 + hour
        amount = 350000
        oldbalanceOrg, newbalanceOrig = 350000, 0
        oldbalanceDest, newbalanceDest = 0, 350000
        analyze_btn = True

    if legit_btn:
        txn_type = "CASH_OUT"
        day, hour = 1, 12
        step = (day - 1) * 24 + hour
        amount = 500
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

        # --- Transaction Summary ---
        st.subheader("💳 Transaction Analyzed")
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Type", transaction["type"])
        t2.metric("Amount", f"${transaction['amount']:,.2f}")
        t3.metric("Time", f"Day {transaction['step'] // 24 + 1}, {transaction['step'] % 24}:00")
        t4.metric("Balance Change", f"${transaction['oldbalanceOrg']:,.0f} → ${transaction['newbalanceOrig']:,.0f}")

        st.divider()

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


# ====================================================
# TAB 2: BATCH UPLOAD
# ====================================================
with tab_batch:
    st.subheader("📁 Batch Transaction Scoring")
    st.markdown(
        "Upload a CSV file of transactions to score them all at once. "
        "Each transaction gets a fraud prediction, probability, risk level, and top risk feature."
    )

    # --- Expected format ---
    with st.expander("📋 Expected CSV Format"):
        st.markdown("Your CSV must have these columns:")
        sample_df = pd.DataFrame({
            "step": [1, 200, 50],
            "type": ["TRANSFER", "CASH_OUT", "TRANSFER"],
            "amount": [200000, 500, 80000],
            "oldbalanceOrg": [200000, 15000, 80000],
            "newbalanceOrig": [0, 14500, 0],
            "oldbalanceDest": [0, 80000, 50000],
            "newbalanceDest": [200000, 80500, 130000],
        })
        st.dataframe(sample_df, use_container_width=True)
        st.download_button(
            "⬇️ Download Sample CSV",
            sample_df.to_csv(index=False),
            file_name="sample_transactions.csv",
            mime="text/csv",
        )

    # --- File Upload ---
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.markdown(f"**Loaded {len(df):,} transactions**")

            # Preview raw data
            with st.expander("👀 Preview Uploaded Data"):
                st.dataframe(df.head(10), use_container_width=True)

            # Score button
            if st.button("🚀 Score All Transactions", type="primary", use_container_width=True):
                with st.spinner(f"Scoring {len(df):,} transactions..."):
                    results_df = predict_batch_df(df, model=model)

                # --- Summary Stats ---
                st.divider()
                st.subheader("📊 Batch Results Summary")

                scored = results_df[results_df["prediction_label"] != "SKIPPED"]
                fraud_count = (scored["prediction"] == 1).sum()
                legit_count = (scored["prediction"] == 0).sum()
                skipped_count = (results_df["prediction_label"] == "SKIPPED").sum()
                fraud_rate = fraud_count / len(scored) * 100 if len(scored) > 0 else 0

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Scored", f"{len(scored):,}")
                m2.metric("🚨 Fraud", f"{fraud_count:,}")
                m3.metric("✅ Legit", f"{legit_count:,}")
                m4.metric("Fraud Rate", f"{fraud_rate:.1f}%")

                if skipped_count > 0:
                    st.info(f"ℹ️ {skipped_count} transactions skipped (type must be TRANSFER or CASH_OUT)")

                st.divider()

                # --- Risk Distribution Chart ---
                st.subheader("📈 Risk Distribution")
                risk_counts = scored["risk_level"].value_counts()

                fig = go.Figure(go.Bar(
                    x=risk_counts.index,
                    y=risk_counts.values,
                    marker_color=["#2ecc71" if r == "LOW" else "#f39c12" if r == "MEDIUM" else "#e74c3c"
                                  for r in risk_counts.index],
                    text=risk_counts.values,
                    textposition="auto",
                ))
                fig.update_layout(
                    xaxis_title="Risk Level",
                    yaxis_title="Count",
                    height=350,
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

                # --- Top Flagged Transactions ---
                if fraud_count > 0:
                    st.subheader("🚨 Top Flagged Transactions")
                    flagged = scored[scored["prediction"] == 1].sort_values(
                        "fraud_probability", ascending=False
                    ).head(10)
                    display_cols = ["type", "amount", "fraud_probability", "risk_level", "top_risk_feature"]
                    st.dataframe(
                        flagged[display_cols].style.format({"amount": "${:,.2f}", "fraud_probability": "{:.1%}"}),
                        use_container_width=True,
                    )

                # --- Full Results Table ---
                st.subheader("📋 Full Results")
                display_cols = ["type", "amount", "prediction_label", "fraud_probability", "risk_level", "top_risk_feature"]
                available_cols = [c for c in display_cols if c in results_df.columns]
                st.dataframe(results_df[available_cols], use_container_width=True)

                # --- Download Results ---
                st.divider()
                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Scored Results (CSV)",
                    csv_data,
                    file_name="fraudlens_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except ValueError as e:
            st.error(f"❌ {str(e)}")
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")