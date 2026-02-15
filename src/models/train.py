"""
Model training pipeline for fraud detection.
Trains XGBoost with SMOTE oversampling for class imbalance.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.data.preprocess import load_data, clean_data, split_data
from src.features.engineering import run_feature_pipeline


MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "xgb_fraud_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


def build_model(random_state: int = 42) -> ImbPipeline:
    """
    Build an imbalanced-learn pipeline with SMOTE + XGBoost.

    Why SMOTE? The dataset is ~99.8% legit, ~0.2% fraud.
    Training directly would make the model predict "legit" for everything.
    SMOTE generates synthetic fraud samples to balance the classes.

    Why XGBoost? Industry standard for tabular fraud detection —
    handles non-linear patterns, fast, and works well with SMOTE.
    """
    model = ImbPipeline([
        ("smote", SMOTE(random_state=random_state, sampling_strategy=0.5)),
        ("classifier", XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=1,  # SMOTE handles imbalance, so keep at 1
            eval_metric="aucpr",  # PR-AUC better than ROC-AUC for imbalanced data
            random_state=random_state,
            n_jobs=-1,
            # use_label_encoder deprecated in newer XGBoost versions
        )),
    ])
    return model


def evaluate_model(
    model: ImbPipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate model with metrics suited for imbalanced classification.

    Key insight: Accuracy is misleading here (99.8% by predicting all legit).
    We focus on PR-AUC, precision, recall, and F1 for the fraud class.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # PR-AUC (most important metric for imbalanced fraud detection)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall_vals, precision_vals)

    # ROC-AUC
    roc = roc_auc_score(y_test, y_prob)

    # Classification report for fraud class (label=1)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc, 4),
        "fraud_precision": round(report["1"]["precision"], 4),
        "fraud_recall": round(report["1"]["recall"], 4),
        "fraud_f1": round(report["1"]["f1-score"], 4),
        "confusion_matrix": cm.tolist(),
        "total_test_samples": len(y_test),
        "total_fraud_samples": int(y_test.sum()),
    }

    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    print(f"PR-AUC:          {metrics['pr_auc']}")
    print(f"ROC-AUC:         {metrics['roc_auc']}")
    print(f"Fraud Precision: {metrics['fraud_precision']}")
    print(f"Fraud Recall:    {metrics['fraud_recall']}")
    print(f"Fraud F1:        {metrics['fraud_f1']}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0][0]:,}  FP={cm[0][1]:,}")
    print(f"  FN={cm[1][0]:,}  TP={cm[1][1]:,}")

    return metrics


def save_model(model: ImbPipeline, metrics: dict) -> None:
    """Save trained model and metrics to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")


def load_model(path: str | Path = MODEL_PATH) -> ImbPipeline:
    """Load a trained model from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run 'python -m src.models.train' first."
        )
    return joblib.load(path)


def train(data_path: str | Path = None) -> tuple[ImbPipeline, dict]:
    """Execute the full training pipeline."""
    print("=" * 50)
    print("TRAINING PIPELINE")
    print("=" * 50)

    # 1. Load and preprocess
    if data_path:
        df = load_data(data_path)
    else:
        df = load_data()
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # 2. Feature engineering
    X_train, X_test = run_feature_pipeline(X_train, X_test)

    # 3. Train
    print("\nTraining XGBoost with SMOTE...")
    model = build_model()
    model.fit(X_train, y_train)
    print("Training complete.")

    # 4. Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # 5. Save
    save_model(model, metrics)

    return model, metrics


if __name__ == "__main__":
    train()