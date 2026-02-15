# 🔍 FraudLens
![CI](https://github.com/srujankothuri/FraudLens/actions/workflows/ci.yml/badge.svg)
**Explainable Credit Card Fraud Detection System**

A production-ready fraud detection pipeline that doesn't just predict fraud — it **explains WHY** each transaction is suspicious using interactive SHAP visualizations.

> 🚀 **[Live Demo](https://fraudlens-srujankothuri.streamlit.app/)**

---

## Features

- **XGBoost Classification** with SMOTE for handling extreme class imbalance (99.8% legit vs 0.2% fraud)
- **SHAP Explainability** — interactive waterfall and force plots showing exactly which features drive each prediction
- **FastAPI Backend** — REST API for real-time transaction scoring
- **Streamlit Dashboard** — interactive frontend with visual explanations
- **Docker Support** — containerized for reproducible deployments
- **Tested & CI/CD** — pytest suite with GitHub Actions

## Tech Stack

| Layer | Tool |
|-------|------|
| ML Model | XGBoost |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| Explainability | SHAP |
| Backend | FastAPI |
| Frontend | Streamlit |
| Testing | pytest |
| CI/CD | GitHub Actions |
| Container | Docker |
| Dataset | [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) |

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/fraudlens.git
cd fraudlens

# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download dataset from Kaggle and place in data/raw/transactions.csv

# Run preprocessing
python -m src.data.preprocess

# Train model
python -m src.models.train

# Start API
uvicorn src.api.app:app --reload

# Start Streamlit (separate terminal)
streamlit run streamlit_app/app.py
```

## Project Structure

```
fraudlens/
├── src/
│   ├── data/preprocess.py          # Data loading & cleaning
│   ├── features/engineering.py     # Feature engineering
│   ├── models/train.py             # Model training pipeline
│   ├── models/predict.py           # Prediction + SHAP explanations
│   └── api/app.py                  # FastAPI endpoints
├── streamlit_app/app.py            # Interactive dashboard
├── notebooks/                      # EDA & analysis
├── tests/                          # pytest suite
├── models/                         # Saved model artifacts
├── Dockerfile & docker-compose.yml
└── .github/workflows/ci.yml        # CI pipeline
```

## License

MIT