"""
FastAPI backend for FraudLens.
Serves fraud predictions with SHAP explanations via REST API.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from src.models.predict import predict_with_explanation, get_shap_explainer
from src.models.train import load_model


# --- Global model + explainer (loaded once at startup) ---
model = None
explainer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model, explainer
    print("Loading model...")
    model = load_model()
    explainer = get_shap_explainer(model)
    print("Model loaded and ready.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="FraudLens API",
    description="Credit card fraud detection with SHAP explainability",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Request / Response schemas ---

class TransactionRequest(BaseModel):
    """Schema for a single transaction to score."""
    step: int = Field(..., description="Hour of the simulation (1-744)", ge=1, le=744)
    type: str = Field(..., description="Transaction type: TRANSFER or CASH_OUT")
    amount: float = Field(..., description="Transaction amount", gt=0)
    oldbalanceOrg: float = Field(..., description="Origin balance before transaction", ge=0)
    newbalanceOrig: float = Field(..., description="Origin balance after transaction", ge=0)
    oldbalanceDest: float = Field(..., description="Destination balance before transaction", ge=0)
    newbalanceDest: float = Field(..., description="Destination balance after transaction", ge=0)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "step": 1,
                    "type": "TRANSFER",
                    "amount": 200000,
                    "oldbalanceOrg": 200000,
                    "newbalanceOrig": 0,
                    "oldbalanceDest": 0,
                    "newbalanceDest": 200000,
                }
            ]
        }
    }


class SHAPReason(BaseModel):
    feature: str
    value: float
    shap_value: float
    direction: str
    impact: str


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    fraud_probability: float
    risk_level: str
    top_reasons: list[SHAPReason]
    shap_values: dict[str, float]
    feature_values: dict[str, float]
    base_value: float


# --- Endpoints ---

@app.get("/")
async def root():
    return {
        "service": "FraudLens API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionRequest):
    """
    Score a transaction for fraud and return SHAP explanation.

    Returns prediction, probability, risk level, and top 5 features
    driving the decision with SHAP values.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if transaction.type not in ("TRANSFER", "CASH_OUT"):
        raise HTTPException(
            status_code=400,
            detail="Transaction type must be TRANSFER or CASH_OUT"
        )

    try:
        result = predict_with_explanation(
            transaction.model_dump(),
            model=model,
            explainer=explainer,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)