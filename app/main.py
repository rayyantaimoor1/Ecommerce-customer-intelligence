"""
FastAPI service that serves the trained K-Means customer segmentation
model, built from RFM (Recency/Frequency/Monetary) + extra behavioral
features. Run from the `app/` directory with:

    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""

from fastapi import FastAPI, HTTPException

from schemas import CustomerInput, ClusterPrediction, HealthResponse
from model_utils import predict_cluster, load_bundle

app = FastAPI(
    title="E-Commerce Customer Segmentation API",
    description=(
        "Predicts a customer's segment/persona from RFM (Recency, Frequency, Monetary) "
        "and related behavioral metrics, using a K-Means model trained on the Online "
        "Retail transaction dataset."
    ),
    version="1.0.0",
)


@app.get("/", response_model=HealthResponse)
def root():
    """Health check — also confirms the model file loaded successfully."""
    try:
        load_bundle()
        return {"status": "ok", "model_loaded": True}
    except FileNotFoundError:
        return {"status": "model not found", "model_loaded": False}


@app.post("/predict-cluster", response_model=ClusterPrediction)
def predict(customer: CustomerInput):
    """
    Accepts a customer's RFM + behavioral metrics and returns the
    predicted segment (cluster) and persona label, e.g.
    "Champions (Recent, Frequent, High-Spend)".

    Note: these are customer-level aggregate features (computed by
    rolling up that customer's transaction history), not raw
    transaction rows — see notebook 02 for how to compute them from
    an invoice-level dataset.
    """
    try:
        result = predict_cluster(
            recency=customer.recency,
            frequency=customer.frequency,
            monetary=customer.monetary,
            avg_basket_value=customer.avg_basket_value,
            avg_items_per_basket=customer.avg_items_per_basket,
            unique_products=customer.unique_products,
            cancellation_rate=customer.cancellation_rate,
            customer_lifespan_days=customer.customer_lifespan_days,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return {
        "cluster_id": result["cluster_id"],
        "persona": result["persona"],
        "input_echo": customer,
    }
