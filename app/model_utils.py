"""
Loads the trained clustering pipeline bundle once at startup and exposes
a simple predict function. Applies the same log1p transform used at
training time (see notebook 03) before scaling — this is recorded in
the bundle's "log_columns" so this code doesn't have to hardcode it.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "clustering_pipeline.joblib"

_bundle = None

# Maps the API's snake_case field names to the internal RFM column names
# the model was trained on.
FIELD_TO_COLUMN = {
    "recency": "Recency",
    "frequency": "Frequency",
    "monetary": "Monetary",
    "avg_basket_value": "AvgBasketValue",
    "avg_items_per_basket": "AvgItemsPerBasket",
    "unique_products": "UniqueProducts",
    "cancellation_rate": "CancellationRate",
    "customer_lifespan_days": "CustomerLifespanDays",
}


def load_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                f"Run the notebooks (01 through 06) first to train and export the pipeline."
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict_cluster(**fields) -> dict:
    """
    fields: recency, frequency, monetary, avg_basket_value,
    avg_items_per_basket, unique_products, cancellation_rate,
    customer_lifespan_days (all raw, un-transformed values).
    """
    bundle = load_bundle()

    row = {FIELD_TO_COLUMN[k]: v for k, v in fields.items()}
    df = pd.DataFrame([row])[bundle["feature_columns"]]

    for col in bundle.get("log_columns", []):
        df[col] = np.log1p(df[col])

    scaled = bundle["scaler"].transform(df)
    cluster_id = int(bundle["kmeans_model"].predict(scaled)[0])
    persona = bundle["persona_labels"].get(cluster_id, "Unknown")

    return {"cluster_id": cluster_id, "persona": persona}
