"""
Shared loading/plotting helpers for the Streamlit dashboard pages.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "customer_features_clustered.csv"
PIPELINE_PATH = ROOT / "models" / "clustering_pipeline.joblib"


@st.cache_data
def load_clustered_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(
            f"Processed data not found at `{DATA_PATH}`. "
            f"Run notebooks 01 through 05 first to generate it."
        )
        st.stop()
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_pipeline_bundle():
    if not PIPELINE_PATH.exists():
        st.error(
            f"Model pipeline not found at `{PIPELINE_PATH}`. "
            f"Run notebook 06_model_export.ipynb first."
        )
        st.stop()
    return joblib.load(PIPELINE_PATH)


def predict_new_customer(recency, frequency, monetary, avg_basket_value,
                          avg_items_per_basket, unique_products,
                          cancellation_rate, customer_lifespan_days):
    bundle = load_pipeline_bundle()

    row = {
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary,
        "AvgBasketValue": avg_basket_value,
        "AvgItemsPerBasket": avg_items_per_basket,
        "UniqueProducts": unique_products,
        "CancellationRate": cancellation_rate,
        "CustomerLifespanDays": customer_lifespan_days,
    }
    df = pd.DataFrame([row])[bundle["feature_columns"]]

    for col in bundle.get("log_columns", []):
        df[col] = np.log1p(df[col])

    scaled = bundle["scaler"].transform(df)
    cluster_id = int(bundle["kmeans_model"].predict(scaled)[0])
    persona = bundle["persona_labels"].get(cluster_id, "Unknown")
    return cluster_id, persona
