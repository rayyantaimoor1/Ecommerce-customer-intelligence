import streamlit as st
from utils import load_clustered_data

st.set_page_config(
    page_title="E-Commerce Customer Intelligence",
    page_icon="🛍️",
    layout="wide",
)

st.title("🛍️ E-Commerce Customer Intelligence Dashboard")

st.markdown(
    """
This dashboard summarizes an unsupervised learning pipeline built on the
**Online Retail** transaction dataset (~540K line items, ~4,300 customers)
to answer:

> **How can an e-commerce business identify distinct customer personas and
> flag anomalous transaction patterns, to optimize targeted marketing and
> reduce financial risk?**

Each customer's full transaction history is rolled up into RFM
(Recency, Frequency, Monetary) features plus basket size, product
variety, cancellation rate, and tenure. Two models run on that
customer-level feature set:

- **K-Means clustering** → groups customers into marketing personas
  (e.g. *Champions*, *At-Risk*, *High Cancellation Rate*).
- **DBSCAN** → flags customers whose behavior doesn't fit any dense
  group, as candidate anomaly / fraud-risk signals.

Use the sidebar to navigate:
- **Customer Segments** — explore the personas found by K-Means
- **Anomaly Detection** — inspect customers flagged by DBSCAN
- **Predict New Customer** — try the live model on a hypothetical customer
"""
)

df = load_clustered_data()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total customers", f"{len(df):,}")
col2.metric("Segments found (K-Means)", df["Cluster"].nunique())
if "Cluster_DBSCAN" in df.columns:
    n_anomalies = int((df["Cluster_DBSCAN"] == -1).sum())
    col3.metric("Flagged anomalies", n_anomalies)
    col4.metric("Anomaly rate", f"{n_anomalies/len(df):.1%}")

st.markdown("---")
st.subheader("Customer feature table (sample)")
display_cols = ["CustomerID", "Recency", "Frequency", "Monetary", "AvgBasketValue",
                 "CancellationRate", "Persona"]
display_cols = [c for c in display_cols if c in df.columns]
st.dataframe(df[display_cols].head(20), use_container_width=True)
