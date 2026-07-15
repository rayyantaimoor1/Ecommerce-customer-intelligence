import streamlit as st
import plotly.express as px

from utils import load_clustered_data, predict_new_customer

st.set_page_config(page_title="Predict New Customer", page_icon="🔮", layout="wide")
st.title("🔮 Predict a New Customer's Segment")

st.markdown(
    "Enter a customer's RFM-style behavioral metrics below — typically "
    "computed by rolling up their transaction history (see notebook 02). "
    "This calls the exact same trained pipeline "
    "(`models/clustering_pipeline.joblib`) used by the FastAPI "
    "`/predict-cluster` endpoint."
)

col1, col2, col3, col4 = st.columns(4)
recency = col1.number_input("Recency (days since last order)", min_value=0, value=12)
frequency = col2.number_input("Frequency (# orders)", min_value=0, value=8)
monetary = col3.number_input("Monetary (total spend)", min_value=0.0, value=650.0)
avg_basket = col4.number_input("Avg Basket Value", min_value=0.0, value=81.25)

col5, col6, col7, col8 = st.columns(4)
avg_items = col5.number_input("Avg Items per Basket", min_value=0.0, value=6.5)
unique_products = col6.number_input("Unique Products Bought", min_value=0, value=22)
cancellation_rate = col7.slider("Cancellation Rate", 0.0, 1.0, 0.05)
lifespan = col8.number_input("Customer Lifespan (days)", min_value=0, value=210)

if st.button("Predict Segment", type="primary"):
    cluster_id, persona = predict_new_customer(
        recency, frequency, monetary, avg_basket,
        avg_items, unique_products, cancellation_rate, lifespan
    )

    st.success(f"**Predicted Segment:** Cluster {cluster_id} — {persona}")

    df = load_clustered_data()
    df["Cluster"] = df["Cluster"].astype(str)
    group_col = "Persona" if "Persona" in df.columns else "Cluster"

    fig = px.scatter(
        df,
        x="Recency",
        y="Monetary",
        color=group_col,
        log_y=True,
        opacity=0.4,
        title="New customer plotted against existing segments (log-scaled Monetary)",
        height=550,
    )
    fig.add_scatter(
        x=[recency],
        y=[max(monetary, 0.01)],  # avoid log(0) issues on the log axis
        mode="markers",
        marker=dict(size=20, color="black", symbol="star"),
        name="New Customer",
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption(
    "Equivalent API call: `POST /predict-cluster` with a JSON body containing "
    "recency, frequency, monetary, avg_basket_value, avg_items_per_basket, "
    "unique_products, cancellation_rate, and customer_lifespan_days."
)
