import streamlit as st
import plotly.express as px

from utils import load_clustered_data

st.set_page_config(page_title="Customer Segments", page_icon="👥", layout="wide")
st.title("👥 Customer Segments (K-Means)")

df = load_clustered_data()
df["Cluster"] = df["Cluster"].astype(str)

st.markdown(
    "Each point is a customer, colored by the persona K-Means assigned it, "
    "on a **log-scaled Monetary axis** (retail spend is heavily right-skewed, "
    "so a linear axis would compress most customers into a corner). "
    "Hover to see individual customer details."
)

group_col = "Persona" if "Persona" in df.columns else "Cluster"

fig = px.scatter(
    df,
    x="Recency",
    y="Monetary",
    color=group_col,
    log_y=True,
    hover_data=["CustomerID", "Frequency", "AvgBasketValue", "CancellationRate"],
    title="Customer Segments — Recency vs Monetary (log scale)",
    height=550,
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Segment profiles")

profile_cols = ["Recency", "Frequency", "Monetary", "AvgBasketValue", "CancellationRate", "CustomerLifespanDays"]
profile_cols = [c for c in profile_cols if c in df.columns]

profile = (
    df.groupby(group_col)[profile_cols]
    .mean()
    .round(2)
)
profile["Customer Count"] = df[group_col].value_counts()
st.dataframe(profile, use_container_width=True)

st.markdown("---")
st.subheader("Segment sizes")
size_fig = px.bar(
    df[group_col].value_counts().reset_index(),
    x=group_col,
    y="count",
    title="Number of customers per segment",
)
st.plotly_chart(size_fig, use_container_width=True)
