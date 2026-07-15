import streamlit as st
import plotly.express as px

from utils import load_clustered_data

st.set_page_config(page_title="Anomaly Detection", page_icon="🚨", layout="wide")
st.title("🚨 Anomaly Detection (DBSCAN)")

df = load_clustered_data()

if "Cluster_DBSCAN" not in df.columns:
    st.warning("DBSCAN results not found. Run notebook 05_dbscan_anomaly.ipynb first.")
    st.stop()

df["Status"] = df["Cluster_DBSCAN"].apply(lambda x: "Anomaly (Noise)" if x == -1 else "Normal (Dense Cluster)")

n_anomalies = (df["Status"] == "Anomaly (Noise)").sum()
st.markdown(
    f"DBSCAN flagged **{n_anomalies}** out of **{len(df):,}** customers "
    f"(**{n_anomalies/len(df):.1%}**) as anomalies — customers whose combination "
    "of Recency, Frequency, Monetary, basket size, and cancellation behavior "
    "doesn't resemble any dense group of typical customers. These are good "
    "candidates for manual fraud/risk review."
)

fig = px.scatter(
    df,
    x="Recency",
    y="Monetary",
    color="Status",
    log_y=True,
    color_discrete_map={"Normal (Dense Cluster)": "#4C72B0", "Anomaly (Noise)": "#D62728"},
    hover_data=["CustomerID", "Frequency", "CancellationRate"],
    title="Normal customers vs flagged anomalies (log-scaled Monetary)",
    height=550,
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("What makes flagged customers different?")
compare_cols = ["Recency", "Frequency", "Monetary", "AvgBasketValue", "CancellationRate", "CustomerLifespanDays"]
compare_cols = [c for c in compare_cols if c in df.columns]
st.dataframe(df.groupby("Status")[compare_cols].mean().round(2), use_container_width=True)

st.markdown("---")
st.subheader("Flagged anomaly records")
record_cols = ["CustomerID"] + compare_cols
anomalies = df[df["Status"] == "Anomaly (Noise)"][record_cols].sort_values("Monetary", ascending=False)
st.dataframe(anomalies, use_container_width=True)

csv = anomalies.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download flagged anomalies as CSV",
    data=csv,
    file_name="flagged_anomalies.csv",
    mime="text/csv",
)
