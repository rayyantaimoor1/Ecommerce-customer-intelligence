"""
Clustering utilities: fitting K-Means / Hierarchical / DBSCAN over the
RFM+extras feature set, choosing K, and producing human-readable
persona labels from cluster centroids.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def elbow_and_silhouette(X_scaled, k_range=range(2, 11)):
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        rows.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X_scaled, labels),
        })
    return pd.DataFrame(rows)


def fit_kmeans(X_scaled, n_clusters: int):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


def fit_hierarchical(X_scaled, n_clusters: int):
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = hc.fit_predict(X_scaled)
    return hc, labels


def k_distance_plot_data(X_scaled, min_samples: int = 5):
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(X_scaled)
    distances, _ = neighbors_fit.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])
    return k_distances


def fit_dbscan(X_scaled, eps: float, min_samples: int = 5):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X_scaled)
    return db, labels


def label_personas(df: pd.DataFrame, cluster_col: str = "Cluster") -> dict:
    """
    Given a dataframe with RFM features + a cluster label column, rank
    clusters by their mean Recency/Frequency/Monetary/CancellationRate
    and assign a descriptive persona name per cluster. Returns
    {cluster_id: persona_name}. Cluster -1 (DBSCAN noise) is labeled as
    a potential anomaly.

    Ranking clusters (rather than a single global median split) avoids
    the failure mode where several distinct clusters all land on the
    same side of a coarse threshold and collapse into one label.
    """
    personas = {}
    real_clusters = [c for c in df[cluster_col].unique() if c != -1]

    grouped = df[df[cluster_col].isin(real_clusters)].groupby(cluster_col)[
        ["Recency", "Frequency", "Monetary", "CancellationRate"]
    ].mean()

    # Rank each cluster on each dimension (1 = best/most-desirable rank)
    recency_rank = grouped["Recency"].rank(method="first")               # lower recency = more recent = better
    frequency_rank = grouped["Frequency"].rank(method="first", ascending=False)
    monetary_rank = grouped["Monetary"].rank(method="first", ascending=False)
    cancellation_rank = grouped["CancellationRate"].rank(method="first", ascending=False)  # higher = worse

    n = len(grouped)
    high_cancel_threshold = grouped["CancellationRate"].quantile(0.75) if n > 1 else grouped["CancellationRate"].max()

    for cluster_id in grouped.index:
        # Flag clusters with a notably elevated cancellation rate first — this is a
        # distinct risk signal that shouldn't be masked by otherwise-average RFM values.
        if n > 2 and grouped.loc[cluster_id, "CancellationRate"] >= high_cancel_threshold and grouped.loc[cluster_id, "CancellationRate"] > 0.05:
            personas[cluster_id] = "High Cancellation Rate (Risk Watch)"
            continue

        r_rank, f_rank, m_rank = recency_rank[cluster_id], frequency_rank[cluster_id], monetary_rank[cluster_id]
        top_third = max(1, round(n / 3))
        bottom_third = n - top_third + 1

        is_top_value = (f_rank <= top_third) and (m_rank <= top_third)
        is_recent = r_rank <= top_third
        is_stale = r_rank >= bottom_third
        is_low_engagement = (f_rank >= bottom_third) and (m_rank >= bottom_third)

        if is_top_value and is_recent:
            name = "Champions (Recent, Frequent, High-Spend)"
        elif is_top_value:
            name = "Loyal High-Value Customers"
        elif is_stale and is_low_engagement:
            name = "At-Risk / Churned Customers"
        elif is_recent and is_low_engagement:
            name = "New / Low-Engagement Customers"
        elif is_low_engagement:
            name = "Occasional Low-Spend Customers"
        else:
            name = "Regular Customers"

        personas[cluster_id] = name

    personas[-1] = "Anomaly / Potential Fraud (Noise)"
    return {k: v for k, v in personas.items() if k in df[cluster_col].unique()}
