"""
Rolls up the cleaned transaction-level data into one row per customer,
with RFM (Recency, Frequency, Monetary) features plus extras. This is
the feature set every downstream model (PCA, t-SNE, K-Means, DBSCAN)
actually operates on — not the raw transactions.
"""

import pandas as pd
import numpy as np

FEATURE_COLUMNS = [
    "Recency",
    "Frequency",
    "Monetary",
    "AvgBasketValue",
    "AvgItemsPerBasket",
    "UniqueProducts",
    "CancellationRate",
    "CustomerLifespanDays",
]


def build_customer_features(df: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    """
    df: cleaned transaction-level dataframe (see preprocessing.clean_transactions)
    snapshot_date: the "as of" date Recency is measured from. Defaults to
    one day after the last transaction in the data, which is standard
    practice for RFM on a fixed historical dataset.
    """
    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    purchases = df[~df["IsCancellation"]]

    grouped = purchases.groupby("CustomerID").agg(
        LastPurchaseDate=("InvoiceDate", "max"),
        FirstPurchaseDate=("InvoiceDate", "min"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
        UniqueProducts=("StockCode", "nunique"),
        TotalItems=("Quantity", "sum"),
    )

    grouped["Recency"] = (snapshot_date - grouped["LastPurchaseDate"]).dt.days
    grouped["CustomerLifespanDays"] = (grouped["LastPurchaseDate"] - grouped["FirstPurchaseDate"]).dt.days
    grouped["AvgBasketValue"] = grouped["Monetary"] / grouped["Frequency"]
    grouped["AvgItemsPerBasket"] = grouped["TotalItems"] / grouped["Frequency"]

    # Cancellation rate needs the full (uncleaned-of-cancellations) transaction set per
    # customer. Work it out at the unique-invoice level, not the line-item level, or a
    # multi-item cancelled order gets double-counted against a denominator of unique
    # invoices and the "rate" can exceed 1.
    invoice_level = df[["CustomerID", "InvoiceNo"]].drop_duplicates()
    invoice_level["IsCancellation"] = invoice_level["InvoiceNo"].astype(str).str.startswith("C")

    cancellation_rate = invoice_level.groupby("CustomerID")["IsCancellation"].mean()
    grouped["CancellationRate"] = cancellation_rate

    grouped["CancellationRate"] = grouped["CancellationRate"].fillna(0)

    features = grouped[FEATURE_COLUMNS].copy()
    features = features.reset_index()

    return features
