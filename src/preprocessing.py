"""
Cleaning logic for the raw Online Retail transaction log. Shared between
notebooks and (indirectly, via the exported RFM feature list) the serving
apps.
"""

import pandas as pd

RAW_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country",
]


def load_raw_transactions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="ISO-8859-1", dtype={"CustomerID": "str"})
    # CustomerID often loads as float-like string ("12346.0") depending on source export;
    # normalize it to a clean string ID or NaN.
    df["CustomerID"] = df["CustomerID"].apply(
        lambda x: str(int(float(x))) if pd.notna(x) and str(x).strip() not in ("", "nan") else None
    )
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standard cleaning for this dataset:
    - Drop rows with no CustomerID (can't attribute them to a customer for segmentation)
    - Drop rows with missing Description
    - Drop non-product stock codes (postage, discounts, manual adjustments, bank charges)
    - Keep cancellations (InvoiceNo starting with 'C') as their own flag rather than
      dropping them outright — they're useful signal for the anomaly-detection side
    - Drop rows with UnitPrice <= 0 (data entry glitches, not real transactions)
    """
    df = df.copy()

    df = df.dropna(subset=["CustomerID", "Description"])

    non_product_codes = {"POST", "D", "M", "BANK CHARGES", "DOT", "CRUK"}
    df = df[~df["StockCode"].isin(non_product_codes)]

    df["IsCancellation"] = df["InvoiceNo"].astype(str).str.startswith("C")

    df = df[df["UnitPrice"] > 0]

    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    return df.reset_index(drop=True)
