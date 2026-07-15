# E-Commerce Customer Intelligence: Segmentation & Anomaly Detection

**How can an e-commerce business identify distinct customer personas and flag
anomalous transaction patterns, to optimize targeted marketing and reduce
financial risk?**

Built on the **UCI/Kaggle Online Retail dataset** — ~540K real transaction
line items from a UK-based online gift retailer (Dec 2010 – Dec 2011),
rolled up into per-customer RFM (Recency, Frequency, Monetary) features
plus basket size, product variety, cancellation rate, and tenure.

1. **Customer Segmentation** — K-Means and Hierarchical Clustering group
   customers into marketing personas (e.g. *"Champions"*, *"At-Risk"*).
2. **Anomaly Detection** — DBSCAN flags customers whose behavior doesn't
   fit any dense cluster, as candidate fraud/risk signals.

Served two ways:
- A **FastAPI** REST endpoint (`/predict-cluster`) for system-to-system use
- A **Streamlit** dashboard for interactive, human-facing exploration

## Publishing this repo to GitHub

If you're starting from this folder locally (not already a git repo):

```bash
cd ecommerce-customer-intelligence
git init
git add .
git commit -m "Initial commit: customer segmentation + anomaly detection pipeline"
git branch -M main
git remote add origin https://github.com/rayyantaimoor1/Ecommerce-customer-intelligence.git
git push -u origin main
```

Create the empty repo on GitHub first (**github.com → New repository**,
don't initialize it with a README or .gitignore — this project already has
both, and a pre-populated remote will conflict with your first push).

GitHub requires a **Personal Access Token** instead of your account
password for `git push` over HTTPS. If prompted for credentials and your
normal password fails, generate one at **GitHub → Settings → Developer
settings → Personal access tokens → Tokens (classic)** with `repo` scope,
and use that token as the password.

`data/raw/OnlineRetail.csv` (~34MB) is committed directly rather than
gitignored, so a fresh clone (e.g. into Colab) works immediately without a
separate upload step — see **Dataset** below.

**Also worth knowing:** the trained model artifacts in `models/` (including
`clustering_pipeline.joblib`) are committed too. That means after cloning,
you can jump straight to running the FastAPI/Streamlit apps without
re-running the notebooks first — useful for a quick demo. Re-run the
notebooks only if you want to retrain on different data or tune the
clustering parameters yourself.

## Project Structure

```
ecommerce-customer-intelligence/
├── data/
│   ├── raw/                  # OnlineRetail.csv (committed to git — see Dataset)
│   └── processed/            # cleaned transactions, RFM features, cluster assignments
├── notebooks/                # cleaning → RFM engineering → PCA/t-SNE → clustering → export
├── colab/                    # Colab-specific runner notebook (see "Running on Google Colab")
├── app/                      # FastAPI service
├── dashboard/                # Streamlit multi-page dashboard
├── models/                   # saved joblib artifacts (scaler, models, pipeline bundle)
├── src/                      # shared preprocessing/feature/clustering code
├── reports/figures/          # saved plots from the notebooks
├── run_dashboard.bat         # Windows: double-click to launch the Streamlit dashboard
├── run_api.bat               # Windows: double-click to launch the FastAPI server
└── requirements.txt
```

## Dataset

`data/raw/OnlineRetail.csv` is committed directly to this repo (~34MB) so
a fresh `git clone` works immediately, with no separate download step.
Columns: `InvoiceNo`, `StockCode`, `Description`, `Quantity`,
`InvoiceDate`, `UnitPrice`, `CustomerID`, `Country` — matching the
standard UCI Online Retail / Kaggle "Online Retail Data Set from UCI ML
Repo" schema.

Want to use the real dataset instead of the bundled one? Download it from
[Kaggle](https://www.kaggle.com/datasets/vijayuv/onlineretail) or the
[UCI ML Repository](https://archive.ics.uci.edu/dataset/352/online+retail)
and overwrite `data/raw/OnlineRetail.csv` — same columns, so nothing else
in the pipeline needs to change.

**Known real-world messiness this pipeline handles:** ~25% of rows have no
CustomerID (dropped — can't attribute to a customer), a small share of
cancelled orders (`InvoiceNo` starting with `C`, kept and used as a
cancellation-rate feature rather than discarded), and a handful of
non-product stock codes (postage, discounts, manual adjustments — dropped).

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/rayyantaimoor1/Ecommerce-customer-intelligence.git
cd ecommerce-customer-intelligence

# 2. Create and activate an environment (conda example)
conda create -n ecomm-clustering python=3.11
conda activate ecomm-clustering

# 3. Install dependencies
pip install -r requirements.txt
```

The dataset is already included at `data/raw/OnlineRetail.csv` — no
separate download needed unless you want to swap in the real Kaggle/UCI
file (see **Dataset** above).

## Running the project

### 1. Run the notebooks (in order)

```bash
jupyter notebook notebooks/
```

Run in order:
1. `01_eda.ipynb` — load, clean, explore the raw transaction log
2. `02_rfm_feature_engineering.ipynb` — roll transactions up into one row per customer
3. `03_pca_tsne.ipynb` — scale, log-transform skewed features, visualize with PCA/t-SNE
4. `04_kmeans_hierarchical.ipynb` — Elbow + Silhouette to pick K, fit K-Means, dendrogram
5. `05_dbscan_anomaly.ipynb` — k-distance plot to tune `eps`, fit DBSCAN, inspect anomalies
6. `06_model_export.ipynb` — bundle scaler + model + persona labels into one joblib file

Each notebook reads the previous notebook's saved output from
`data/processed/` or `models/`. The final notebook exports
`models/clustering_pipeline.joblib`, which both apps below depend on.

### 2. Run the FastAPI service

```bash
cd app
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000/docs** for the interactive Swagger UI, or call
it directly:

```bash
curl -X POST http://127.0.0.1:8000/predict-cluster \
  -H "Content-Type: application/json" \
  -d '{
    "recency": 12, "frequency": 8, "monetary": 650.0,
    "avg_basket_value": 81.25, "avg_items_per_basket": 6.5,
    "unique_products": 22, "cancellation_rate": 0.05,
    "customer_lifespan_days": 210
  }'
```

### 3. Run the Streamlit dashboard

In a separate terminal:

```bash
cd dashboard
streamlit run Home.py
```

Opens automatically at **http://localhost:8501**. Pages:
- **Customer Segments** — interactive scatter of K-Means personas
- **Anomaly Detection** — DBSCAN-flagged outliers with CSV export
- **Predict New Customer** — live form that calls the trained pipeline

### 4. One-click launch on Windows (after the notebooks have run once)

Two `.bat` files sit in the project root for double-click launching, once
`models/clustering_pipeline.joblib` exists (i.e. after you've run the
notebooks at least once):

- **`run_dashboard.bat`** — activates the `ecomm-clustering` conda
  environment and launches the Streamlit dashboard, opening it in your
  default browser at `http://localhost:8501`.
- **`run_api.bat`** — same, but launches the FastAPI server at
  `http://127.0.0.1:8000` (Swagger docs at `/docs`).

Both check that the conda environment and trained model exist first and
print a clear error with next steps if not, rather than failing silently.
Requires the `ecomm-clustering` conda environment to already exist (see
**Setup** above) and `conda` to be on your system PATH — if double-clicking
opens and immediately closes a window, right-click the `.bat` file →
**Edit** to see the full error, or run it from an **Anaconda Prompt**
instead of double-clicking from File Explorer.

## Running on Google Colab (via GitHub)

Colab is a good option if you don't want to set up a local Python
environment at all, or want to share a live, clickable link to the
dashboard rather than a screenshot. Once this repo is on GitHub, Colab
can pull it directly with `git clone` — no manual zip upload needed. It
still works differently from a local machine in two ways worth knowing:
Colab gives you a fresh VM each session (nothing persists unless you
mount Google Drive), and Streamlit needs to be tunneled to a public URL
since Colab doesn't expose ports directly.

**Steps:**

1. Open **https://colab.research.google.com/github/rayyantaimoor1/Ecommerce-customer-intelligence/blob/main/colab/Ecommerce_Customer_Intelligence_Colab.ipynb**
   directly — this opens the Colab notebook straight from your GitHub
   repo, no download/upload step at all.
   Alternatively: **colab.research.google.com → File → Open notebook →
   GitHub tab** → paste your repo URL → select the notebook.
2. Run the cells from top to bottom (**Runtime → Run all**, or one at a
   time to follow along):
   - **Mount Google Drive** — so your trained models and processed data
     persist between sessions instead of vanishing when the VM resets.
   - **Clone the repo** — pulls the project (including the bundled
     dataset) directly from GitHub into the Colab VM / your mounted Drive.
   - **Install dependencies** — FastAPI, Streamlit, and a few others
     Colab doesn't include by default.
   - **(Optional) upload the real dataset** if you want to replace the
     bundled `OnlineRetail.csv` with the full Kaggle/UCI file.
   - **Run all six notebooks** — executes the full pipeline unattended
     and reports OK/FAILED per notebook.
   - **Launch the dashboard with a public link** — starts Streamlit
     inside the Colab VM, then uses `localtunnel` to expose it at a
     public URL. Open that URL, enter the printed IP as the "Tunnel
     Password" when prompted, and the dashboard loads in your browser.
3. Keep the final tunnel cell running for as long as you want the
   dashboard link to stay live — stopping it (or closing the tab) ends
   the session.

**Updated your code and pushed to GitHub?** Re-run the clone cell — it
pulls (`git pull` if the folder already exists, or clones fresh
otherwise) so Colab always runs your latest pushed version rather than a
stale copy.

The Colab notebook includes a troubleshooting section at the bottom for
the most common snags (tunnel not loading yet, session disconnects, etc).

## Methodology summary

| Phase | Technique | Purpose |
|---|---|---|
| Cleaning | Drop missing CustomerID, non-product rows | Handle known real-world data quality issues |
| Feature engineering | RFM + basket size, product variety, cancellation rate, tenure | Roll ~540K transactions into one feature vector per customer |
| Preprocessing | log1p on skewed columns, then StandardScaler | Retail spend/frequency are heavily right-skewed; without this, a handful of wholesale accounts dominate every distance-based model |
| Visualization | PCA (8→3 components for 90% variance), t-SNE | Confirm clusterable structure exists before modeling |
| Segmentation | K-Means (Elbow + Silhouette), Hierarchical (Dendrogram) | Find K, assign marketing personas |
| Anomaly Detection | DBSCAN (k-distance plot for `eps`) | Flag outlier customers as noise |
| Serving | FastAPI + Streamlit | Real-time prediction, both API and UI |

## A key modeling lesson worth knowing (and worth mentioning in an interview)

Running K-Means directly on raw RFM values pushes the silhouette score to
strongly favor **K=2** — but that split is just "the few extreme
wholesale-scale accounts" vs "everyone else," which is statistically clean
but not actionable for marketing. Log-transforming the skewed columns
(Frequency, Monetary, AvgBasketValue, AvgItemsPerBasket, UniqueProducts)
before scaling fixes this and produces a genuinely useful K=4 with distinct,
interpretable personas. This is a real, common pitfall in RFM clustering —
see notebook 03 for the before/after comparison.

## Notes / Limitations

- This dataset has no ground-truth fraud labels — DBSCAN's "noise" points
  are a proxy for anomalous behavior, not a validated fraud model. For a
  labeled fraud-detection benchmark, pair this approach with a dataset like
  Credit Card Fraud Detection (Kaggle, mlg-ulb) and evaluate with
  precision/recall against known fraud cases.
- K-Means is the model actually served live, since it supports predicting
  on brand-new points naturally (`.predict()`). DBSCAN and Hierarchical
  Clustering don't support this the same way, so they're used for
  offline/dashboard analysis only.
- The FastAPI/Streamlit apps expect pre-aggregated RFM-style features per
  customer, not raw transaction rows — computing those features from a
  transaction log is a batch operation (see notebook 02), not something
  done per API call.

## License

MIT — see `LICENSE`.
