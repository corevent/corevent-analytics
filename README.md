# Corevent Analytics

Offline data mining for the Corevent event/ticketing domain:

- **Apriori** — association rules over event categories (favorites + paid orders)
- **K-Means** — user segmentation from engagement features
- **Random Forest + MLP** — supervised classification (favorite → purchase conversion)

This repo is separate from `api-corevent` / the frontend. It only **reads** Postgres (or falls back to synthetic data).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Point `.env` at the same Postgres used by the API (`DB_URL` or `DB_HOST` / `DB_USER` / …).

## Data source

1. Tries a read-only load from Postgres (users, events, orders, favorites, tickets, ratings, ticket types — **no PII** such as email, document, password, phone, or QR tokens).
2. If the connection fails or critical tables (`events` / engagement) are empty, generates a **synthetic** dataset with the same enums as Corevent.

## Run the pipeline

```bash
python -m corevent_analytics.pipeline
# or
corevent-analytics
```

Artifacts land in `outputs/` (rules CSV, cluster summary, metrics JSON, plots).

## Notebooks

For team visualization only — logic lives in `src/corevent_analytics/`:

```bash
jupyter lab notebooks/
```

- `01_eda.ipynb` — overview of loaded frames
- `02_apriori.ipynb` — association rules
- `03_kmeans.ipynb` — clusters
- `04_classification.ipynb` — RF vs MLP

## Tests

```bash
pytest
```

## Course mapping

| Requirement | Technique |
|-------------|-----------|
| Preprocessing / cleaning | `data/preprocess.py` |
| Association | Apriori |
| Clustering | K-Means |
| Supervised + neural | Random Forest + MLP |
| Unsupervised | Apriori + K-Means |
