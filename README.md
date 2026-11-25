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

API-facing consolidated JSON (consumed by `api-corevent`):

- `outputs/eda.json`
- `outputs/apriori.json`
- `outputs/kmeans.json`
- `outputs/classification.json`

When `GCS_BUCKET` is set, those four files are also uploaded to
`gs://$GCS_BUCKET/$ANALYTICS_GCS_PREFIX` (default prefix `analytics/`).

## CI

GitHub Actions runs `pytest` and a Docker image build on push/PR
(`.github/workflows/ci.yml`).

## Docker

```bash
docker build -t corevent-analytics .
docker run --rm \
  -e DB_URL="postgresql+psycopg://..." \
  -e GCS_BUCKET=your-bucket \
  -e ANALYTICS_GCS_PREFIX=analytics/ \
  corevent-analytics
```

On GCP, prefer Cloud Run Jobs with a service account that can read Postgres
and write to the bucket (Application Default Credentials — no key file).

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
