# AI-Powered Claim Denial Prevention & Remediation System

> Real-time decision support for validating healthcare claims **before** submission.  
> Predicts denial risk · Explains reasons · Suggests fixes

---

## Project status

| Week | Module | Status |
|------|--------|--------|
| 1 | Bronze ingestion pipeline | ✅ Complete |
| 2 | Analytics layer + Streamlit basics | — |
| 3 | Silver layer (cleaning) | — |
| 4 | Gold layer (feature engineering) | — |
| 5 | ML model (Isolation Forest + SHAP) | — |
| 6 | RAG system (sentence-transformers + FAISS) | — |
| 7 | Decision engine + FastAPI + Auth | — |
| 8 | Full integration + AWS deployment | — |

---

## Week 1 — Bronze Layer

### What it does

Ingests four raw CSV files into an immutable Bronze Parquet layer.

- Schema validation (Pandera) — soft warnings, never rejects data
- Metadata attachment — `ingestion_timestamp`, `source_file` per row
- Data profiling — null rates, duplicates, cardinality, numeric stats

### Setup

```bash
# 1. Clone and enter project
git clone <repo-url>
cd claim-denial-prevention

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place raw data files
cp /path/to/claims_1000.csv    data/raw/
cp /path/to/providers_1000.csv data/raw/
cp /path/to/diagnosis.csv      data/raw/
cp /path/to/cost.csv           data/raw/
```

### Run ingestion

```bash
python run_ingestion.py
```

Expected output:
```
╔══════════════════════════════════════════════════════════════╗
║   Claim Denial Prevention — Week 1: Bronze Layer Ingestion  ║
╚══════════════════════════════════════════════════════════════╝

[ Step 1/2 ] Running ingestion pipeline …

  ┌──────────────────┬──────────┬──────────────────────────────┐
  │ Dataset          │     Rows │ Status                       │
  ├──────────────────┼──────────┼──────────────────────────────┤
  │ ✓ claims         │     1000 │ success                      │
  │ ✓ providers      │       21 │ success                      │
  │ ✓ diagnosis      │        6 │ success                      │
  │ ✓ cost           │        6 │ success                      │
  └──────────────────┴──────────┴──────────────────────────────┘

[ Step 2/2 ] Running data profiler …
...
```

### Run tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=term-missing

# Single test file
pytest tests/ingestion/test_ingest.py -v
```

---

## Project structure (Week 1)

```
claim-denial-prevention/
├── data/
│   ├── raw/                    ← place source CSVs here (gitignored)
│   └── bronze/                 ← Parquet output (gitignored)
├── src/
│   ├── config.py               ← structlog setup
│   └── ingestion/
│       ├── schema.py           ← Pandera schemas for all 4 datasets
│       ├── ingest.py           ← BronzeIngestionPipeline
│       └── profiler.py         ← DataProfiler
├── tests/
│   ├── conftest.py             ← shared fixtures
│   └── ingestion/
│       └── test_ingest.py      ← 16 unit + integration tests
├── run_ingestion.py            ← entry point
├── requirements.txt
└── README.md
```

---

## Architecture

```
Raw CSVs (data/raw/)
    │
    ▼
Schema Validation (Pandera) — soft warnings
    │
    ▼
Bronze Layer (data/bronze/*.parquet)
    │  + ingestion_timestamp
    │  + source_file
    │  [NO transformations, NO cleaning]
    │
    ▼  [Week 3]
Silver Layer — cleaned, validated, joined
    │
    ▼  [Week 4]
Gold Layer — feature store
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
Rule Engine   ML Model (IF→XGB+SHAP)  RAG (sentence-transformers + FAISS)
    │                  │                           │
    └──────────────────┴───────────────────────────┘
                       │  [Week 7]
                       ▼
              Decision Engine (Rules + ML + RAG)
                       │
                       ▼
              FastAPI (POST /validate-claim)
                       │
                       ▼
              Streamlit Dashboard
```

---

## Design principles

- **Bronze is immutable** — never modify after creation. Cleaning is Silver's job.
- **Soft validation** — schema errors are warnings in Bronze, not failures. Preserves all data.
- **Extendable registry** — adding a new data source = 1 line in `DATASET_REGISTRY` + 1 schema.
- **Local-first** — runs entirely on local machine. AWS-ready when needed (Week 8).
- **No placeholders** — only files that are actually used exist in the repo.
