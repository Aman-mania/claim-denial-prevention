# Phase 8 Docker Runtime Fix

## Why this patch exists

The product UI crashed inside Docker with:

```text
ImportError: cannot import name 'DEFAULT_API_BASE_URL' from product_ui.api_client
```

The Phase 8 runtime refactor removed a constant that `product_ui/app.py` still imports. This patch restores that backwards-compatible constant while keeping the centralized runtime configuration.

The Docker build was also very slow because the generic `requirements.txt` includes `sentence-transformers`, which pulls PyTorch. The Docker runtime now uses lightweight deployment requirement files and keeps semantic RAG dependencies optional.

## Runtime options

### Recommended corporate/local Docker mode

Use TF-IDF Week 6 artifacts and lightweight Docker images:

```bash
rm -rf data/vector_store/* data/policies/processed/*
python run_week6.py --mode fallback --min-score 0.10
python scripts/check_phase8_vector_backend.py

docker compose build --no-cache
docker compose up
```

### Optional semantic mode

Only use this if Hugging Face access/cache is available and vector artifacts were built with sentence-transformers:

```env
INSTALL_SEMANTIC=true
```

Then rebuild Docker images. This installs `sentence-transformers` and `faiss-cpu`, so builds will be much slower.

## Test commands

```bash
python -m py_compile product_ui/api_client.py scripts/check_phase8_vector_backend.py
pytest tests/config/test_api_client_defaults.py tests/deployment/test_docker_lightweight_files.py -v
python scripts/check_phase8_vector_backend.py
bash scripts/docker_smoke_test.sh
```
