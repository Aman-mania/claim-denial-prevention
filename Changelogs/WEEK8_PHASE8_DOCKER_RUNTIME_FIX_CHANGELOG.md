# Week 8 Phase 8 Docker Runtime Fix Changelog

## Fixed

- Restored `DEFAULT_API_BASE_URL` and `DEFAULT_API_TIMEOUT_SECONDS` exports in `product_ui.api_client` for compatibility with `product_ui/app.py`.
- Added lightweight Docker requirement files to avoid installing `sentence-transformers`/PyTorch unless explicitly needed.
- Updated Dockerfiles to use lightweight runtime dependencies.
- Added optional `INSTALL_SEMANTIC=true` path for semantic RAG Docker builds.
- Added vector backend compatibility check for Docker runtime.
- Improved Docker smoke test to also check that Streamlit is reachable.

## Notes

For corporate devices and fastest Docker builds, use Week 6 fallback TF-IDF artifacts before starting Docker Compose.
