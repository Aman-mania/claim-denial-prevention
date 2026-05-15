# Phase 7 UI hardening notes

This patch improves the role-aware Streamlit product UI without changing backend APIs.

## Changes

- Replaces raw System Health JSON with role-aware health cards, readiness tables, and developer-only raw payload expanders.
- Adds a developer Risk Model governance snapshot before the embedded ML dashboard.
- Adds a Model Artifacts and Feature Contract section after the embedded ML dashboard.
- Keeps analyst UI business-focused and avoids local paths / raw JSON.

## Why

The previous System Health tab was technically correct but rendered raw JSON, which was not useful for analyst demos and not structured enough for developers. The Risk Model tab also needed a clearer governance summary and artifact summary around the older internal dashboard content.
