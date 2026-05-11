#!/usr/bin/env python3
"""Check Week 6 dependencies in the active Python environment.

Week 6 now has a resilient local backend stack:
- sentence-transformers is preferred for semantic embeddings, optional locally;
- faiss-cpu is preferred for faster local vector search, optional locally;
- scikit-learn is required for TF-IDF and hashing fallback embedders;
- pypdf is required only if PDF policy documents are used.
"""

from __future__ import annotations

import importlib.util
import sys

CHECKS = [
    ("sentence_transformers", "sentence-transformers", "optional", "preferred semantic embedding backend"),
    ("faiss", "faiss-cpu", "optional", "preferred local FAISS vector index"),
    ("pypdf", "pypdf", "conditional", "PDF policy document parsing"),
    ("sklearn", "scikit-learn", "required", "TF-IDF and HashingVectorizer fallback embedding backends"),
    ("numpy", "numpy", "required", "NumPy vector-search fallback and matrix storage"),
]


def main() -> int:
    print("Week 6 dependency check")
    print(f"Python: {sys.executable}")
    missing_required: list[str] = []
    missing_optional: list[str] = []
    missing_conditional: list[str] = []

    for module, package, level, purpose in CHECKS:
        ok = importlib.util.find_spec(module) is not None
        status = "OK" if ok else "MISSING"
        print(f"  {status:<8} {module:<24} package={package:<24} level={level:<11} purpose={purpose}")
        if not ok:
            if level == "required":
                missing_required.append(package)
            elif level == "conditional":
                missing_conditional.append(package)
            else:
                missing_optional.append(package)

    if missing_required:
        print("\nMissing required Week 6 package(s):")
        print("  python -m pip install " + " ".join(missing_required))
        return 1

    if missing_conditional:
        print("\nConditional package(s) missing:")
        print("  Install only if you ingest PDF policy documents:")
        print("  python -m pip install " + " ".join(missing_conditional))

    if missing_optional:
        print("\nOptional package(s) missing:")
        print("  The pipeline will still run locally using TF-IDF + NumPy vector search.")
        print("  For stronger semantic retrieval / faster local search, install:")
        print("  python -m pip install " + " ".join(missing_optional))

    print("\nWeek 6 can run with the available required dependencies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
