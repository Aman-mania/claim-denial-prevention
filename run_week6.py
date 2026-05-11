#!/usr/bin/env python3
"""Week 6 convenience runner with explicit RAG backend configuration.

This wrapper intentionally keeps Week 6 local-first while making the backend
choices visible. Local development can run in either:

  preferred: SentenceTransformers embeddings + FAISS vector index
  fallback : TF-IDF embeddings + NumPy vector search
  auto     : let run_policy_ingest.py choose available fallbacks

Configuration can come from .env, environment variables, or CLI flags.
CLI flags override environment values.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE pairs from .env without adding a dependency.

    Values already present in the shell environment are preserved so CI,
    Databricks jobs, ECS tasks, or local shell exports can override .env.
    """
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _resolve_backend(mode: str, embedding_backend: str | None, vector_backend: str | None) -> tuple[str, str]:
    """Resolve user mode into explicit ingestion backend flags."""
    mode = mode.strip().lower()
    if mode == "preferred":
        return embedding_backend or "sentence-transformers", vector_backend or "faiss"
    if mode == "fallback":
        return embedding_backend or "tfidf", vector_backend or "numpy"
    if mode == "auto":
        return embedding_backend or "auto", vector_backend or "auto"
    raise ValueError("RAG mode must be one of: preferred, fallback, auto")


def parse_args() -> argparse.Namespace:
    _load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run Week 6 policy RAG ingestion + matching.")
    parser.add_argument(
        "--mode",
        choices=["preferred", "fallback", "auto"],
        default=_env_str("RAG_MODE", "preferred"),
        help="preferred=SentenceTransformers+FAISS, fallback=TF-IDF+NumPy, auto=ingest auto fallback.",
    )
    parser.add_argument("--embedding-backend", choices=["auto", "sentence-transformers", "tfidf", "sklearn-hashing"], default=None)
    parser.add_argument("--vector-backend", choices=["auto", "faiss", "numpy"], default=None)
    parser.add_argument("--embedding-model", default=_env_str("RAG_EMBEDDING_MODEL", DEFAULT_MODEL))
    parser.add_argument("--top-k", type=int, default=_env_int("RAG_TOP_K", 3))
    parser.add_argument("--min-score", type=float, default=_env_float("RAG_MIN_SCORE", 0.20))
    parser.add_argument("--limit", type=int, default=None, help="Optional claim limit for debugging policy matching.")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / _env_str("RAG_POLICY_RAW_DIR", "data/policies/raw"))
    parser.add_argument("--processed-dir", type=Path, default=ROOT / _env_str("RAG_POLICY_PROCESSED_DIR", "data/policies/processed"))
    parser.add_argument("--vector-dir", type=Path, default=ROOT / _env_str("RAG_VECTOR_DIR", "data/vector_store"))
    parser.add_argument("--chunk-size", type=int, default=_env_int("RAG_CHUNK_SIZE_WORDS", 450))
    parser.add_argument("--overlap", type=int, default=_env_int("RAG_CHUNK_OVERLAP_WORDS", 70))
    parser.add_argument("--hashing-features", type=int, default=_env_int("RAG_HASHING_FEATURES", 4096))
    parser.add_argument("--tfidf-max-features", type=int, default=_env_int("RAG_TFIDF_MAX_FEATURES", 8192))
    parser.add_argument("--fallback-backend", choices=["tfidf", "sklearn-hashing"], default=_env_str("RAG_FALLBACK_BACKEND", "tfidf"))
    parser.add_argument("--skip-ingest", action="store_true", help="Only run policy matching using an existing vector index.")
    parser.add_argument("--skip-match", action="store_true", help="Only run ingestion/index build.")
    parser.add_argument("--check-only", action="store_true", help="Print resolved config and dependency checks without running.")
    parser.add_argument(
        "--allow-preferred-fallback",
        action="store_true",
        default=_env_bool("RAG_ALLOW_PREFERRED_FALLBACK", False),
        help="In preferred mode, allow run_policy_ingest.py to fall back if SentenceTransformers is unavailable.",
    )
    return parser.parse_args()


def _configure_huggingface_env() -> None:
    """Set Hugging Face cache/timeouts before any HF-dependent subprocess starts."""
    hf_home = os.getenv("HF_HOME") or str(ROOT / ".cache" / "huggingface")
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", os.getenv("RAG_HF_ETAG_TIMEOUT", "60"))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", os.getenv("RAG_HF_DOWNLOAD_TIMEOUT", "120"))
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def _print_preflight(args: argparse.Namespace, embedding_backend: str, vector_backend: str) -> bool:
    """Print mode/dependency state. Return True when execution may continue."""
    st_ok = _module_available("sentence_transformers")
    faiss_ok = _module_available("faiss")
    sklearn_ok = _module_available("sklearn")
    numpy_ok = _module_available("numpy")

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Claim Denial Prevention — Week 6 RAG Preflight            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Mode:                 {args.mode}")
    print(f"  Embedding backend:    {embedding_backend}")
    print(f"  Embedding model:      {args.embedding_model}")
    print(f"  Vector backend:       {vector_backend}")
    print(f"  Top-K / min-score:    {args.top_k} / {args.min_score}")
    print(f"  Fallback backend:     {getattr(args, 'fallback_backend', 'tfidf')}")
    print(f"  Raw policy dir:       {args.raw_dir}")
    print(f"  Processed dir:        {args.processed_dir}")
    print(f"  Vector dir:           {args.vector_dir}")
    print(f"  HF_HOME:              {os.environ.get('HF_HOME')}")
    print(f"  HF timeouts:          etag={os.environ.get('HF_HUB_ETAG_TIMEOUT')}s, download={os.environ.get('HF_HUB_DOWNLOAD_TIMEOUT')}s")
    print()
    print("  Dependency status:")
    print(f"    sentence_transformers: {'OK' if st_ok else 'MISSING'}")
    print(f"    faiss:                 {'OK' if faiss_ok else 'MISSING'}")
    print(f"    sklearn:               {'OK' if sklearn_ok else 'MISSING'}")
    print(f"    numpy:                 {'OK' if numpy_ok else 'MISSING'}")

    ok = True
    if embedding_backend == "sentence-transformers":
        print()
        print("  Preflight note:")
        print("    SentenceTransformers may contact Hugging Face on first run to download/cache the model.")
        print("    If your network times out, run the model-cache command shown below or switch to fallback mode.")
        if not st_ok:
            ok = False
            print("\n  Missing required preferred dependency: sentence-transformers")
            print("    python -m pip install -U sentence-transformers")

    if vector_backend == "faiss" and not faiss_ok:
        ok = False
        print("\n  Missing required preferred dependency: faiss-cpu")
        print("    python -m pip install -U faiss-cpu")
        print("    If faiss-cpu is difficult on your Mac, run with --vector-backend numpy.")

    if embedding_backend in {"sklearn-hashing", "tfidf"} and not sklearn_ok:
        ok = False
        print("\n  Missing fallback dependency: scikit-learn")
        print("    python -m pip install -U scikit-learn")

    if vector_backend == "numpy" and not numpy_ok:
        ok = False
        print("\n  Missing fallback dependency: numpy")
        print("    python -m pip install -U numpy")

    if embedding_backend == "sentence-transformers":
        print("\n  Optional model-cache command:")
        print("    python tools/cache_week6_embedding_model.py")

    if not ok:
        print("\n  Preflight failed. Fix the dependency/config issue above, then rerun Week 6.")
    else:
        print("\n  Preflight passed.")
    return ok


def _run(cmd: list[str], *, env: dict[str, str]) -> int:
    printable = " ".join(shlex.quote(part) for part in cmd)
    print("\n$ " + printable)
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


def main() -> int:
    args = parse_args()
    _configure_huggingface_env()

    embedding_backend, vector_backend = _resolve_backend(
        args.mode,
        args.embedding_backend,
        args.vector_backend,
    )

    if not _print_preflight(args, embedding_backend, vector_backend):
        return 2
    if args.check_only:
        return 0

    env = os.environ.copy()
    commands: list[list[str]] = []

    if not args.skip_ingest:
        ingest_cmd = [
            sys.executable,
            str(ROOT / "run_policy_ingest.py"),
            "--raw-dir", str(args.raw_dir),
            "--processed-dir", str(args.processed_dir),
            "--vector-dir", str(args.vector_dir),
            "--model", args.embedding_model,
            "--embedding-backend", embedding_backend,
            "--vector-backend", vector_backend,
            "--chunk-size", str(args.chunk_size),
            "--overlap", str(args.overlap),
            "--hashing-features", str(args.hashing_features),
            "--tfidf-max-features", str(args.tfidf_max_features),
            "--fallback-backend", args.fallback_backend,
        ]
        if args.mode == "preferred" and not args.allow_preferred_fallback and embedding_backend == "sentence-transformers":
            ingest_cmd.append("--no-embedding-fallback")
        commands.append(ingest_cmd)

    if not args.skip_match:
        match_cmd = [
            sys.executable,
            str(ROOT / "run_policy_match.py"),
            "--vector-dir", str(args.vector_dir),
            "--top-k", str(args.top_k),
            "--min-score", str(args.min_score),
        ]
        if args.limit is not None:
            match_cmd.extend(["--limit", str(args.limit)])
        commands.append(match_cmd)

    for cmd in commands:
        code = _run(cmd, env=env)
        if code != 0:
            print(f"\nWeek 6 stopped because command exited with code {code}.")
            return code

    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║   ✓  Week 6 Policy RAG completed.                          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    return 0


if __name__ == "__main__":
    sys.exit(main())
