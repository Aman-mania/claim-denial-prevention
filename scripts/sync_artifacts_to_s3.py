#!/usr/bin/env python3
"""Optional helper to upload/download project artifacts to/from S3.

Install optional deps first:
    python -m pip install -r requirements-aws.txt

Examples:
    python scripts/sync_artifacts_to_s3.py upload --bucket my-bucket --prefix claim-denial/dev
    python scripts/sync_artifacts_to_s3.py download --bucket my-bucket --prefix claim-denial/dev
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

PATHS = [
    Path("data/gold"),
    Path("data/vector_store"),
    Path("data/policies/raw"),
    Path("data/policies/processed"),
    Path("models"),
]


def _client():
    try:
        import boto3
    except Exception as exc:  # pragma: no cover - optional dependency
        raise SystemExit("boto3 is required. Install with: python -m pip install -r requirements-aws.txt") from exc
    return boto3.client("s3")


def _iter_files(base: Path):
    if base.is_file():
        yield base
    elif base.exists():
        yield from (p for p in base.rglob("*") if p.is_file())


def upload(bucket: str, prefix: str) -> None:
    s3 = _client()
    for rel_base in PATHS:
        base = ROOT / rel_base
        for path in _iter_files(base):
            rel = path.relative_to(ROOT).as_posix()
            key = f"{prefix.strip('/')}/{rel}"
            print(f"upload {rel} -> s3://{bucket}/{key}")
            s3.upload_file(str(path), bucket, key)


def download(bucket: str, prefix: str) -> None:
    s3 = _client()
    paginator = s3.get_paginator("list_objects_v2")
    prefix_norm = prefix.strip("/") + "/"
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix_norm):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix_norm):]
            if not rel or rel.endswith("/"):
                continue
            dest = ROOT / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"download s3://{bucket}/{key} -> {rel}")
            s3.download_file(bucket, key, str(dest))


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload/download local model/RAG/data artifacts to S3.")
    parser.add_argument("action", choices=["upload", "download"])
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", default="claim-denial/artifacts")
    args = parser.parse_args()
    if args.action == "upload":
        upload(args.bucket, args.prefix)
    else:
        download(args.bucket, args.prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
