#!/usr/bin/env python3
"""Materialize source-controlled policy documents into data/policies/raw.

The project keeps policy seed documents under policy_docs/ because data/ is
intentionally gitignored. This utility copies curated seed docs into the RAG raw
folder and can optionally download official public government sources from the
registry.

Private payer pages are not downloaded by default. They are represented in the
curated payer reference pack and source registry to avoid redistributing payer
content or accidentally violating site terms.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_DOCS_DIR = PROJECT_ROOT / "policy_docs"
REGISTRY_PATH = POLICY_DOCS_DIR / "official_policy_source_registry.json"
SEED_DIR = POLICY_DOCS_DIR / "rag_seed"
RAW_POLICY_DIR = PROJECT_ROOT / "data" / "policies" / "raw"


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs):
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self._skip = True
        if tag.lower() in {"p", "div", "li", "h1", "h2", "h3", "h4", "br", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str):
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self._skip = False
        if tag.lower() in {"p", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str):
        if not self._skip:
            text = data.strip()
            if text:
                self.parts.append(text + " ")

    def text(self) -> str:
        raw = "".join(self.parts)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _safe_name(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "policy_source"


def _load_registry() -> dict:
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def copy_curated_docs(raw_dir: Path) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for src in sorted(SEED_DIR.glob("*.md")):
        dst = raw_dir / src.name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def write_source_summary_doc(raw_dir: Path, registry: dict) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dst = raw_dir / "official_policy_source_registry_summary.md"
    lines = [
        "---",
        "document_id: official_policy_source_registry_summary",
        "source_type: source_registry_summary",
        "tags: [policy_sources, cms, hhs, payer, hipaa, ncci, medical_necessity]",
        "---",
        "",
        "# Official Policy Source Registry Summary",
        "",
        "This document lists authoritative policy sources used by the Week 6 RAG corpus. It helps retrieval return source-level guidance even when full PDFs/pages are not downloaded locally.",
        "",
    ]
    for item in registry.get("sources", []):
        lines.extend([
            f"## {item['title']}",
            "",
            f"Source ID: `{item['source_id']}`",
            f"Publisher: {item.get('publisher', 'Unknown')}",
            f"Source type: {item.get('source_type', 'unknown')}",
            f"URL: {item.get('source_url', '')}",
            f"Policy tags: {', '.join(item.get('tags', []))}",
            "",
        ])
    dst.write_text("\n".join(lines), encoding="utf-8")
    return dst


def _download_url(url: str, timeout: int = 60) -> tuple[bytes, str]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "claim-denial-prevention-policy-sync/1.0 (+local educational project)"
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("content-type", "")
        return response.read(), content_type


def _materialize_download(item: dict, raw_dir: Path) -> Path:
    source_id = item["source_id"]
    title = item.get("title", source_id)
    url = item["source_url"]
    raw, content_type = _download_url(url)
    base = _safe_name(source_id)

    if "pdf" in content_type.lower() or url.lower().endswith(".pdf"):
        dst = raw_dir / f"{base}.pdf"
        dst.write_bytes(raw)
        return dst

    # Convert HTML/text pages into a RAG-friendly text document with metadata.
    text: str
    decoded = raw.decode("utf-8", errors="replace")
    if "html" in content_type.lower() or "<html" in decoded[:1000].lower():
        parser = _HTMLTextExtractor()
        parser.feed(decoded)
        text = parser.text()
    else:
        text = decoded

    dst = raw_dir / f"{base}.txt"
    header = (
        f"Source ID: {source_id}\n"
        f"Title: {title}\n"
        f"Publisher: {item.get('publisher', '')}\n"
        f"Source URL: {url}\n"
        f"Tags: {', '.join(item.get('tags', []))}\n\n"
    )
    dst.write_text(header + text, encoding="utf-8")
    return dst


def iter_downloadable_sources(registry: dict, include_payer_downloads: bool) -> Iterable[dict]:
    for item in registry.get("sources", []):
        if item.get("source_type") == "internal_project_policy":
            continue
        if item.get("download_allowed_by_default"):
            yield item
        elif include_payer_downloads and item.get("source_type") == "public_payer_reference":
            yield item


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync policy docs into data/policies/raw for Week 6 RAG.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_POLICY_DIR)
    parser.add_argument("--include-curated", action="store_true", help="Copy curated seed Markdown docs.")
    parser.add_argument("--include-source-summaries", action="store_true", help="Write registry summary markdown.")
    parser.add_argument("--download-official", action="store_true", help="Download official government pages/PDFs from registry.")
    parser.add_argument("--include-payer-downloads", action="store_true", help="Also download public payer pages. Review payer site terms before use.")
    parser.add_argument("--all", action="store_true", help="Equivalent to curated + source summaries + official downloads, excluding payer downloads.")
    args = parser.parse_args()

    if args.all:
        args.include_curated = True
        args.include_source_summaries = True
        args.download_official = True

    registry = _load_registry()
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    if args.include_curated:
        written.extend(str(p) for p in copy_curated_docs(raw_dir))
    if args.include_source_summaries:
        written.append(str(write_source_summary_doc(raw_dir, registry)))
    if args.download_official:
        for item in iter_downloadable_sources(registry, args.include_payer_downloads):
            try:
                path = _materialize_download(item, raw_dir)
                written.append(str(path))
                print(f"downloaded: {item['source_id']} -> {path}")
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                print(f"warning: failed to download {item.get('source_id')}: {exc}", file=sys.stderr)

    manifest = {
        "registry_version": registry.get("registry_version"),
        "raw_policy_dir": str(raw_dir),
        "files_written": written,
        "note": "Run python run_policy_ingest.py after syncing policy documents.",
    }
    manifest_path = raw_dir / "policy_sync_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Policy sync complete. Files written: {len(written)}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
