#!/usr/bin/env python3
"""
Idempotently wire ErrorTracker into local pipeline entry points.

Run from the repository root:
    python tools/apply_observability_integration.py

Why a patcher instead of replacing files?
-----------------------------------------
Your project has been evolving quickly across Week 4 patches. This script adds
small observability hooks to the files that exist in your current working tree
without overwriting your optimized training / feature-engineering code.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HOOK_IMPORT = """\nfrom src.observability.pipeline_integration import (\n    record_pipeline_report,\n    summarize_error_events,\n    tracker_from_env,\n)\n"""


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _insert_after_imports(text: str) -> str:
    if "src.observability.pipeline_integration" in text:
        return text
    # Insert after last import/from block near top.
    lines = text.splitlines()
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            insert_at = i + 1
    lines.insert(insert_at, HOOK_IMPORT.strip("\n"))
    return "\n".join(lines) + "\n"


def _insert_tracker_after_setup_logging(text: str) -> str:
    if "error_tracker = tracker_from_env()" in text:
        return text
    pattern = re.compile(r"(setup_logging\([^\n]*\)\n)")
    match = pattern.search(text)
    if not match:
        return text
    indent = re.match(r"^(\s*)", match.group(1)).group(1)
    insertion = match.group(1) + f"{indent}error_tracker = tracker_from_env()\n"
    return text[: match.start()] + insertion + text[match.end():]


def _insert_after_report_assignment(text: str, *, component: str, stage: str) -> str:
    marker = f'record_pipeline_report(report, component="{component}"'
    if marker in text:
        return text
    # Match the first assignment to a variable named report.
    pattern = re.compile(r"^(\s*)report\s*=\s*.*?\.run\([^\n]*\)\n", flags=re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return text
    indent = match.group(1)
    insertion = (
        match.group(0)
        + f'{indent}error_events = record_pipeline_report(report, component="{component}", tracker=error_tracker, stage="{stage}")\n'
        + f'{indent}error_summary = summarize_error_events(error_events)\n'
    )
    return text[: match.start()] + insertion + text[match.end():]


def patch_run_file(filename: str, *, component: str, stage: str) -> bool:
    path = ROOT / filename
    if not path.exists():
        print(f"SKIP missing {filename}")
        return False
    original = _read(path)
    text = _insert_after_imports(original)
    text = _insert_tracker_after_setup_logging(text)
    text = _insert_after_report_assignment(text, component=component, stage=stage)
    if text != original:
        _write(path, text)
        print(f"PATCHED {filename}")
        return True
    print(f"OK already patched {filename}")
    return False


def main() -> int:
    patch_run_file("run_ingestion.py", component="ingestion", stage="run_ingestion")
    patch_run_file("run_silver.py", component="silver", stage="run_silver")
    patch_run_file("run_gold.py", component="gold", stage="run_gold")
    patch_run_file("run_train.py", component="ml", stage="run_train")
    print("\nDone. Rerun your pipeline and inspect logs/error_events.jsonl or run:")
    print("  python run_error_report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
