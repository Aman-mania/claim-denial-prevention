#!/usr/bin/env python3
"""Idempotently wire table contracts into direct Parquet writers.

Most new Week 5+ code writes through LocalTableStore, which already enforces
contracts. Earlier pipeline layers still write directly with df.to_parquet().
This script updates those direct writers without replacing whole pipeline files.

Run from repo root:
    python tools/apply_table_contracts_integration.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _patch_file(path: Path, replacements: list[tuple[str, str]], imports: list[str]) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text

    for import_line in imports:
        if import_line not in text:
            # Add after structlog import where possible; otherwise after pandas import.
            if "import structlog\n" in text:
                text = text.replace("import structlog\n", f"import structlog\n{import_line}\n", 1)
            elif "import pandas as pd\n" in text:
                text = text.replace("import pandas as pd\n", f"import pandas as pd\n{import_line}\n", 1)
            else:
                text = f"{import_line}\n" + text

    for old, new in replacements:
        if new in text:
            continue
        if old not in text:
            print(f"[skip] Pattern not found in {path.relative_to(ROOT)}: {old[:80]!r}")
            continue
        text = text.replace(old, new, 1)

    if text != original:
        path.write_text(text, encoding="utf-8")
        print(f"[updated] {path.relative_to(ROOT)}")
        return True
    print(f"[ok] {path.relative_to(ROOT)} already integrated")
    return False


def main() -> int:
    updates = 0

    updates += _patch_file(
        ROOT / "src" / "ingestion" / "ingest.py",
        imports=["from src.io.table_contracts import parquet_write_kwargs"],
        replacements=[(
            '        df.to_parquet(out_path, index=False, engine="pyarrow")',
            '        df.to_parquet(out_path, index=False, engine="pyarrow", **parquet_write_kwargs())',
        )],
    )

    updates += _patch_file(
        ROOT / "src" / "silver" / "clean.py",
        imports=["from src.io.table_contracts import enforce_table_contract, parquet_write_kwargs"],
        replacements=[(
            '        df.to_parquet(out_path, index=False, engine="pyarrow")',
            '        table_name = f"{dataset_name}_silver"\n'
            '        df = enforce_table_contract(df, table_name)\n'
            '        df.to_parquet(out_path, index=False, engine="pyarrow", **parquet_write_kwargs())',
        )],
    )

    updates += _patch_file(
        ROOT / "src" / "gold" / "features.py",
        imports=["from src.io.table_contracts import enforce_table_contract, parquet_write_kwargs"],
        replacements=[(
            '        df.to_parquet(path, index=False, engine="pyarrow")',
            '        df = enforce_table_contract(df, table_name)\n'
            '        df.to_parquet(path, index=False, engine="pyarrow", **parquet_write_kwargs())',
        )],
    )

    print(f"\nDone. Files changed: {updates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
