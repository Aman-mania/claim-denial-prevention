#!/usr/bin/env python3
"""Audit generated Parquet tables for dtype/storage optimization opportunities.

Run from repo root:
    python tools/audit_tables.py
    python tools/audit_tables.py --root data/gold --details

The audit is advisory. It helps catch columns that should be bool/int8/float32,
mixed object columns that may fail Parquet/Delta writes, and repeated small-domain
strings that should have encoded columns for ML/cloud cost efficiency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from src.io.table_contracts import TABLE_CONTRACTS, enforce_table_contract, normalize_table_name
except Exception:  # pragma: no cover - useful when run before env setup
    TABLE_CONTRACTS = {}

    def normalize_table_name(name: str | Path) -> str:
        return Path(str(name)).stem

    def enforce_table_contract(df: pd.DataFrame, table_name: str | Path) -> pd.DataFrame:
        return df


BINARY_VALUES = {0, 1, True, False}


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _suggestions_for_column(df: pd.DataFrame, col: str) -> list[str]:
    s = df[col]
    suggestions: list[str] = []
    non_null = s.dropna()
    nunique = int(non_null.nunique()) if len(non_null) else 0

    if pd.api.types.is_integer_dtype(s):
        vals = set(non_null.unique().tolist()) if nunique <= 10 else set()
        max_val = int(non_null.max()) if len(non_null) else 0
        min_val = int(non_null.min()) if len(non_null) else 0
        if vals and vals.issubset(BINARY_VALUES):
            suggestions.append("binary integer → bool/int8")
        elif -128 <= min_val and max_val <= 127:
            suggestions.append("int64/integer → int8")
        elif -32768 <= min_val and max_val <= 32767:
            suggestions.append("int64/integer → int16")
        elif -2147483648 <= min_val and max_val <= 2147483647:
            suggestions.append("int64/integer → int32")

    if pd.api.types.is_float_dtype(s):
        if col not in {"billed_amount", "expected_cost", "average_cost"}:
            suggestions.append("float64/model-derived numeric → consider float32")

    if s.dtype == "object":
        suggestions.append("object dtype → explicit string/bool/numeric contract")

    if pd.api.types.is_string_dtype(s) or s.dtype == "object":
        if 0 < nunique <= 10 and len(non_null) > 100:
            suggestions.append("low-cardinality text → keep text for audit, add/use encoded column for ML")

    return suggestions


def audit_table(path: Path) -> dict[str, Any]:
    df = pd.read_parquet(path)
    table_name = normalize_table_name(path)
    contract = TABLE_CONTRACTS.get(table_name) if isinstance(TABLE_CONTRACTS, dict) else None
    contracted = enforce_table_contract(df, table_name) if contract else df

    cols = []
    for col in df.columns:
        suggestions = _suggestions_for_column(df, col)
        expected = contract.dtypes.get(col) if contract and col in contract.dtypes else None
        after_dtype = str(contracted[col].dtype) if col in contracted.columns else None
        cols.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "expected_contract_dtype": expected,
            "contracted_dtype": after_dtype,
            "null_pct": round(float(df[col].isna().mean() * 100), 2),
            "unique_count": int(df[col].nunique(dropna=True)),
            "suggestions": suggestions,
        })

    return {
        "table": table_name,
        "path": str(path),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "file_size_bytes": _file_size(path),
        "has_contract": bool(contract),
        "columns_detail": cols,
    }


def find_tables(root: Path) -> list[Path]:
    if root.is_file() and root.suffix == ".parquet":
        return [root]
    return sorted(root.rglob("*.parquet")) if root.exists() else []


def print_report(reports: list[dict[str, Any]], details: bool = False) -> None:
    if not reports:
        print("No Parquet tables found.")
        return

    print("\nTable dtype/storage audit")
    print("=" * 80)
    for r in reports:
        mb = r["file_size_bytes"] / 1024 / 1024
        contract_flag = "contract" if r["has_contract"] else "no contract"
        print(f"\n{r['table']}  rows={r['rows']:,} cols={r['columns']} size={mb:.3f} MB  [{contract_flag}]")
        flagged = [c for c in r["columns_detail"] if c["suggestions"] or (c["expected_contract_dtype"] and c["dtype"] != c["contracted_dtype"])]
        if not flagged:
            print("  ✓ No obvious dtype issues detected")
        else:
            for c in flagged:
                sugg = "; ".join(c["suggestions"]) or "contract would coerce dtype"
                expected = f" expected={c['expected_contract_dtype']}→{c['contracted_dtype']}" if c["expected_contract_dtype"] else ""
                print(f"  - {c['column']}: dtype={c['dtype']}{expected}; null={c['null_pct']}%; {sugg}")

        if details:
            for c in r["columns_detail"]:
                print(f"    {c['column']:<38} {c['dtype']:<12} null={c['null_pct']:>6}% unique={c['unique_count']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit local Parquet tables for dtype/storage optimization.")
    parser.add_argument("--root", default="data", help="Root directory or single parquet file to audit.")
    parser.add_argument("--details", action="store_true", help="Print all column details.")
    parser.add_argument("--json", dest="json_path", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    root = Path(args.root)
    tables = find_tables(root)
    reports = [audit_table(p) for p in tables]
    print_report(reports, details=args.details)

    if args.json_path:
        out = Path(args.json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(reports, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote JSON audit: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
