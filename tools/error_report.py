#!/usr/bin/env python3
"""Print a compact local error report from logs/error_summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Show repeated pipeline errors.")
    parser.add_argument("--summary", default="logs/error_summary.json", help="Path to error_summary.json")
    parser.add_argument("--min-count", type=int, default=2, help="Minimum count to display")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of table")
    args = parser.parse_args()

    path = Path(args.summary)
    if not path.exists():
        print(f"No error summary found at {path}")
        return 0

    with open(path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    rows = [
        {"occurrence_key": key, **value}
        for key, value in summary.items()
        if int(value.get("count", 0)) >= args.min_count
    ]
    rows.sort(key=lambda r: int(r.get("count", 0)), reverse=True)

    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return 0

    if not rows:
        print(f"No repeated errors found with count >= {args.min_count}.")
        return 0

    print("\nRepeated errors")
    print("-" * 120)
    print(f"{'Count':>5}  {'Code':<12}  {'Component':<18}  {'Stage':<18}  {'Severity':<9}  Last message")
    print("-" * 120)
    for row in rows:
        print(
            f"{int(row.get('count', 0)):>5}  "
            f"{row.get('error_code', '—'):<12}  "
            f"{row.get('component', '—'):<18}  "
            f"{row.get('stage', '—'):<18}  "
            f"{row.get('severity', '—'):<9}  "
            f"{row.get('last_message', '—')}"
        )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
