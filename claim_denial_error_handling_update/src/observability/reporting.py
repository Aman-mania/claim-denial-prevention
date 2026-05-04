"""Small helpers for showing error summaries in local CLI runs."""

from __future__ import annotations

from src.observability.error_tracker import ErrorTracker


def format_repeated_errors(tracker: ErrorTracker, min_count: int = 2, max_rows: int = 5) -> str:
    rows = tracker.get_repeated_errors(min_count=min_count)[:max_rows]
    if not rows:
        return "No repeated errors detected."

    lines = ["Repeated errors detected:"]
    for row in rows:
        lines.append(
            f"- {row.get('error_code')} in {row.get('component')}/{row.get('stage')} "
            f"occurred {row.get('count')} times: {row.get('last_message')}"
        )
    return "\n".join(lines)
