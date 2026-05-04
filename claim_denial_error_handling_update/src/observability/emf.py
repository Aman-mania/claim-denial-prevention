"""
CloudWatch Embedded Metric Format helpers.

Local development writes normal JSONL files. In AWS/ECS/Lambda/Databricks jobs,
these EMF JSON objects can be printed to stdout and CloudWatch can extract
metrics from logs.
"""

from __future__ import annotations

import json
import time
from typing import Any

from src.observability.constants import ERROR_METRICS_NAMESPACE, EMF_DIMENSIONS


def build_error_emf_event(
    *,
    error_code: str,
    component: str,
    stage: str,
    severity: str,
    count: int = 1,
    namespace: str = ERROR_METRICS_NAMESPACE,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a CloudWatch EMF-compatible error metric event."""
    event: dict[str, Any] = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": namespace,
                    "Dimensions": [list(EMF_DIMENSIONS)],
                    "Metrics": [
                        {"Name": "ErrorCount", "Unit": "Count"},
                        {"Name": "RepeatedErrorCount", "Unit": "Count"},
                    ],
                }
            ],
        },
        "component": component,
        "stage": stage,
        "error_code": error_code,
        "severity": severity,
        "ErrorCount": 1,
        "RepeatedErrorCount": max(count - 1, 0),
    }
    if extra:
        # Extra fields are allowed in EMF, but avoid putting high-cardinality
        # values in dimensions.
        event.update(extra)
    return event


def build_error_emf_json(**kwargs: Any) -> str:
    """Return the EMF event as a single-line JSON string."""
    return json.dumps(build_error_emf_event(**kwargs), default=str, separators=(",", ":"))
