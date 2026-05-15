from product_ui.rendering import (
    artifact_health_counts,
    artifact_rows_from_payload,
    deployment_readiness_rows,
    overall_health_label,
)


def test_artifact_health_counts_and_label_ready():
    payload = {"artifacts": {"gold_claim_features": True, "xgb_model": True}}
    assert artifact_health_counts(payload) == {"total": 2, "ready": 2, "missing": 0}
    assert overall_health_label(True, payload) == "Ready"


def test_artifact_health_counts_and_label_degraded():
    payload = {"artifacts": {"gold_claim_features": True, "xgb_model": False}}
    assert artifact_health_counts(payload) == {"total": 2, "ready": 1, "missing": 1}
    assert overall_health_label(True, payload) == "Degraded"


def test_artifact_rows_hide_raw_json_shape():
    payload = {"artifacts": {"gold_claim_features": True, "vector_metadata": False}}
    rows = artifact_rows_from_payload(payload)
    assert rows == [
        {"artifact": "Gold Claim Features", "status": "Available", "ready": True},
        {"artifact": "Vector Metadata", "status": "Missing", "ready": False},
    ]


def test_deployment_readiness_mentions_rds_and_s3_as_aws_pending():
    rows = deployment_readiness_rows(api_ok=True, artifact_payload={"artifacts": {"xgb_model": True}})
    checks = {row["check"]: row["status"] for row in rows}
    assert checks["FastAPI backend"] == "Ready"
    assert checks["RDS PostgreSQL"] == "Pending AWS setup"
    assert checks["S3 artifact bucket"] == "Pending AWS setup"
