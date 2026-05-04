"""
Optional MLflow / Model Registry integration
===========================================
This file is intentionally optional: local development works without mlflow.
During AWS + Databricks migration, call log_training_run_to_mlflow() after
run_train.py to register reports and model artifacts in MLflow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def log_training_run_to_mlflow(
    models_dir: Path,
    experiment_name: str = "claim-denial-prevention",
    registered_model_name: str | None = "claim_denial_risk_model",
) -> dict[str, Any]:
    """
    Log local model artifacts to MLflow when mlflow is installed/configured.

    This keeps the core project dependency-light. In Databricks, mlflow is
    available by default, so this function can be used without changing the
    training pipeline itself.
    """
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError as exc:
        return {
            "status": "skipped",
            "reason": "mlflow is not installed in this local environment",
            "error": str(exc),
        }

    models_dir = Path(models_dir)
    report_path = models_dir / "training_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Training report not found: {report_path}. Run run_train.py first.")

    with open(report_path) as f:
        report = json.load(f)

    recommended = report.get("recommended_model", "xgboost")
    model_file = models_dir / ("xgb_model.pkl" if recommended == "xgboost" else "lr_model.pkl")

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"{recommended}-week4") as run:
        mlflow.log_params({
            "recommended_model": recommended,
            "feature_count": report.get("feature_count"),
            "selected_threshold": report.get("risk_band_policy", {}).get("classification_threshold"),
        })
        tuned = report.get("recommended_model_test_at_tuned_threshold", {})
        for key in ["roc_auc", "recall", "precision", "f1", "accuracy"]:
            if key in tuned:
                mlflow.log_metric(f"test_{key}", tuned[key])
        calibration = report.get("calibration", {})
        if "brier_score" in calibration:
            mlflow.log_metric("brier_score", calibration["brier_score"])
        if "expected_calibration_error" in calibration:
            mlflow.log_metric("expected_calibration_error", calibration["expected_calibration_error"])

        for path in models_dir.glob("*.json"):
            mlflow.log_artifact(str(path), artifact_path="reports")
        if model_file.exists():
            mlflow.log_artifact(str(model_file), artifact_path="model_pickle")

        # We log the pickle as an artifact rather than registering automatically,
        # because registration strategy differs between local MLflow, Databricks
        # workspace registry, and Unity Catalog registry.
        return {
            "status": "success",
            "run_id": run.info.run_id,
            "registered_model_name": registered_model_name,
            "model_file": str(model_file),
        }
