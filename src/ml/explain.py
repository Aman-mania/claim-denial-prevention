"""
SHAP Explainer
===============
Computes SHAP values for a single claim prediction and turns them into
human-readable denial reasons.

SHAP values for the current XGBoost TreeExplainer are raw-margin/log-odds
contributions. They are excellent for ranking reasons, but should not be read
as percentage-point probability changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pickle

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# Human-readable labels for each feature name.
FEATURE_LABELS: dict[str, str] = {
    "diagnosis_code_missing": "Missing Diagnosis Code",
    "procedure_code_missing": "Missing Procedure Code",
    "billed_amount_missing": "Missing Billed Amount",
    "proc_no_diag": "Procedure Without Diagnosis",
    "diag_no_proc": "Diagnosis Without Procedure",
    "billed_deviation_imputed_capped": "Billing Deviation From Expected",
    "billed_amount_imputed": "Median-Imputed Billed Amount",
    "log_billed_amount_imputed": "Billed Amount (log scale, imputed)",
    "is_high_cost": "High-Cost Claim",
    "cost_match_encoded": "Cost Benchmark Match Quality",
    "provider_claim_count": "Provider Activity Volume",
    "provider_violation_rate": "Provider Historical Violation Rate",
    "patient_claim_count": "Patient Claim Frequency",
    "severity_rank": "Diagnosis Severity",
    "specialty_encoded": "Provider Specialty",
    # Legacy / audit features retained for older model artifacts and dashboards.
    "billed_deviation_capped": "Billing Deviation From Expected",
    "log_billed_amount": "Billed Amount (log scale)",
    "severity_encoded": "Diagnosis Severity",
}

# Fix messages shown for top denial reasons.
FIX_SUGGESTIONS: dict[str, str] = {
    "diagnosis_code_missing": "Add a valid ICD diagnosis code before submission.",
    "procedure_code_missing": "Specify the procedure code performed/billed.",
    "billed_amount_missing": "Include the billed amount; do not submit amount-missing claims.",
    "proc_no_diag": "Link the billed procedure to a clinical diagnosis code.",
    "diag_no_proc": "Add the procedure performed for this diagnosis, or remove the unsupported diagnosis-only entry.",
    "billed_deviation_imputed_capped": "Review billed amount against the expected benchmark; attach support if unusually high.",
    "billed_amount_imputed": "Verify the actual billed amount rather than relying on imputation.",
    "log_billed_amount_imputed": "Verify the billed amount is correct and not inflated.",
    "is_high_cost": "Check payer policy for prior authorization or extra documentation.",
    "cost_match_encoded": "Verify that procedure and region/location have a reliable benchmark.",
    "provider_claim_count": "High-volume provider — check for duplicate or batch submission issues.",
    "provider_violation_rate": "Provider has a history of incomplete claims — review the submission checklist.",
    "patient_claim_count": "High claim frequency for this patient — verify no duplicate or repeated submission.",
    "severity_rank": "Ensure documentation supports diagnosis severity.",
    "specialty_encoded": "Verify claim specialty matches provider credentials.",
    # Legacy keys.
    "billed_deviation_capped": "Review billed amount — significantly exceeds expected cost benchmark.",
    "log_billed_amount": "Verify the billed amount is correct.",
    "severity_encoded": "Ensure all supporting documentation is attached.",
}


class SHAPExplainer:
    """
    Wraps a fitted XGBoost pipeline to produce per-claim SHAP explanations.
    """

    def __init__(self, pipeline: Any, feature_names: list[str]) -> None:
        import shap

        self.pipeline = pipeline
        self.feature_names = feature_names
        self._xgb_model = pipeline.named_steps["model"]
        self._imputer = pipeline.named_steps["imputer"]
        self._explainer = shap.TreeExplainer(self._xgb_model)

    @classmethod
    def from_model_file(cls, xgb_model_path: Path) -> "SHAPExplainer":
        """Load from a saved XGBoost model pkl file."""
        with open(xgb_model_path, "rb") as f:
            saved = pickle.load(f)
        return cls(
            pipeline=saved["pipeline"],
            feature_names=saved["features"],
        )

    def _expected_value(self) -> float:
        expected = self._explainer.expected_value
        arr = np.asarray(expected).ravel()
        return float(arr[0])

    def explain(
        self,
        claim_features: dict,
        top_n: int = 3,
    ) -> dict:
        """
        Compute SHAP explanation for a single claim.

        Returns top contributing features. Values are raw log-odds SHAP
        contributions, not probability percentages.
        """
        row = {feat: claim_features.get(feat, np.nan) for feat in self.feature_names}
        X = pd.DataFrame([row])

        X_imputed = self._imputer.transform(X)

        raw_shap = self._explainer.shap_values(X_imputed)
        shap_vals = raw_shap[0] if getattr(raw_shap, "ndim", 1) == 2 else raw_shap

        all_contribs = [
            {"feature": name, "shap_value": round(float(val), 4)}
            for name, val in zip(self.feature_names, shap_vals)
        ]

        increasing = sorted(
            [c for c in all_contribs if c["shap_value"] > 0],
            key=lambda x: -x["shap_value"],
        )
        decreasing = sorted(
            [c for c in all_contribs if c["shap_value"] < 0],
            key=lambda x: x["shap_value"],
        )

        selected = (increasing + decreasing)[:top_n]

        top_reasons = []
        for rank, contrib in enumerate(selected, start=1):
            feat = contrib["feature"]
            shap_val = contrib["shap_value"]
            top_reasons.append({
                "rank": rank,
                "feature": feat,
                "label": FEATURE_LABELS.get(feat, feat),
                "shap_value": shap_val,
                "direction": "increases_risk" if shap_val > 0 else "decreases_risk",
                "fix": FIX_SUGGESTIONS.get(feat, "Review claim data."),
                "value_type": "raw_log_odds_contribution",
            })

        expected_value = self._expected_value()
        contribution_sum = float(np.asarray(shap_vals).sum())
        model_output = expected_value + contribution_sum

        logger.info(
            "shap_explanation_computed",
            claim_id=claim_features.get("claim_id"),
            top_features=[r["feature"] for r in top_reasons],
        )

        return {
            "top_reasons": top_reasons,
            "base_value": round(expected_value, 4),
            "contribution_sum": round(contribution_sum, 4),
            "model_output": round(model_output, 4),
            "model_output_type": "raw_log_odds",
            "note": "SHAP values are raw log-odds contributions for ranking reasons, not percentage-point probability changes.",
        }
