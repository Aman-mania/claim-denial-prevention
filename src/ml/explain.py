"""
SHAP Explainer
===============
Computes SHAP values for a single claim prediction.
Returns top N features with their contribution direction and magnitude.

This module is the bridge between ML scores and human-readable explanations:
  risk_score = 0.82
    → reason 1: proc_no_diag contributed +0.31
    → reason 2: billed_deviation_capped contributed +0.24
    → reason 3: diagnosis_code_missing contributed +0.18

Week 7: the Decision Engine will call SHAPExplainer.explain() and include
the result in the final JSON output sent to FastAPI.

Only supports XGBoost (TreeExplainer) — Logistic Regression uses
the model's own coefficients × feature values which is less informative.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import pickle

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# Human-readable labels for each feature name
# Displayed in the dashboard and final API output
FEATURE_LABELS: dict[str, str] = {
    "diagnosis_code_missing":    "Missing Diagnosis Code",
    "procedure_code_missing":    "Missing Procedure Code",
    "billed_amount_missing":     "Missing Billed Amount",
    "proc_no_diag":              "Procedure Without Diagnosis",
    "diag_no_proc":              "Diagnosis Without Procedure",
    "billed_deviation_capped":   "Billing Deviation From Expected",
    "log_billed_amount":         "Billed Amount (log scale)",
    "is_high_cost":              "High Cost Claim",
    "provider_claim_count":      "Provider Activity Volume",
    "provider_violation_rate":   "Provider Historical Violation Rate",
    "patient_claim_count":       "Patient Claim Frequency",
    "severity_encoded":          "Diagnosis Severity",
    "specialty_encoded":         "Provider Specialty",
}

# Fix messages shown for top denial reasons (Week 7 uses these)
FIX_SUGGESTIONS: dict[str, str] = {
    "diagnosis_code_missing":   "Add a valid ICD diagnosis code before submission.",
    "procedure_code_missing":   "Specify the procedure code (CPT/PROC) performed.",
    "billed_amount_missing":    "Include the billed amount for the procedure.",
    "proc_no_diag":             "Link the billed procedure to a clinical diagnosis code.",
    "diag_no_proc":             "Add the procedure performed for this diagnosis.",
    "billed_deviation_capped":  "Review billed amount — significantly exceeds expected cost benchmark.",
    "log_billed_amount":        "Verify the billed amount is correct.",
    "is_high_cost":             "High-cost claim may require prior authorisation.",
    "provider_claim_count":     "High-volume provider — verify no duplicate submissions in this batch.",
    "provider_violation_rate":  "Provider has a history of incomplete claims — review submission checklist.",
    "patient_claim_count":      "High claim frequency for this patient — verify no duplicate submissions.",
    "severity_encoded":         "High-severity diagnosis — ensure all supporting documentation is attached.",
    "specialty_encoded":        "Verify claim specialty matches provider credentials.",
}


class SHAPExplainer:
    """
    Wraps a fitted XGBoost pipeline to produce per-claim SHAP explanations.

    Parameters
    ----------
    pipeline      : Fitted sklearn Pipeline containing an XGBoost model.
    feature_names : Ordered list of feature names (same order as training).
    """

    def __init__(self, pipeline: Any, feature_names: list[str]) -> None:
        import shap
        self.pipeline      = pipeline
        self.feature_names = feature_names
        self._xgb_model    = pipeline.named_steps["model"]
        self._imputer      = pipeline.named_steps["imputer"]
        self._explainer    = shap.TreeExplainer(self._xgb_model)

    @classmethod
    def from_model_file(cls, xgb_model_path: Path) -> "SHAPExplainer":
        """Load from a saved XGBoost model pkl file."""
        with open(xgb_model_path, "rb") as f:
            saved = pickle.load(f)
        return cls(
            pipeline=saved["pipeline"],
            feature_names=saved["features"],
        )

    def explain(
        self,
        claim_features: dict,
        top_n: int = 3,
    ) -> dict:
        """
        Compute SHAP explanation for a single claim.

        Parameters
        ----------
        claim_features : Dict of feature_name → value (same keys as training).
        top_n          : Number of top contributing features to return.

        Returns
        -------
        {
            "top_reasons": [
                {
                    "rank":        1,
                    "feature":     "proc_no_diag",
                    "label":       "Procedure Without Diagnosis",
                    "shap_value":  0.31,
                    "direction":   "increases_risk",  or "decreases_risk"
                    "fix":         "Link the billed procedure to a clinical diagnosis code.",
                },
                ...
            ],
            "base_value":  float,   # model's expected output (log-odds)
            "model_output": float,  # final log-odds for this claim
        }
        """
        # Build feature row (same logic as ClaimPredictor)
        row = {feat: claim_features.get(feat, np.nan) for feat in self.feature_names}
        X   = pd.DataFrame([row])

        # Apply imputer (same as training pipeline)
        X_imputed = self._imputer.transform(X)

        # Compute SHAP values
        raw_shap  = self._explainer.shap_values(X_imputed)
        # TreeExplainer returns shape (n_samples, n_features) for XGBoost
        # For a single sample we take [0]; handle both 1D and 2D
        shap_vals = raw_shap[0] if raw_shap.ndim == 2 else raw_shap

        # Build ranked list by absolute SHAP magnitude
        all_contribs = [
            {"feature": name, "shap_value": round(float(val), 4)}
            for name, val in zip(self.feature_names, shap_vals)
        ]

        # For denial explanation: show risk-INCREASING features first.
        # Only fall back to risk-decreasing features if fewer than top_n positives.
        # This answers "why is this claim at risk?" rather than "what's the largest signal?".
        increasing = sorted(
            [c for c in all_contribs if c["shap_value"] > 0],
            key=lambda x: -x["shap_value"],
        )
        decreasing = sorted(
            [c for c in all_contribs if c["shap_value"] < 0],
            key=lambda x: x["shap_value"],   # most negative first
        )

        # Fill top_n from positive contributors first; pad with negative if needed
        selected = (increasing + decreasing)[:top_n]

        top_reasons = []
        for rank, contrib in enumerate(selected, start=1):
            feat     = contrib["feature"]
            shap_val = contrib["shap_value"]
            top_reasons.append({
                "rank":       rank,
                "feature":    feat,
                "label":      FEATURE_LABELS.get(feat, feat),
                "shap_value": shap_val,
                "direction":  "increases_risk" if shap_val > 0 else "decreases_risk",
                "fix":        FIX_SUGGESTIONS.get(feat, "Review claim data."),
            })

        logger.info(
            "shap_explanation_computed",
            claim_id=claim_features.get("claim_id"),
            top_features=[r["feature"] for r in top_reasons],
        )

        return {
            "top_reasons":  top_reasons,
            "base_value":   round(float(self._explainer.expected_value), 4),
            "model_output": round(float(shap_vals.sum()), 4),
        }
