"""
Map model-level SHAP explanations to business reasons.

The model may expose features such as `billed_deviation_imputed_capped`.
Users should see a reason such as "Billed amount above expected benchmark".
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.explainability.reason_catalog import ReasonDefinition, get_reason_for_feature


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _feature_label(feature_name: str) -> str:
    try:
        from src.ml.explain import FEATURE_LABELS
        return FEATURE_LABELS.get(feature_name, feature_name)
    except Exception:
        return feature_name


def _direction_from_shap(shap_value: float) -> str:
    if shap_value > 0:
        return "increases_risk"
    if shap_value < 0:
        return "decreases_risk"
    return "neutral"


def _custom_reason_text(definition: ReasonDefinition, feature_name: str, feature_value: Any) -> str:
    """Add feature-aware wording while keeping the catalog as the source of truth."""
    val = _safe_float(feature_value)

    if feature_name == "billed_deviation_imputed_capped" and val is not None:
        if val > 0:
            return f"The billed amount is about {val:.1f}% above the expected benchmark for this procedure."
        if val < 0:
            return f"The billed amount is about {abs(val):.1f}% below the expected benchmark; verify the benchmark and amount."
    if feature_name == "billed_amount_missing" and bool(feature_value):
        return "The source claim did not include a billed amount; the model had to use median imputation."
    if feature_name == "diagnosis_code_missing" and bool(feature_value):
        return "The source claim is missing a diagnosis code, which weakens medical-necessity support."
    if feature_name == "procedure_code_missing" and bool(feature_value):
        return "The source claim is missing a procedure code, so the billed service is incomplete."
    if feature_name == "provider_violation_rate" and val is not None:
        return f"The provider's historical claim-quality risk score is {val:.2f}, which increased the model risk."
    if feature_name == "cost_match_encoded" and val is not None and val <= 1:
        return "The cost benchmark is not a strong regional match, so billing review confidence is lower."
    if feature_name == "severity_rank" and val is not None and val >= 2:
        return "The diagnosis has medium/high severity and may require stronger supporting documentation."

    return definition.default_reason_text


class ReasonMapper:
    """
    Converts SHAP top reasons into deduplicated business-reason rows.
    """

    def __init__(self, max_reasons: int = 3, positive_only_for_risky_claims: bool = True) -> None:
        self.max_reasons = max_reasons
        self.positive_only_for_risky_claims = positive_only_for_risky_claims

    def map(
        self,
        *,
        shap_explanation: dict[str, Any],
        claim_features: dict[str, Any],
        prediction: dict[str, Any],
    ) -> list[dict[str, Any]]:
        top_reasons = shap_explanation.get("top_reasons", []) or []
        risk_level = str(prediction.get("risk_level", "")).upper()
        risky_claim = risk_level in {"HIGH", "MEDIUM"}

        rows: list[dict[str, Any]] = []
        seen_reason_codes: set[str] = set()

        for item in top_reasons:
            feature = item.get("feature")
            shap_value = float(item.get("shap_value", 0.0) or 0.0)

            if self.positive_only_for_risky_claims and risky_claim and shap_value <= 0:
                continue

            definition = get_reason_for_feature(feature)
            if not definition:
                continue
            if definition.reason_code in seen_reason_codes:
                continue

            feature_value = claim_features.get(feature)
            direction = item.get("direction") or _direction_from_shap(shap_value)

            row = {
                "reason_code": definition.reason_code,
                "reason_title": definition.title,
                "reason_text": _custom_reason_text(definition, feature, feature_value),
                "business_category": definition.category,
                "feature_name": feature,
                "feature_label": _feature_label(feature),
                "feature_value": feature_value,
                "shap_value": round(shap_value, 6),
                "shap_direction": direction,
                "shap_output_unit": "raw_log_odds_contribution",
                "fix_suggestion": definition.fix_suggestion,
                "policy_query": definition.policy_query_template,
                "policy_tags": list(definition.policy_tags),
                "reason_definition": asdict(definition),
            }
            rows.append(row)
            seen_reason_codes.add(definition.reason_code)

            if len(rows) >= self.max_reasons:
                break

        # Graceful fallback: do not leave claims unexplained.
        if not rows and top_reasons:
            for item in top_reasons:
                feature = item.get("feature")
                definition = get_reason_for_feature(feature)
                if not definition or definition.reason_code in seen_reason_codes:
                    continue
                feature_value = claim_features.get(feature)
                shap_value = float(item.get("shap_value", 0.0) or 0.0)
                rows.append({
                    "reason_code": definition.reason_code,
                    "reason_title": definition.title,
                    "reason_text": _custom_reason_text(definition, feature, feature_value),
                    "business_category": definition.category,
                    "feature_name": feature,
                    "feature_label": _feature_label(feature),
                    "feature_value": feature_value,
                    "shap_value": round(shap_value, 6),
                    "shap_direction": item.get("direction") or _direction_from_shap(shap_value),
                    "shap_output_unit": "raw_log_odds_contribution",
                    "fix_suggestion": definition.fix_suggestion,
                    "policy_query": definition.policy_query_template,
                    "policy_tags": list(definition.policy_tags),
                    "reason_definition": asdict(definition),
                })
                seen_reason_codes.add(definition.reason_code)
                if len(rows) >= self.max_reasons:
                    break

        for rank, row in enumerate(rows, start=1):
            row["reason_rank"] = rank

        return rows
