"""
Map model-level SHAP explanations to business reasons.

SHAP tells us which features pushed the model score up/down. This mapper turns
those technical features into business reasons. It also inserts critical rule
reasons first, so obvious claim-quality problems are never hidden just because
another feature has a larger SHAP value.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import structlog

from src.explainability.reason_catalog import (
    CRITICAL_FEATURES,
    ReasonDefinition,
    get_reason_for_feature,
)

logger = structlog.get_logger(__name__)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


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
    if feature_name == "billed_amount_missing" and _truthy_flag(feature_value):
        return "The source claim did not include a billed amount; the model had to use median imputation."
    if feature_name == "diagnosis_code_missing" and _truthy_flag(feature_value):
        return "The source claim is missing a diagnosis code, which weakens medical-necessity support."
    if feature_name == "procedure_code_missing" and _truthy_flag(feature_value):
        return "The source claim is missing a procedure code, so the billed service is incomplete."
    if feature_name == "proc_no_diag" and _truthy_flag(feature_value):
        return "A procedure is present but no supporting diagnosis is available, which weakens medical-necessity support."
    if feature_name == "diag_no_proc" and _truthy_flag(feature_value):
        return "A diagnosis is present but no billed procedure is available, so the claim line is incomplete."
    if feature_name == "provider_violation_rate" and val is not None:
        return f"The provider's historical claim-quality risk score is {val:.2f}, which increased the model risk."
    if feature_name == "cost_match_encoded" and val is not None and val <= 1:
        return "The cost benchmark is not a strong regional match, so billing review confidence is lower."
    if feature_name == "severity_rank" and val is not None and val >= 2:
        return "The diagnosis has medium/high severity and may require stronger supporting documentation."

    return definition.default_reason_text


class ReasonMapper:
    """
    Converts SHAP top contributors into deduplicated business-reason rows.

    Ordering policy:
    1. Critical rule reasons triggered by the claim data.
    2. Positive SHAP contributors for HIGH/MEDIUM risk claims.
    3. Graceful fallback to mapped SHAP contributors if nothing else maps.
    """

    def __init__(self, max_reasons: int = 3, positive_only_for_risky_claims: bool = True) -> None:
        self.max_reasons = max_reasons
        self.positive_only_for_risky_claims = positive_only_for_risky_claims
        self.last_unmapped_features: list[str] = []

    def _row_from_feature(
        self,
        *,
        definition: ReasonDefinition,
        feature: str,
        feature_value: Any,
        shap_value: float,
        shap_direction: str,
        evidence_type: str,
    ) -> dict[str, Any]:
        return {
            "reason_code": definition.reason_code,
            "reason_title": definition.title,
            "reason_text": _custom_reason_text(definition, feature, feature_value),
            "business_category": definition.category,
            "evidence_type": evidence_type,
            "feature_name": feature,
            "feature_label": _feature_label(feature),
            "feature_value": feature_value,
            "shap_value": round(float(shap_value), 6),
            "shap_direction": shap_direction,
            "shap_output_unit": "raw_log_odds_contribution",
            "fix_suggestion": definition.fix_suggestion,
            "policy_query": definition.policy_query_template,
            "policy_tags": list(definition.policy_tags),
            "reason_definition": asdict(definition),
        }

    def _critical_rule_rows(
        self,
        *,
        claim_features: dict[str, Any],
        shap_by_feature: dict[str, dict[str, Any]],
        seen_reason_codes: set[str],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for feature in CRITICAL_FEATURES:
            if not _truthy_flag(claim_features.get(feature)):
                continue
            definition = get_reason_for_feature(feature)
            if not definition or definition.reason_code in seen_reason_codes:
                continue
            shap_item = shap_by_feature.get(feature, {})
            shap_value = float(shap_item.get("shap_value", 0.0) or 0.0)
            direction = shap_item.get("direction") or _direction_from_shap(shap_value)
            rows.append(
                self._row_from_feature(
                    definition=definition,
                    feature=feature,
                    feature_value=claim_features.get(feature),
                    shap_value=shap_value,
                    shap_direction=direction,
                    evidence_type="critical_rule",
                )
            )
            seen_reason_codes.add(definition.reason_code)
        return rows

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

        shap_by_feature = {item.get("feature"): item for item in top_reasons if item.get("feature")}
        rows: list[dict[str, Any]] = []
        seen_reason_codes: set[str] = set()
        self.last_unmapped_features = []

        # Critical data-quality rules are business-mandatory, not optional ML text.
        rows.extend(
            self._critical_rule_rows(
                claim_features=claim_features,
                shap_by_feature=shap_by_feature,
                seen_reason_codes=seen_reason_codes,
            )
        )

        for item in top_reasons:
            if len(rows) >= self.max_reasons:
                break
            feature = item.get("feature")
            shap_value = float(item.get("shap_value", 0.0) or 0.0)

            if self.positive_only_for_risky_claims and risky_claim and shap_value <= 0:
                continue

            definition = get_reason_for_feature(feature)
            if not definition:
                if feature:
                    self.last_unmapped_features.append(str(feature))
                continue
            if definition.reason_code in seen_reason_codes:
                continue

            rows.append(
                self._row_from_feature(
                    definition=definition,
                    feature=str(feature),
                    feature_value=claim_features.get(feature),
                    shap_value=shap_value,
                    shap_direction=item.get("direction") or _direction_from_shap(shap_value),
                    evidence_type="shap",
                )
            )
            seen_reason_codes.add(definition.reason_code)

        # Graceful fallback: avoid returning an empty reason set.
        if not rows and top_reasons:
            for item in top_reasons:
                feature = item.get("feature")
                definition = get_reason_for_feature(feature)
                if not definition or definition.reason_code in seen_reason_codes:
                    continue
                shap_value = float(item.get("shap_value", 0.0) or 0.0)
                rows.append(
                    self._row_from_feature(
                        definition=definition,
                        feature=str(feature),
                        feature_value=claim_features.get(feature),
                        shap_value=shap_value,
                        shap_direction=item.get("direction") or _direction_from_shap(shap_value),
                        evidence_type="fallback",
                    )
                )
                seen_reason_codes.add(definition.reason_code)
                if len(rows) >= self.max_reasons:
                    break

        if self.last_unmapped_features:
            logger.warning(
                "unmapped_shap_features_seen",
                count=len(self.last_unmapped_features),
                sample=self.last_unmapped_features[:5],
            )

        for rank, row in enumerate(rows[: self.max_reasons], start=1):
            row["reason_rank"] = rank

        return rows[: self.max_reasons]
