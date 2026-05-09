"""
Business reason catalog for claim-denial explainability.

The catalog is the bridge between raw ML features and user-facing reasoning.
It is intentionally deterministic: the system does not need GenAI to decide why
something is risky. SHAP identifies influential features; this catalog maps
those features to reusable business explanations and Week 6 policy queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ReasonDefinition:
    reason_code: str
    title: str
    category: str
    default_reason_text: str
    fix_suggestion: str
    source_features: tuple[str, ...]
    policy_query_template: str
    policy_tags: tuple[str, ...]
    severity: str = "review"
    critical_rule: bool = False


REASON_CATALOG: dict[str, ReasonDefinition] = {
    "MISSING_DIAGNOSIS": ReasonDefinition(
        reason_code="MISSING_DIAGNOSIS",
        title="Missing diagnosis code",
        category="claim_completeness",
        default_reason_text="The claim is missing a diagnosis code, so the payer may not be able to verify medical necessity.",
        fix_suggestion="Add a valid diagnosis code that supports the billed procedure.",
        source_features=("diagnosis_code_missing",),
        policy_query_template=(
            "Diagnosis code is required on a healthcare claim to support medical necessity "
            "and payer adjudication."
        ),
        policy_tags=("diagnosis", "medical_necessity", "claim_completeness"),
        severity="high",
        critical_rule=True,
    ),
    "MISSING_PROCEDURE": ReasonDefinition(
        reason_code="MISSING_PROCEDURE",
        title="Missing procedure code",
        category="claim_completeness",
        default_reason_text="The claim is missing a procedure code, so the payer may not know what service was billed.",
        fix_suggestion="Add the correct procedure/CPT code before submission.",
        source_features=("procedure_code_missing",),
        policy_query_template="Procedure code is required to identify the service billed on a claim.",
        policy_tags=("procedure", "coding", "claim_completeness"),
        severity="high",
        critical_rule=True,
    ),
    "MISSING_AMOUNT": ReasonDefinition(
        reason_code="MISSING_AMOUNT",
        title="Missing billed amount",
        category="claim_completeness",
        default_reason_text="The billed amount is missing, which makes payment adjudication incomplete.",
        fix_suggestion="Enter the billed amount and verify it against the expected cost benchmark.",
        source_features=("billed_amount_missing",),
        policy_query_template="Billed amount charge information is required for payer claim adjudication.",
        policy_tags=("billing", "amount", "claim_completeness"),
        severity="high",
        critical_rule=True,
    ),
    "PROCEDURE_WITHOUT_DIAGNOSIS": ReasonDefinition(
        reason_code="PROCEDURE_WITHOUT_DIAGNOSIS",
        title="Procedure without diagnosis support",
        category="medical_necessity",
        default_reason_text="A procedure is present but the diagnosis is missing, so medical necessity is not supported.",
        fix_suggestion="Link the billed procedure to a valid supporting diagnosis code.",
        source_features=("proc_no_diag",),
        policy_query_template=(
            "Procedure must be supported by a diagnosis code or clinical indication "
            "to establish medical necessity."
        ),
        policy_tags=("diagnosis", "procedure", "medical_necessity"),
        severity="high",
        critical_rule=True,
    ),
    "DIAGNOSIS_WITHOUT_PROCEDURE": ReasonDefinition(
        reason_code="DIAGNOSIS_WITHOUT_PROCEDURE",
        title="Diagnosis without billed procedure",
        category="claim_completeness",
        default_reason_text="A diagnosis is present but the procedure is missing, so the billed service is incomplete.",
        fix_suggestion="Add the procedure code for the performed service or remove unsupported diagnosis-only billing.",
        source_features=("diag_no_proc",),
        policy_query_template="Claim line requires procedure/service information when billing for a diagnosis-related encounter.",
        policy_tags=("diagnosis", "procedure", "claim_completeness"),
        severity="medium",
        critical_rule=True,
    ),
    "HIGH_BILLING_AMOUNT": ReasonDefinition(
        reason_code="HIGH_BILLING_AMOUNT",
        title="Billed amount above expected benchmark",
        category="billing_review",
        default_reason_text="The billed amount is materially higher than the expected benchmark for this procedure.",
        fix_suggestion="Verify the billed amount, check modifiers/units, and attach supporting documentation if justified.",
        source_features=(
            "billed_deviation_imputed_capped",
            "billed_amount_imputed",
            "log_billed_amount_imputed",
            "is_high_cost",
        ),
        policy_query_template=(
            "Claims with charges above expected or usual cost benchmarks may require "
            "documentation, justification, or additional review."
        ),
        policy_tags=("billing", "high_cost", "documentation"),
        severity="medium",
    ),
    "WEAK_COST_BENCHMARK": ReasonDefinition(
        reason_code="WEAK_COST_BENCHMARK",
        title="Weak or missing cost benchmark",
        category="billing_review",
        default_reason_text="The system could not find a strong regional cost benchmark for this procedure.",
        fix_suggestion="Review the procedure cost manually or add/update regional benchmark data.",
        source_features=("cost_match_encoded",),
        policy_query_template="Cost benchmark or fee schedule review for billed procedure and region.",
        policy_tags=("billing", "benchmark", "fee_schedule"),
        severity="low",
    ),
    "PROVIDER_HISTORY_RISK": ReasonDefinition(
        reason_code="PROVIDER_HISTORY_RISK",
        title="Provider history indicates higher claim-quality risk",
        category="provider_history",
        default_reason_text="This provider has a higher historical rate of incomplete or structurally risky claims.",
        fix_suggestion="Review the provider submission checklist before sending the claim.",
        source_features=("provider_violation_rate",),
        policy_query_template="Provider claims with repeated missing documentation or incomplete claim fields require review.",
        policy_tags=("provider", "documentation", "quality_review"),
        severity="medium",
    ),
    "PROVIDER_VOLUME_RISK": ReasonDefinition(
        reason_code="PROVIDER_VOLUME_RISK",
        title="Provider has high claim volume",
        category="provider_history",
        default_reason_text="The provider has a high volume of claims, which can increase duplicate or batch-submission review risk.",
        fix_suggestion="Check for duplicate submissions and validate claim batch quality.",
        source_features=("provider_claim_count",),
        policy_query_template="High-volume provider claim submissions should be checked for duplicate or incomplete billing.",
        policy_tags=("provider", "duplicate_claim", "batch_review"),
        severity="low",
    ),
    "PATIENT_FREQUENCY_RISK": ReasonDefinition(
        reason_code="PATIENT_FREQUENCY_RISK",
        title="Patient has frequent claims",
        category="utilization_review",
        default_reason_text="This patient has multiple claims in the dataset, so duplicate or utilization review may be needed.",
        fix_suggestion="Verify this is not a duplicate claim and ensure medical documentation supports the service.",
        source_features=("patient_claim_count",),
        policy_query_template="Frequent patient claims may require duplicate claim review or utilization review.",
        policy_tags=("patient", "duplicate_claim", "utilization_review"),
        severity="low",
    ),
    "HIGH_SEVERITY_SUPPORT": ReasonDefinition(
        reason_code="HIGH_SEVERITY_SUPPORT",
        title="Diagnosis severity needs supporting documentation",
        category="documentation",
        default_reason_text="The diagnosis severity may require stronger supporting documentation for the billed service.",
        fix_suggestion="Attach relevant clinical notes and supporting documentation.",
        source_features=("severity_rank", "severity_encoded"),
        policy_query_template="Severe diagnoses and high-risk services require supporting clinical documentation.",
        policy_tags=("severity", "documentation", "medical_necessity"),
        severity="medium",
    ),
    "SPECIALTY_REVIEW": ReasonDefinition(
        reason_code="SPECIALTY_REVIEW",
        title="Provider specialty should match billed service",
        category="provider_credentials",
        default_reason_text="The provider specialty may need to be checked against the billed service.",
        fix_suggestion="Verify provider specialty and credentials match the procedure being billed.",
        source_features=("specialty_encoded",),
        policy_query_template="Provider specialty or credentials must be appropriate for the billed procedure.",
        policy_tags=("provider", "specialty", "credentialing"),
        severity="low",
    ),
}


FEATURE_TO_REASON: dict[str, str] = {
    feature: reason_code
    for reason_code, definition in REASON_CATALOG.items()
    for feature in definition.source_features
}

CRITICAL_FEATURES: tuple[str, ...] = tuple(
    feature
    for definition in REASON_CATALOG.values()
    if definition.critical_rule
    for feature in definition.source_features
)


def get_reason_for_feature(feature_name: str) -> ReasonDefinition | None:
    """Return the reason definition for a model feature."""
    reason_code = FEATURE_TO_REASON.get(feature_name)
    if not reason_code:
        return None
    return REASON_CATALOG[reason_code]


def policy_tags_for_reasons(reason_codes: Iterable[str]) -> list[str]:
    """Return unique policy tags for a set of reason codes."""
    tags: list[str] = []
    for code in reason_codes:
        definition = REASON_CATALOG.get(code)
        if not definition:
            continue
        for tag in definition.policy_tags:
            if tag not in tags:
                tags.append(tag)
    return tags
