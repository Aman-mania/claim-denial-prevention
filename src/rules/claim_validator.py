"""Flexible input validation for the custom claim product flow.

This validator is deliberately not a denial-decision rule engine. It is the API/UI
input gate before ML inference:
- blocking errors stop impossible/invalid requests;
- warnings preserve risky but scoreable situations for the ML/XAI/RAG layers.

Cloud-readiness: the validator accepts optional reference sets loaded from
``inference_artifacts.json`` today. In AWS/Databricks, those sets can come from
RDS/Feature Store/Unity Catalog without changing the public validation contract.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.rules.schemas import ClaimValidationResult, ValidationIssue

_REQUIRED_FIELDS = ("claim_id", "patient_id", "provider_id")
_OPTIONAL_CODE_FIELDS = ("diagnosis_code", "procedure_code")


def _empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == "" or value.strip().lower() in {"none", "null", "nan"}
    return False


def _clean_id(value: Any) -> str | None:
    if _empty(value):
        return None
    text = str(value).strip()
    return text if text else None


def _clean_code(value: Any) -> str | None:
    if _empty(value):
        return None
    text = str(value).strip().upper()
    return text if text else None


def _coerce_amount(value: Any) -> tuple[float | None, str | None]:
    """Return (amount, error). Missing amount is allowed and returns (None, None)."""
    if _empty(value):
        return None, None
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None, "Billed amount must be numeric when provided."
    if math.isnan(amount) or math.isinf(amount):
        return None, "Billed amount must be a finite number when provided."
    if amount < 0:
        return None, "Billed amount cannot be negative."
    return amount, None


class ClaimInputValidator:
    """Validate and normalize custom claim input before inference."""

    def __init__(
        self,
        *,
        known_provider_ids: set[str] | None = None,
        known_diagnosis_codes: set[str] | None = None,
        known_procedure_codes: set[str] | None = None,
        require_amount: bool = False,
    ) -> None:
        self.known_provider_ids = {str(v) for v in known_provider_ids or set() if str(v).strip()}
        self.known_diagnosis_codes = {str(v).upper() for v in known_diagnosis_codes or set() if str(v).strip()}
        self.known_procedure_codes = {str(v).upper() for v in known_procedure_codes or set() if str(v).strip()}
        self.require_amount = require_amount

    @classmethod
    def from_gold_dir(cls, gold_dir: Path, *, require_amount: bool = False) -> "ClaimInputValidator":
        """Create validator with optional known-code/provider checks from Gold artifacts."""
        path = Path(gold_dir) / "inference_artifacts.json"
        if not path.exists():
            return cls(require_amount=require_amount)
        try:
            artifacts = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return cls(require_amount=require_amount)

        provider_ids = set((artifacts.get("provider_history") or {}).keys())
        diagnosis_codes = set((artifacts.get("diagnosis_lookup") or {}).keys())
        procedure_codes = set((artifacts.get("cost", {}).get("procedure_lookup") or {}).keys())
        return cls(
            known_provider_ids=provider_ids,
            known_diagnosis_codes=diagnosis_codes,
            known_procedure_codes=procedure_codes,
            require_amount=require_amount,
        )

    def validate(self, claim: dict[str, Any] | Any) -> ClaimValidationResult:
        blocking: list[ValidationIssue] = []
        warnings: list[ValidationIssue] = []
        infos: list[ValidationIssue] = []

        if not isinstance(claim, dict):
            issue = ValidationIssue(
                code="CLAIM_ERR_INVALID_BODY",
                field=None,
                message="Claim payload must be a JSON object/dictionary.",
                severity="ERROR",
                blocking=True,
            )
            return ClaimValidationResult(False, {}, [issue], [], [])

        normalized: dict[str, Any] = {}
        for field in _REQUIRED_FIELDS:
            normalized[field] = _clean_id(claim.get(field))
            if normalized[field] is None:
                blocking.append(ValidationIssue(
                    code="CLAIM_ERR_REQUIRED_FIELD_MISSING",
                    field=field,
                    message=f"{field} is required.",
                    severity="ERROR",
                    blocking=True,
                ))

        for field in _OPTIONAL_CODE_FIELDS:
            normalized[field] = _clean_code(claim.get(field))

        amount, amount_error = _coerce_amount(claim.get("billed_amount"))
        normalized["billed_amount"] = amount
        if amount_error:
            blocking.append(ValidationIssue(
                code="CLAIM_ERR_INVALID_AMOUNT",
                field="billed_amount",
                message=amount_error,
                severity="ERROR",
                blocking=True,
            ))
        elif amount is None:
            issue_code = "CLAIM_ERR_AMOUNT_REQUIRED" if self.require_amount else "CLAIM_WARN_AMOUNT_MISSING"
            issue = ValidationIssue(
                code=issue_code,
                field="billed_amount",
                message=(
                    "Billed amount is required."
                    if self.require_amount else
                    "Billed amount is missing. The model can still score the claim using imputation, but review confidence is lower."
                ),
                severity="ERROR" if self.require_amount else "WARNING",
                blocking=self.require_amount,
            )
            (blocking if self.require_amount else warnings).append(issue)

        # Preserve optional context fields for feature builder if supplied.
        if not _empty(claim.get("specialty")):
            normalized["specialty"] = str(claim.get("specialty")).strip().title()
        if not _empty(claim.get("location")):
            normalized["location"] = str(claim.get("location")).strip().title()

        if normalized.get("diagnosis_code") is None:
            warnings.append(ValidationIssue(
                code="CLAIM_WARN_MISSING_DIAGNOSIS",
                field="diagnosis_code",
                message="Diagnosis code is missing. This may weaken medical-necessity support and increase denial risk.",
                severity="WARNING",
                blocking=False,
            ))
        if normalized.get("procedure_code") is None:
            warnings.append(ValidationIssue(
                code="CLAIM_WARN_MISSING_PROCEDURE",
                field="procedure_code",
                message="Procedure code is missing. The billed service is incomplete for review.",
                severity="WARNING",
                blocking=False,
            ))

        if normalized.get("procedure_code") and not normalized.get("diagnosis_code"):
            warnings.append(ValidationIssue(
                code="CLAIM_WARN_PROC_WITHOUT_DIAGNOSIS",
                field="diagnosis_code",
                message="Procedure is present without a supporting diagnosis code.",
                severity="WARNING",
                blocking=False,
            ))
        if normalized.get("diagnosis_code") and not normalized.get("procedure_code"):
            warnings.append(ValidationIssue(
                code="CLAIM_WARN_DIAGNOSIS_WITHOUT_PROC",
                field="procedure_code",
                message="Diagnosis is present without a billed procedure code.",
                severity="WARNING",
                blocking=False,
            ))

        provider_id = normalized.get("provider_id")
        if provider_id and self.known_provider_ids and provider_id not in self.known_provider_ids:
            warnings.append(ValidationIssue(
                code="CLAIM_WARN_UNKNOWN_PROVIDER",
                field="provider_id",
                message="Provider ID was not found in training/reference artifacts. Provider-history features will use safe defaults.",
                severity="WARNING",
                blocking=False,
            ))

        diagnosis_code = normalized.get("diagnosis_code")
        if diagnosis_code and self.known_diagnosis_codes and diagnosis_code not in self.known_diagnosis_codes:
            warnings.append(ValidationIssue(
                code="CLAIM_WARN_UNKNOWN_DIAGNOSIS",
                field="diagnosis_code",
                message="Diagnosis code was not found in reference artifacts. Severity features will use safe defaults.",
                severity="WARNING",
                blocking=False,
            ))

        procedure_code = normalized.get("procedure_code")
        if procedure_code and self.known_procedure_codes and procedure_code not in self.known_procedure_codes:
            warnings.append(ValidationIssue(
                code="CLAIM_WARN_UNKNOWN_PROCEDURE",
                field="procedure_code",
                message="Procedure code was not found in cost benchmark artifacts. Cost features may use safe defaults.",
                severity="WARNING",
                blocking=False,
            ))

        if not blocking:
            infos.append(ValidationIssue(
                code="CLAIM_INFO_INPUT_NORMALIZED",
                field=None,
                message="Claim input passed blocking validation and was normalized for inference.",
                severity="INFO",
                blocking=False,
            ))

        return ClaimValidationResult(
            is_valid=not blocking,
            normalized_claim=normalized,
            blocking_errors=blocking,
            warnings=warnings,
            infos=infos,
        )
